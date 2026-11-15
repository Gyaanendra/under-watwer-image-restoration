#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define MAX_N 16
#define CHUNK_SIZE 128

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

// Warp reduce sum over 16 lanes (half-warp)
__device__ __forceinline__ float reduce_sum_16(float val) {
    unsigned mask = 0xffffu;
    val += __shfl_xor_sync(mask, val, 8, 16);
    val += __shfl_xor_sync(mask, val, 4, 16);
    val += __shfl_xor_sync(mask, val, 2, 16);
    val += __shfl_xor_sync(mask, val, 1, 16);
    return val;
}

// Forward kernel: 1 block of 16 threads per (b, c)
// Shapes:
//   u: (B, D, L)
//   delta: (B, D, L)
//   A: (D, N)
//   B_in: (B, G, N, L)
//   C_in: (B, G, N, L)
//   D_skip: (D)
//   delta_bias: (D)
//   y: (B, D, L)
//   chunk_states: (B, D, N, n_chunks)
__global__ void selective_scan_fwd_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B_in,
    const float* __restrict__ C_in,
    const float* __restrict__ D_skip,
    const float* __restrict__ delta_bias,
    float* __restrict__ y,
    float* __restrict__ chunk_states,
    int B_batch, int D, int G, int N, int L, int n_chunks
) {
    int bc = blockIdx.x;
    int bi = bc / D;
    int c = bc % D;
    int n = threadIdx.x; // 0 .. MAX_N-1 (MAX_N = 16)
    bool active = (n < N);

    if (bi >= B_batch) return;

    int grp_sz = D / G;
    int g = c / grp_sz;

    const float* u_ptr = u + (size_t)bc * L;
    const float* delta_ptr = delta + (size_t)bc * L;
    float* y_ptr = y + (size_t)bc * L;

    const float* B_ptr = active ? (B_in + ((size_t)(bi * G + g) * N + n) * L) : nullptr;
    const float* C_ptr = active ? (C_in + ((size_t)(bi * G + g) * N + n) * L) : nullptr;
    float* cs_ptr = active ? (chunk_states + ((size_t)bc * N + n) * n_chunks) : nullptr;

    float A_val = active ? A[c * N + n] : 0.0f;
    float d_bias = delta_bias[c];
    float d_skip_val = D_skip[c];

    float x = 0.0f;

    for (int t = 0; t < L; ++t) {
        if (active && (t % CHUNK_SIZE) == 0) {
            int chk = t / CHUNK_SIZE;
            cs_ptr[chk] = x;
        }
        float u_t = u_ptr[t];
        float dt_raw = delta_ptr[t] + d_bias;
        float dt = softplus_f(dt_raw);
        float a_t = active ? expf(dt * A_val) : 0.0f;
        float b_t = active ? (dt * B_ptr[t] * u_t) : 0.0f;
        x = active ? (a_t * x + b_t) : 0.0f;

        float contrib = active ? (x * C_ptr[t]) : 0.0f;
        float sum_c = reduce_sum_16(contrib);
        if (n == 0) {
            y_ptr[t] = sum_c + u_t * d_skip_val;
        }
    }
}

// Backward kernel: 1 block of 16 threads per (b, c)
// Note: dB_out and dC_out must be zero-initialized before launch
__global__ void selective_scan_bwd_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B_in,
    const float* __restrict__ C_in,
    const float* __restrict__ D_skip,
    const float* __restrict__ delta_bias,
    const float* __restrict__ chunk_states,
    const float* __restrict__ dy,
    float* __restrict__ du,
    float* __restrict__ ddelta,
    float* __restrict__ dA,
    float* __restrict__ dB_out,
    float* __restrict__ dC_out,
    float* __restrict__ dD_skip,
    float* __restrict__ ddelta_bias,
    int B_batch, int D, int G, int N, int L, int n_chunks
) {
    int bc = blockIdx.x;
    int bi = bc / D;
    int c = bc % D;
    int n = threadIdx.x; // 0 .. MAX_N-1
    bool active = (n < N);

    if (bi >= B_batch) return;

    __shared__ float x_hist[MAX_N][CHUNK_SIZE];      // x_t after update
    __shared__ float x_prev_hist[MAX_N][CHUNK_SIZE]; // x_{t-1} before update

    int grp_sz = D / G;
    int g = c / grp_sz;

    const float* u_ptr = u + (size_t)bc * L;
    const float* delta_ptr = delta + (size_t)bc * L;
    const float* dy_ptr = dy + (size_t)bc * L;
    float* du_ptr = du + (size_t)bc * L;
    float* ddelta_ptr = ddelta + (size_t)bc * L;

    const float* B_ptr = active ? (B_in + ((size_t)(bi * G + g) * N + n) * L) : nullptr;
    const float* C_ptr = active ? (C_in + ((size_t)(bi * G + g) * N + n) * L) : nullptr;
    float* dB_ptr = active ? (dB_out + ((size_t)(bi * G + g) * N + n) * L) : nullptr;
    float* dC_ptr = active ? (dC_out + ((size_t)(bi * G + g) * N + n) * L) : nullptr;
    const float* cs_ptr = active ? (chunk_states + ((size_t)bc * N + n) * n_chunks) : nullptr;

    float A_val = active ? A[c * N + n] : 0.0f;
    float d_bias = delta_bias[c];
    float d_skip_val = D_skip[c];

    float dx_next = 0.0f;
    float dA_acc = 0.0f;
    float dD_acc = 0.0f;
    float ddbias_acc = 0.0f;

    for (int chk = n_chunks - 1; chk >= 0; --chk) {
        int t0 = chk * CHUNK_SIZE;
        int len = L - t0;
        if (len > CHUNK_SIZE) len = CHUNK_SIZE;

        // 1. Forward recompute within chunk
        float x = active ? cs_ptr[chk] : 0.0f;
        for (int s = 0; s < len; ++s) {
            int t = t0 + s;
            float x_prev = x;
            float u_t = u_ptr[t];
            float dt_raw = delta_ptr[t] + d_bias;
            float dt = softplus_f(dt_raw);
            float a_t = active ? expf(dt * A_val) : 0.0f;
            float b_t = active ? (dt * B_ptr[t] * u_t) : 0.0f;
            x = active ? (a_t * x_prev + b_t) : 0.0f;
            x_prev_hist[n][s] = x_prev;
            x_hist[n][s] = x;
        }
        __syncthreads();

        // 2. Backward pass within chunk
        for (int s = len - 1; s >= 0; --s) {
            int t = t0 + s;
            float grad_y = dy_ptr[t];
            float u_t = u_ptr[t];
            float dt_raw = delta_ptr[t] + d_bias;
            float dt = softplus_f(dt_raw);
            float sig_dt = sigmoid_f(dt_raw);
            float a_t = active ? expf(dt * A_val) : 0.0f;
            float B_t = active ? B_ptr[t] : 0.0f;
            float C_t = active ? C_ptr[t] : 0.0f;
            float x_t = x_hist[n][s];
            float x_prev = x_prev_hist[n][s];

            if (n == 0) {
                dD_acc += grad_y * u_t;
            }

            float dx = active ? (dx_next + grad_y * C_t) : 0.0f;
            if (active) {
                atomicAdd(&dC_ptr[t], grad_y * x_t);
                atomicAdd(&dB_ptr[t], dx * dt * u_t);
                dA_acc += dx * x_prev * dt * a_t;
            }

            float du_part = active ? (dx * dt * B_t) : 0.0f;
            float ddt_part = active ? (dx * (x_prev * A_val * a_t + B_t * u_t)) : 0.0f;

            float du_sum = reduce_sum_16(du_part);
            float ddt_sum = reduce_sum_16(ddt_part);

            if (n == 0) {
                du_ptr[t] = du_sum + grad_y * d_skip_val;
                float ddt_raw = ddt_sum * sig_dt;
                ddelta_ptr[t] = ddt_raw;
                ddbias_acc += ddt_raw;
            }

            dx_next = active ? (dx * a_t) : 0.0f;
        }
        __syncthreads();
    }

    if (active) {
        atomicAdd(&dA[c * N + n], dA_acc);
    }
    if (n == 0) {
        atomicAdd(&dD_skip[c], dD_acc);
        atomicAdd(&ddelta_bias[c], ddbias_acc);
    }
}

extern "C" __declspec(dllexport) int launch_ssm_fwd(
    uintptr_t u, uintptr_t delta, uintptr_t A, uintptr_t B_in, uintptr_t C_in,
    uintptr_t D_skip, uintptr_t delta_bias, uintptr_t y, uintptr_t chunk_states,
    int B_batch, int D, int G, int N, int L, int n_chunks, uintptr_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int blocks = B_batch * D;
    selective_scan_fwd_kernel<<<blocks, MAX_N, 0, stream>>>(
        (const float*)u, (const float*)delta, (const float*)A,
        (const float*)B_in, (const float*)C_in, (const float*)D_skip,
        (const float*)delta_bias, (float*)y, (float*)chunk_states,
        B_batch, D, G, N, L, n_chunks
    );
    return (int)cudaGetLastError();
}

extern "C" __declspec(dllexport) int launch_ssm_bwd(
    uintptr_t u, uintptr_t delta, uintptr_t A, uintptr_t B_in, uintptr_t C_in,
    uintptr_t D_skip, uintptr_t delta_bias, uintptr_t chunk_states, uintptr_t dy,
    uintptr_t du, uintptr_t ddelta, uintptr_t dA, uintptr_t dB_out, uintptr_t dC_out,
    uintptr_t dD_skip, uintptr_t ddelta_bias,
    int B_batch, int D, int G, int N, int L, int n_chunks, uintptr_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int blocks = B_batch * D;
    selective_scan_bwd_kernel<<<blocks, MAX_N, 0, stream>>>(
        (const float*)u, (const float*)delta, (const float*)A,
        (const float*)B_in, (const float*)C_in, (const float*)D_skip,
        (const float*)delta_bias, (const float*)chunk_states, (const float*)dy,
        (float*)du, (float*)ddelta, (float*)dA, (float*)dB_out, (float*)dC_out,
        (float*)dD_skip, (float*)ddelta_bias,
        B_batch, D, G, N, L, n_chunks
    );
    return (int)cudaGetLastError();
}

