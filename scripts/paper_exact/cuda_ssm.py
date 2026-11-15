"""
Runtime CUDA S6 Selective Scan Loader & Autograd Wrapper for Windows (RTX 4060).
Compiles 'selective_scan_cuda.cu' directly with nvcc + MSVC into a standalone DLL
without Python/PyTorch C++ ABI version conflicts, preserving exact mathematical
equivalence to selective_scan_ref while delivering native GPU register speed.
"""

import os
import ctypes
import shutil
import subprocess
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_CU_PATH = _DIR / "selective_scan_cuda.cu"
_DLL_PATH = _DIR / "selective_scan_cuda.dll"
_CHUNK_SIZE = 128
_MAX_N = 16

_LIB = None


def _find_msvc_and_sdk():
    """Locate MSVC toolchain and Windows 10/11 SDK headers & libs."""
    vs_candidates = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
    ]
    msvc_dir = None
    for root in vs_candidates:
        if root.exists():
            vers = sorted([p for p in root.iterdir() if p.is_dir()])
            if vers:
                msvc_dir = vers[-1]
                break

    if msvc_dir is None:
        raise FileNotFoundError("Could not find Visual Studio 2022 MSVC toolchain.")

    sdk_inc_root = Path(r"C:\Program Files (x86)\Windows Kits\10\Include")
    sdk_lib_root = Path(r"C:\Program Files (x86)\Windows Kits\10\Lib")
    inc_vers = sorted([p for p in sdk_inc_root.iterdir() if p.is_dir()]) if sdk_inc_root.exists() else []
    lib_vers = sorted([p for p in sdk_lib_root.iterdir() if p.is_dir()]) if sdk_lib_root.exists() else []
    if not inc_vers or not lib_vers:
        raise FileNotFoundError("Could not find Windows Kits 10 Include/Lib directories.")

    return msvc_dir, inc_vers[-1], lib_vers[-1]


def _find_nvcc():
    """Locate nvcc executable from PATH, CUDA_HOME, or standard toolkit paths."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cand = Path(cuda_home) / "bin" / "nvcc.exe"
        if cand.exists():
            return str(cand)
    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.exists():
        vers = sorted([p for p in toolkit_root.iterdir() if p.is_dir()])
        for v in reversed(vers):
            cand = v / "bin" / "nvcc.exe"
            if cand.exists():
                return str(cand)
    raise FileNotFoundError("Could not find nvcc.exe in PATH or CUDA Toolkit.")


def _detect_sm_arch():
    """Detect active GPU compute capability (defaults to sm_89 for RTX 4060)."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    return "sm_89"


def _load_or_compile_dll():
    global _LIB
    if _LIB is not None:
        return _LIB

    need_compile = (not _DLL_PATH.exists()) or (_CU_PATH.stat().st_mtime > _DLL_PATH.stat().st_mtime)
    if need_compile:
        msvc_dir, sdk_inc, sdk_lib = _find_msvc_and_sdk()
        nvcc = _find_nvcc()
        arch = _detect_sm_arch()

        env = os.environ.copy()
        cuda_bin = str(Path(nvcc).parent)
        env["PATH"] = f"{msvc_dir}\\bin\\Hostx64\\x64;{cuda_bin};" + env.get("PATH", "")
        env["INCLUDE"] = f"{msvc_dir}\\include;{sdk_inc}\\ucrt;{sdk_inc}\\um;{sdk_inc}\\shared"
        env["LIB"] = f"{msvc_dir}\\lib\\x64;{sdk_lib}\\ucrt\\x64;{sdk_lib}\\um\\x64"

        cmd = [
            nvcc, "--use-local-env", "-O3", f"-arch={arch}",
            "--shared", str(_CU_PATH), "-o", str(_DLL_PATH)
        ]
        print(f"[CUDA-SSM] Compiling {arch} selective_scan kernel -> {_DLL_PATH.name} ...")
        subprocess.check_call(cmd, env=env)
        print(f"[CUDA-SSM] Compilation successful.")

    lib = ctypes.CDLL(str(_DLL_PATH))
    lib.launch_ssm_fwd.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
    ]
    lib.launch_ssm_fwd.restype = ctypes.c_int

    lib.launch_ssm_bwd.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
    ]
    lib.launch_ssm_bwd.restype = ctypes.c_int

    _LIB = lib
    return _LIB


class SelectiveScanCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D_skip, delta_bias):
        lib = _load_or_compile_dll()

        u_f = u.float().contiguous()
        delta_f = delta.float().contiguous()
        A_f = A.float().contiguous()
        B_f = B.float().contiguous()
        C_f = C.float().contiguous()
        D_f = D_skip.float().contiguous()
        db_f = delta_bias.float().contiguous()

        b, d_total, L = u_f.shape
        g = B_f.shape[1]
        N = A_f.shape[1]
        n_chunks = (L + _CHUNK_SIZE - 1) // _CHUNK_SIZE

        y = torch.empty_like(u_f)
        chunk_states = torch.empty((b, d_total, N, n_chunks), device=u_f.device, dtype=torch.float32)

        stream = torch.cuda.current_stream(u_f.device).cuda_stream
        err = lib.launch_ssm_fwd(
            u_f.data_ptr(), delta_f.data_ptr(), A_f.data_ptr(), B_f.data_ptr(), C_f.data_ptr(),
            D_f.data_ptr(), db_f.data_ptr(), y.data_ptr(), chunk_states.data_ptr(),
            b, d_total, g, N, L, n_chunks, stream
        )
        if err != 0:
            raise RuntimeError(f"CUDA selective_scan_fwd failed with error code {err}")

        ctx.save_for_backward(u_f, delta_f, A_f, B_f, C_f, D_f, db_f, chunk_states)
        ctx.orig_dtypes = (u.dtype, delta.dtype, A.dtype, B.dtype, C.dtype, D_skip.dtype, delta_bias.dtype)
        return y.to(u.dtype)

    @staticmethod
    def backward(ctx, dy):
        lib = _load_or_compile_dll()
        u_f, delta_f, A_f, B_f, C_f, D_f, db_f, chunk_states = ctx.saved_tensors
        dy_f = dy.float().contiguous()

        b, d_total, L = u_f.shape
        g = B_f.shape[1]
        N = A_f.shape[1]
        n_chunks = chunk_states.shape[-1]

        du = torch.empty_like(u_f)
        ddelta = torch.empty_like(delta_f)
        dA = torch.zeros_like(A_f)
        dB = torch.zeros_like(B_f)
        dC = torch.zeros_like(C_f)
        dD = torch.zeros_like(D_f)
        ddb = torch.zeros_like(db_f)

        stream = torch.cuda.current_stream(u_f.device).cuda_stream
        err = lib.launch_ssm_bwd(
            u_f.data_ptr(), delta_f.data_ptr(), A_f.data_ptr(), B_f.data_ptr(), C_f.data_ptr(),
            D_f.data_ptr(), db_f.data_ptr(), chunk_states.data_ptr(), dy_f.data_ptr(),
            du.data_ptr(), ddelta.data_ptr(), dA.data_ptr(), dB.data_ptr(), dC.data_ptr(),
            dD.data_ptr(), ddb.data_ptr(),
            b, d_total, g, N, L, n_chunks, stream
        )
        if err != 0:
            raise RuntimeError(f"CUDA selective_scan_bwd failed with error code {err}")

        dt = ctx.orig_dtypes
        return (
            du.to(dt[0]), ddelta.to(dt[1]), dA.to(dt[2]),
            dB.to(dt[3]), dC.to(dt[4]), dD.to(dt[5]), ddb.to(dt[6])
        )


def selective_scan_fast(u, delta, A, B, C, D_skip, delta_bias):
    """Exact CUDA selective scan with safe fallback if on CPU or N > 16."""
    if u.is_cuda and A.shape[1] <= _MAX_N:
        return SelectiveScanCudaFn.apply(u, delta, A, B, C, D_skip, delta_bias)
    from scripts.waterMamba.watermamba_model import selective_scan_ref
    return selective_scan_ref(u, delta, A, B, C, D_skip, delta_bias)


def verify_exactness():
    """Verifies forward & backward numerical equivalence against selective_scan_ref."""
    from scripts.waterMamba.watermamba_model import selective_scan_ref
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    B_sz, G_sz, d_g, N_sz, L_sz = 2, 4, 4, 16, 200
    D_sz = G_sz * d_g
    u = torch.randn(B_sz, D_sz, L_sz, device="cuda", requires_grad=True)
    delta = torch.randn(B_sz, D_sz, L_sz, device="cuda", requires_grad=True)
    A = -torch.exp(torch.randn(D_sz, N_sz, device="cuda")).detach().requires_grad_(True)
    B_in = torch.randn(B_sz, G_sz, N_sz, L_sz, device="cuda", requires_grad=True)
    C_in = torch.randn(B_sz, G_sz, N_sz, L_sz, device="cuda", requires_grad=True)
    D_skip = torch.randn(D_sz, device="cuda", requires_grad=True)
    db = torch.randn(D_sz, device="cuda", requires_grad=True)

    y_ref = selective_scan_ref(u, delta, A, B_in, C_in, D_skip, db)
    g_ref = torch.autograd.grad(y_ref.sum(), [u, delta, A, B_in, C_in, D_skip, db])

    y_fast = selective_scan_fast(u, delta, A, B_in, C_in, D_skip, db)
    g_fast = torch.autograd.grad(y_fast.sum(), [u, delta, A, B_in, C_in, D_skip, db])

    fwd_diff = (y_ref - y_fast).abs().max().item()
    grad_rel = max(
        ((a - b).abs().max() / (a.abs().max() + 1e-6)).item()
        for a, b in zip(g_ref, g_fast)
    )
    print(f"[CUDA-SSM Self-Check] Fwd max diff: {fwd_diff:.2e} | Bwd max rel diff: {grad_rel:.2e}")
    assert fwd_diff < 1e-3 and grad_rel < 1e-3, f"CUDA SSM numerical mismatch! fwd={fwd_diff}, rel={grad_rel}"


def install_fast_selective_scan(verify=True):
    """Injects the fast CUDA selective scan into WaterMamba and HeroMamba modules in memory."""
    import scripts.waterMamba.watermamba_model as wm_mod
    import scripts.HeroMamba.heromamba_model as hm_mod

    _load_or_compile_dll()
    if verify:
        verify_exactness()

    wm_mod.selective_scan = selective_scan_fast
    hm_mod.selective_scan = selective_scan_fast
    print("[CUDA-SSM] Fast CUDA selective scan wired into WaterMamba & HeroMamba (scan_mode='1d').")
