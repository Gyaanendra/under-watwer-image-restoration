# 🚀 Post-Baseline Experimental Suite & Execution Guide

**Semester 5 Project — Underwater Image Enhancement**

This directory hosts the automated, modular experiment infrastructure implementing all **16 model variants (19 runs with Adaptive Gate)** derived from the post-baseline design specification.

---

## 📂 File Architecture

| File | Purpose |
| :--- | :--- |
| [`scripts/experiments/modules.py`](file:///e:/sem5_project/scripts/experiments/modules.py) | Standalone implementations of **SCA**, **RowColSpatialGate**, **SimpleGate**, **AdaptiveGate**, **LayerNorm2d**, **ResidualScale**, and **Pooled1DConvScan**. |
| [`scripts/experiments/models_factory.py`](file:///e:/sem5_project/scripts/experiments/models_factory.py) | Dynamic parametric builders for **WaterMamba** and **Hero-Mamba** with configurable hooks for attention, activation, normalization, residual scale, and scan modes. |
| [`scripts/experiments/metrics_tracker.py`](file:///e:/sem5_project/scripts/experiments/metrics_tracker.py) | **PSNR**, **SSIM**, **Mean-Matched PSNR**, **ExperimentTeeLogger**, GPU Step-1 VRAM guardrail (7.0 GB threshold), and CSV summary updater. |
| [`scripts/experiments/train_experiment.py`](file:///e:/sem5_project/scripts/experiments/train_experiment.py) | Single-experiment training harness with automated validation, TeeLogging, and checkpoint saving. |
| [`scripts/experiments/run_matrix.py`](file:///e:/sem5_project/scripts/experiments/run_matrix.py) | Master orchestrator with **pause & resume**, single-run, or group-level execution. |

---

## ⚡ Quick-Start Commands

### 1. View Status Matrix
Check which experiments are pending vs completed at any time:
```powershell
python scripts/experiments/run_matrix.py --list
```

### 2. Run All Experiments (with Pause & Resume)
Run the entire suite sequentially. You can press `Ctrl+C` at any time to pause; re-running the command automatically skips already completed runs!
```powershell
python scripts/experiments/run_matrix.py
```

### 3. Run by Specific Experimental Group
You can run any group independently:

#### Group A: Attention Variations (6 runs)
```powershell
python scripts/experiments/run_matrix.py --group A
```
*Models*: `W-1a`, `W-1b`, `W-1c`, `H-1a`, `H-1b`, `H-1c`

#### Group B: NAFNet & Adaptive Gate Variations (7 runs)
```powershell
python scripts/experiments/run_matrix.py --group B
```
*Models*: `W-N1-SG`, `W-N1-Ada`, `H-N1-SG`, `H-N1-Ada`, `H-D2`, `H-D3-SG`, `H-D3-Ada`

#### Group D: Structural & Crop Ablations (6 runs)
```powershell
python scripts/experiments/run_matrix.py --group D
```
*Models*: `W-R2`, `H-R2`, `W-R0` (64×64 full scan), `W-R1` (64×64 pooled), `H-R0`, `H-R1`

---

## 🎯 Running Any Experiment Individually

You can launch any single model with a single flag:

```powershell
# Run WaterMamba with Simplified Channel Attention (W-1a)
python scripts/experiments/run_matrix.py --run W-1a

# Run WaterMamba with Row/Col Gate (W-1b)
python scripts/experiments/run_matrix.py --run W-1b

# Run Hero-Mamba with Channel Attention (H-1a)
python scripts/experiments/run_matrix.py --run H-1a

# Run WaterMamba with Simple Gate (W-N1-SG)
python scripts/experiments/run_matrix.py --run W-N1-SG

# Run WaterMamba with Adaptive Gate (W-N1-Ada)
python scripts/experiments/run_matrix.py --run W-N1-Ada

# Run Hero-Mamba with NAFNet LayerNorm Decoder (H-D2)
python scripts/experiments/run_matrix.py --run H-D2

# Run Hero-Mamba with LayerNorm + Adaptive Gate + Residual Scales (H-D3-Ada)
python scripts/experiments/run_matrix.py --run H-D3-Ada

# Run 1D Conv SSM-Free Ablation (W-R2)
python scripts/experiments/run_matrix.py --run W-R2
```

*(Optional: add `--force` to re-run an experiment that already completed)*:
```powershell
python scripts/experiments/run_matrix.py --run W-1a --force
```

---

## 🛠️ Direct Single-Command Training (`train_experiment.py`)

If you want custom hyperparameters (e.g., custom learning rate, different epochs, custom subset):

```powershell
python scripts/experiments/train_experiment.py `
  --run_id W-1a `
  --group A `
  --model wm `
  --keep_ccoss 0 `
  --attn sca `
  --act base `
  --res 128 `
  --subset 0.25 `
  --epochs 5 `
  --batch_size 4
```

---

## 📊 Outputs & Tracking

1. **Live Logs**:
   Dual console + disk logs are saved to [`logs/experiments/<run_id>_<timestamp>.log`](file:///e:/sem5_project/logs/experiments/).
2. **Central Matrix Summary**:
   Results auto-append to [`logs/experiments/experiment_matrix.csv`](file:///e:/sem5_project/logs/experiments/experiment_matrix.csv) containing:
   - `run_id`, `params_M`, `peak_vram_mb`, `best_val_psnr`, `best_val_ssim`, `best_val_psnr_mm`, `final_train_loss`, `avg_sec_per_epoch`, `status`.
3. **Checkpoints**:
   Optimal model weights saved to [`checkpoints/experiments/<run_id>/best_model.pth`](file:///e:/sem5_project/checkpoints/experiments/).
