# LPCVC 2026 Track 1 — Solution

**Team**: mangojump
**Members**: Bo-Qing Wu (Chung Yuan Christian University) · Min-Quan Wang (National Yang Ming Chiao Tung University)
**Competition**: [2026 Low-Power Computer Vision Challenge (LPCVC)](https://lpcv.ai), Track 1: Image-to-Text Retrieval
**Leaderboard Score**: 0.6122 (Recall@10)

---

## Overview

This repository contains the winning solution for **LPCVC 2026 Track 1**, which tasks participants with building an image-to-text retrieval model that runs efficiently on the **Qualcomm XR2 Gen 2 NPU** via [Qualcomm AI Hub](https://aihub.qualcomm.com/).

We start from **MobileCLIP-B** pretrained on DataComp-DR and fine-tune it using Gemini-generated referring-expression captions derived from Visual Genome images. The final model is produced by **Model Soup** — a simple weight averaging of two complementary fine-tuned checkpoints — which consistently outperforms any individual checkpoint on the competition metric.

---

## Key Techniques

| Technique | Description |
|-----------|-------------|
| **[0,1] Normalization** | Competition eval pipeline uses `/255.0` only (no CLIP mean/std). Training with correct normalization was critical. |
| **Gemini Caption Fine-tuning** | Fine-tune on ~68K Visual Genome images re-captioned by Gemini 2.0 Flash with structured referring expressions (`data/gemini_captions.json`) |
| **Spatial Hard Negatives** | Auto-generate hard negatives by swapping spatial words (left↔right, upper↔lower) during training |
| **Model Soup** | Weight-average two complementary checkpoints (alpha=0.42) to improve generalization beyond any single run |
| **Sigmoid GELU** | Replace standard GELU with `x × σ(1.702x)` at export time for Qualcomm NPU compatibility |
| **Proxy Metric** | `proxy_v2 = 0.1 × Test_R@10 + 0.9 × Sample_R@10` reliably tracks leaderboard performance |

---

## Full Pipeline

```
MobileCLIP-B/datacompdr (pretrained)
         │
         ▼
 ┌──── Step 1 ────────────────────────────────────┐
 │  finetune.py  (Run04)                          │
 │  VG + Gemini captions, lr=5e-6, 5 epochs       │
 │  → models/run04/epoch_1.pt ~ epoch_5.pt        │
 └────────────────────────────────────────────────┘
         │ epoch_1 ~ epoch_5
         ▼
 ┌──── Step 2 ────────────────────────────────────┐
 │  create_soup.py  (intermediate soup)           │
 │  Scan alpha, pick best Run04 soup checkpoint   │
 │  → models/run04_soup.pt                        │
 └────────────────────────────────────────────────┘
         │
         ▼
 ┌──── Step 3 ────────────────────────────────────┐
 │  finetune.py  (Run05)                          │
 │  Resume from run04_soup, lr=3e-6, 1 epoch      │
 │  → models/run05/epoch_1.pt                     │
 └────────────────────────────────────────────────┘
         │
         │  Run04/epoch_2.pt  ×  Run05/epoch_1.pt
         ▼
 ┌──── Step 4 ────────────────────────────────────┐
 │  create_soup.py  (final soup)                  │
 │  Scan alpha ∈ [0.25, 0.60]                     │
 │  Best: alpha=0.42                              │
 │  → models/soup_fresh2_x_s05e1_a042.pt  ✓      │
 └────────────────────────────────────────────────┘
         │
         ▼
 ┌──── Step 5 ────────────────────────────────────┐
 │  export_onnx.py                                │
 │  PyTorch → ONNX (Sigmoid GELU replacement)     │
 │  → upload to Qualcomm AI Hub                  │
 │  → INT8 compile for XR2 Gen 2 NPU             │
 │  → profile latency                            │
 └────────────────────────────────────────────────┘
```

---

## Key Findings

These are the insights that most impacted our final score, in order of importance:

**1. [0,1] normalization is critical.**
The competition evaluation pipeline normalizes images by `/255.0` only — it does **not** apply the standard CLIP mean/std normalization. Training with incorrect CLIP normalization caused a train/inference mismatch that significantly hurt performance. All models must be trained with `CompetitionTransform` (see `lpcvc2026/modules/data.py`).

**2. Checkpoint diversity beats single-model optimization.**
Model Soup (weight averaging two complementary checkpoints) consistently outperforms any individual checkpoint. The key is that the two parent checkpoints must be *different enough* — Run04 (fresh fine-tune) and Run05 (fine-tuned from an intermediate soup) represent different loss basins, and their average generalizes better than either alone. Fine-tuning a soup checkpoint never helped; it always collapsed the diversity that made the soup work. The mixing ratio alpha=0.42 was selected by scanning alpha ∈ [0.25, 0.60] in steps of 0.02~0.05 and picking the value that maximized `proxy_v2` on the local validation set (`create_soup.py`).

**3. The proxy metric must match the competition sample distribution.**
Our first proxy metric (`proxy_v1 = 0.6×Test_R@10 + 0.4×Sample_R@10`) was misleading: models with higher calibration-test R@10 actually performed *worse* on the real leaderboard. After analysis we found the competition sample set uses longer, relational referring expressions (avg. 5 words, "with" clauses), while the calibration test set has shorter captions. We switched to `proxy_v2 = 0.1×Test_R@10 + 0.9×Sample_R@10`, which aligns correctly with leaderboard ordering.

**4. Prompt baking hurts performance.**
We tested prepending prompt templates ("a photograph of", "a photo of") into the text encoder at export time. All variants degraded the proxy score by 2–4%. The model was fine-tuned on plain captions without prompts, so adding them at inference time creates a distribution mismatch. We use `--no_prompt` for all exports.

**5. Sigmoid GELU is required for NPU deployment.**
Standard GELU uses the error function (`erf`), which is not supported by the Qualcomm Hexagon NPU. We replace all `nn.GELU` instances with `x × σ(1.702x)` at ONNX export time (`npu_utils.py`). This approximation has negligible accuracy impact but is essential for the model to run on-device.

---

## Repository Structure

```
├── finetune.py            # Fine-tune MobileCLIP-B on Gemini captions (Steps 1 & 3)
├── finetune_utils.py      # Dataset loaders, loss functions, training utilities
├── create_soup.py         # Model soup: scan alpha and weight-average checkpoints (Steps 2 & 4)
├── export_onnx.py         # Export PyTorch model to ONNX with Sigmoid GELU (Step 5)
├── evaluate.py            # Local Recall@10 evaluation on competition sample set
├── npu_utils.py           # NPU compatibility utilities (Sigmoid GELU, no-CLS pooling)
├── ptqat_utils.py         # Post-training quantization-aware training utilities
├── generate_captions.py   # Generate Gemini referring-expression captions for COCO images
├── data/
│   └── gemini_captions.json    # Pre-generated captions for 68K COCO train2014 images
├── models/
│   └── soup_fresh2_x_s05e1_a042.pt   # Final model (LB=0.6122) — tracked by Git LFS
├── lpcvc2026/
│   └── modules/
│       ├── data.py        # CompetitionTransform, dataset loaders, path resolution
│       ├── evaluate.py    # evaluate_on_sample() — proxy metric computation
│       └── soup.py        # Weight averaging helpers
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (for fine-tuning)
- Qualcomm AI Hub account (for NPU compilation, Step 5 only)

```bash
pip install -r requirements.txt
pip install qai-hub  # see https://aihub.qualcomm.com
```

---

## Data Setup

### Caption data (included)

`data/gemini_captions.json` is already included in this repo. It maps each COCO image filename to a list of Gemini-generated referring expressions:

```json
{
  "COCO_train2014_000000000009.jpg": [
    "the pink container",
    "the yellow broccoli",
    "a person in a blue jacket holding an umbrella",
    "a dog sitting on a wooden floor near the window"
  ],
  ...
}
```

This file covers ~68K COCO train2014 images and is **already included** — no need to regenerate it. It was produced using `generate_captions.py` with Gemini 2.0 Flash (provided for reference only).

### Image data (download separately)

Training requires the following datasets. Place them relative to the repo root or set the `LPCVC_BASE_DIR` environment variable:

```
<repo_root>/                        ← BASE_DIR (auto-detected)
├── train2014/train2014/            ← COCO train2014 images
│   └── COCO_train2014_*.jpg
├── vg_images/VG_100K/              ← Visual Genome images
│   └── *.jpg
└── region_descriptions.json        ← Visual Genome region descriptions
```

| Dataset | Download | Used for |
|---------|----------|---------|
| [COCO train2014](https://cocodataset.org/#download) | ~13GB | Gemini caption fine-tuning |
| [Visual Genome images](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | ~15GB | Fine-tuning diversity |
| [VG region descriptions](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | ~1GB | VG text annotations |
| RefCOCO/+/g | auto-downloaded via HuggingFace | Proxy metric evaluation |

To use a different data root, set the environment variable:
```bash
export LPCVC_BASE_DIR=/path/to/your/data
```

---

## Score Reproduction (Quickstart)

The final model is included in this repo via Git LFS. To verify the leaderboard score locally:

```bash
python evaluate.py --checkpoint models/soup_fresh2_x_s05e1_a042.pt
# Expected: Sample R@10 ≈ 92.5%
```

To reproduce the full submission (ONNX export + AI Hub compilation):

```bash
# Export ONNX + compile on AI Hub + profile (all-in-one)
python export_onnx.py --checkpoint models/soup_fresh2_x_s05e1_a042.pt --arch MobileCLIP-B --pretrained datacompdr --no_prompt

# Export ONNX only (skip AI Hub steps)
python export_onnx.py --checkpoint models/soup_fresh2_x_s05e1_a042.pt --arch MobileCLIP-B --pretrained datacompdr --no_prompt --export_only
```

`export_onnx.py` handles the full pipeline: ONNX export → Sigmoid GELU replacement → AI Hub compile job → latency profiling. Requires a [Qualcomm AI Hub](https://aihub.qualcomm.com) account.

---

## Training from Scratch

> Training scripts are provided for reference. Due to GPU non-determinism and DataLoader shuffle order, retraining may yield ±0.3% variation. Exact score reproduction is guaranteed via the provided model weights above.

### Step 1 — Fine-tune Run04

```bash
python finetune.py --arch MobileCLIP-B --pretrained datacompdr --lr 5e-6 --epochs 5 --batch_size 64 --save_dir models/run04
```

Use `models/run04/epoch_2.pt` as the Run04 checkpoint (best on proxy).

### Step 2 — Intermediate soup of Run04 checkpoints

Edit `create_soup.py` to point `A_CKPT` / `B_CKPT` at Run04 epoch checkpoints, then:

```bash
python create_soup.py
```

### Step 3 — Fine-tune Run05 (from soup)

```bash
python finetune.py --arch MobileCLIP-B --pretrained datacompdr --lr 3e-6 --epochs 1 --batch_size 64 --resume models/run04_soup.pt --save_dir models/run05
```

### Step 4 — Final soup (Run04-ep2 × Run05-ep1)

Edit `create_soup.py`:
- `A_CKPT = "models/run04/epoch_2.pt"`
- `B_CKPT = "models/run05/epoch_1.pt"`

```bash
python create_soup.py
# Saves best soup → models/soup_fresh2_x_s05e1_a042.pt  (alpha=0.42)
```

### Step 5 — Export ONNX + Compile on AI Hub

```bash
python export_onnx.py --checkpoint models/soup_fresh2_x_s05e1_a042.pt --arch MobileCLIP-B --pretrained datacompdr --no_prompt
```

This exports ONNX with Sigmoid GELU, submits compile and profile jobs to Qualcomm AI Hub, and prints the resulting job IDs for submission.

---

## Results

| Model | Sample R@10 | Leaderboard Score |
|-------|------------|------------------|
| MobileCLIP-B pretrained (baseline) | ~75% | ~0.50 |
| Run04 epoch_2 (fine-tuned) | ~91% | — |
| **Final soup (alpha=0.42)** | **~92.5%** | **0.6122** |

---

## Acknowledgements

- [MobileCLIP](https://github.com/apple/ml-mobileclip) by Apple
- [open_clip](https://github.com/mlfoundations/open_clip) by LAION
- [Qualcomm AI Hub](https://aihub.qualcomm.com) for NPU compilation infrastructure
- Google Gemini 2.0 Flash for caption generation
