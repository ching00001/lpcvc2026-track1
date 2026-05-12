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
 │  PyTorch → ONNX, replace GELU → Sigmoid GELU  │
 │  → onnx_models/image_encoder.onnx             │
 │  → onnx_models/text_encoder.onnx              │
 └────────────────────────────────────────────────┘
         │
         ▼
 ┌──── Step 6 ────────────────────────────────────┐
 │  compile_image.py / compile_text.py            │
 │  Upload ONNX to Qualcomm AI Hub                │
 │  INT8 quantization + XR2 Gen 2 compilation     │
 └────────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── finetune.py           # Fine-tune MobileCLIP-B on Gemini captions (Steps 1 & 3)
├── finetune_utils.py     # Dataset loaders, loss functions, training utilities
├── create_soup.py        # Model soup: scan alpha and weight-average checkpoints (Steps 2 & 4)
├── export_onnx.py        # Export PyTorch model to ONNX with Sigmoid GELU (Step 5)
├── evaluate.py           # Local Recall@10 evaluation on competition sample set
├── npu_utils.py          # NPU compatibility utilities (Sigmoid GELU, no-CLS pooling)
├── ptqat_utils.py        # Post-training quantization-aware training utilities
├── compile_image.py      # Compile image encoder on Qualcomm AI Hub (Step 6)
├── compile_text.py       # Compile text encoder on Qualcomm AI Hub (Step 6)
├── data/
│   └── gemini_captions.json   # 68K Visual Genome images with Gemini-generated captions
├── models/
│   └── soup_fresh2_x_s05e1_a042.pt   # Final model (LB=0.6122) — tracked by Git LFS
├── lpcvc2026/
│   └── modules/
│       ├── data.py       # CompetitionTransform, dataset path resolution
│       ├── evaluate.py   # evaluate_on_sample() — proxy metric computation
│       └── soup.py       # Weight averaging helpers
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (for fine-tuning)
- Qualcomm AI Hub account (for NPU compilation, Step 6 only)

```bash
pip install -r requirements.txt
pip install qai-hub  # see https://aihub.qualcomm.com
```

**External datasets** (download separately, not included in this repo):
- [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) — images for fine-tuning
- [COCO 2014](https://cocodataset.org/) — train/val images used in proxy evaluation

---

## Score Reproduction (Quickstart)

The final model is included in this repo via Git LFS. To verify the leaderboard score locally:

```bash
python evaluate.py --checkpoint models/soup_fresh2_x_s05e1_a042.pt
# Expected: Sample R@10 ≈ 92.5%
```

To reproduce the full submission (ONNX export + AI Hub compilation):

```bash
# Step 5: Export to ONNX
python export_onnx.py --checkpoint models/soup_fresh2_x_s05e1_a042.pt --arch MobileCLIP-B --pretrained datacompdr --no_prompt --export_only

# Step 6: Compile on AI Hub (requires qai-hub account)
python compile_image.py --onnx_dir onnx_models/models_soup_fresh2_x_s05e1_a042
python compile_text.py  --onnx_dir onnx_models/models_soup_fresh2_x_s05e1_a042
```

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
