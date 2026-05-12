# LPCVC 2026 Track 1 — Winning Solution

**Team**: [Your Team Name]  
**Competition**: [2026 Low-Power Computer Vision Challenge (LPCVC)](https://lpcv.ai), Track 1: Image-to-Text Retrieval  
**Result**: 1st Place — Leaderboard Score **0.6122** (Recall@10)

---

## Overview

This repository contains the winning solution for **LPCVC 2026 Track 1**, which targets on-device image-to-text retrieval on the Qualcomm XR2 Gen 2 NPU via [Qualcomm AI Hub](https://aihub.qualcomm.com/).

Our approach fine-tunes **MobileCLIP-B** (pretrained on DataComp-DR) using Gemini-generated referring-expression captions from Visual Genome images, then constructs a **Model Soup** (weight averaging) across fine-tuned checkpoints to maximize generalization.

### Key Techniques

| Technique | Description |
|-----------|-------------|
| **Gemini Caption Fine-tuning** | Fine-tune MobileCLIP-B on ~68K VG images re-captioned by Gemini 2.0 Flash with structured referring expressions |
| **Spatial Hard Negatives** | Automatically swap spatial words (left/right/upper/lower) to generate hard negative pairs during training |
| **Sigmoid GELU** | Replace `x * Φ(1.702x)` for NPU-compatible GELU approximation at export time |
| **Model Soup** | Weight-average two complementary fine-tuned checkpoints to improve generalization |
| **Proxy Metric** | `proxy_v2 = 0.1 × Test_R@10 + 0.9 × Sample_R@10` tracks submission-relevant performance |

---

## Pipeline

```
MobileCLIP-B (datacompdr pretrained)
       │
       ▼
  Fine-tune Run04          Fine-tune Run05
  (VG, lr=5e-6, ep2)      (from soup, lr=3e-6, ep1)
       │                          │
       └──────── Model Soup ──────┘
                 alpha=0.42
                     │
                     ▼
         soup_fresh2_x_s05e1_a042.pt
                 (LB = 0.6122)
                     │
                     ▼
          export_prompt_baked.py
          (ONNX + Sigmoid GELU)
                     │
                     ▼
         Qualcomm AI Hub Compilation
              (XR2 Gen 2 NPU)
```

---

## Repository Structure

```
├── train_gemini_full.py          # Main training script (Gemini captions + VG)
├── train_fullimg.py              # Base training utilities and dataset loaders
├── soup_fresh2_x_s05e1.py        # Model soup search (Run04-ep2 × Run05-ep1)
├── export_prompt_baked.py        # ONNX export with Sigmoid GELU replacement
├── local_eval.py                 # Local PyTorch R@10 evaluation
├── no_cls_utils.py               # Sigmoid GELU, no-CLS pooling utilities
├── compile_image_with_calibration.py  # AI Hub image encoder compilation
├── compile_text_with_calibration.py   # AI Hub text encoder compilation
└── lpcvc2026/
    └── modules/
        ├── data.py               # CompetitionTransform, dataset paths
        ├── evaluate.py           # evaluate_on_sample (proxy metric)
        └── soup.py               # Weight averaging utilities
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (for training)
- Qualcomm AI Hub account (for NPU compilation)

```bash
pip install -r requirements.txt
pip install qai-hub  # from https://aihub.qualcomm.com
```

---

## Reproducibility

### Step 1 — Fine-tune Run04 (fresh, VG, lr=5e-6)

```bash
python train_gemini_full.py --arch MobileCLIP-B --pretrained datacompdr --lr 5e-6 --epochs 5 --batch_size 64 --save_dir models/mobileclipB_04_vg_fresh
```

Use `epoch_2.pt` (epoch 2 checkpoint) as `A_CKPT`.

### Step 2 — Fine-tune Run05 (from soup, VG, lr=3e-6)

First create an intermediate soup from Run04 checkpoints, then:

```bash
python train_gemini_full.py --arch MobileCLIP-B --pretrained datacompdr --lr 3e-6 --epochs 1 --batch_size 64 --resume models/<run04_soup>.pt --save_dir models/mobileclipB_05_vg_from_soup
```

Use `epoch_1.pt` as `B_CKPT`.

### Step 3 — Model Soup

```bash
python soup_fresh2_x_s05e1.py
```

This scans alpha ∈ [0.25, 0.60] and saves the best soup to `models/soup_fresh2_x_s05e1_a042.pt`.

### Step 4 — Export ONNX (Sigmoid GELU)

```bash
python export_prompt_baked.py --model_path models/soup_fresh2_x_s05e1_a042.pt --out_dir onnx_models/winner
```

### Step 5 — Compile on AI Hub (XR2 Gen 2)

```bash
python compile_image_with_calibration.py --onnx_path onnx_models/winner/image_encoder.onnx
python compile_text_with_calibration.py  --onnx_path onnx_models/winner/text_encoder.onnx
```

---

## Model Weights

Pre-trained model checkpoints are available at:

> **[Google Drive / HuggingFace link — TBD]**

| File | Description | MD5 |
|------|-------------|-----|
| `soup_fresh2_x_s05e1_a042.pt` | Best model (LB=0.6122) | — |
| `mobileclipB_04_vg_fresh/epoch_2.pt` | Parent A | — |
| `mobileclipB_05_vg_from_soup/epoch_1.pt` | Parent B | — |

---

## Results

| Checkpoint | Sample R@10 | LB Score |
|------------|------------|---------|
| MobileCLIP-B pretrained (baseline) | ~75% | ~0.50 |
| Run04 epoch_2 (fresh fine-tune) | ~91% | — |
| **soup_fresh2_x_s05e1_a042** | **~92.5%** | **0.6122** |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{lpcvc2026track1,
  title  = {LPCVC 2026 Track 1 Winning Solution: Model Soup for On-Device Image-to-Text Retrieval},
  author = {[Author Names]},
  year   = {2026},
  url    = {https://github.com/[repo]}
}
```

---

## Acknowledgements

- [MobileCLIP](https://github.com/apple/ml-mobileclip) by Apple
- [open_clip](https://github.com/mlfoundations/open_clip) by LAION
- [Qualcomm AI Hub](https://aihub.qualcomm.com) for NPU compilation
- Google Gemini 2.0 Flash for caption generation
