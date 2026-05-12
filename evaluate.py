"""
Local evaluation script for LPCVC 2026 Track 1.
Tests models on the sample dataset BEFORE uploading to avoid wasting submissions.

Simulates the competition eval pipeline:
  1. Load images, resize to model input size, normalize to [0,1] then CLIP mean/std
  2. Tokenize texts with CLIP tokenizer (77 tokens)
  3. Compute image & text embeddings
  4. Cosine similarity → Recall@10

Supports testing:
  - PyTorch models (OpenCLIP pretrained) → fastest, no export needed
  - ONNX models → verify export correctness
  - Both with/without attn_mask to diagnose issues

Usage:
    python local_eval.py                                    # Test B-16 datacomp_xl (default)
    python local_eval.py --model ViT-B-16 --pretrained openai
    python local_eval.py --model ViT-B-32 --pretrained openai
    python local_eval.py --onnx_dir onnx_B16_datacomp_v3    # Test ONNX export
    python local_eval.py --all                               # Test all candidates
"""

import os
import sys
import csv
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F


# ── Paths ──────────────────────────────────────────────────────────────────
SAMPLE_DIR = os.path.join(os.environ.get("LPCVC_DATA", "LPCVC_DATA"), "track1_sample", "sample")
IMAGE_DIR = os.path.join(SAMPLE_DIR, "images", "default")
IMG2TXT_CSV = os.path.join(SAMPLE_DIR, "SampleDataset_Image_to_Textnums 1.csv")
TXT2NUM_CSV = os.path.join(SAMPLE_DIR, "SampleDataset_Textnums_to_Texts 1.csv")

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def load_sample_dataset():
    """Load sample dataset: images → text_nums, text_num → text."""

    # text_num → text
    num2text = {}
    with open(TXT2NUM_CSV, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if row[0] == 'Text_nums':
                continue
            num2text[int(row[0])] = row[1]

    # image → [text_nums]
    img2nums = {}
    with open(IMG2TXT_CSV, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if row[0] == 'Image_names':
                continue
            img2nums[row[0]] = [int(x) for x in row[1].split(';')]

    # Collect all unique texts (ordered by text_num)
    all_text_nums = sorted(set(n for nums in img2nums.values() for n in nums))
    # Map text_num → index in our text array
    num2idx = {n: i for i, n in enumerate(all_text_nums)}
    all_texts = [num2text.get(n, f"MISSING:{n}") for n in all_text_nums]

    # Ground truth: for each image, which text indices are correct
    image_names = list(img2nums.keys())
    gt = []  # list of sets
    for img_name in image_names:
        correct_indices = set(num2idx[n] for n in img2nums[img_name])
        gt.append(correct_indices)

    return image_names, all_texts, gt


def preprocess_image_clip(image_path, image_size=224):
    """
    Preprocess image matching the competition eval pipeline:
    1. Resize to (image_size, image_size)
    2. Convert to float32, divide by 255 → [0, 1]
    3. Apply CLIP mean/std normalization
    4. Return (1, 3, H, W) tensor
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    img = np.array(img, dtype=np.float32) / 255.0  # [0, 1]

    # CLIP normalization: (x - mean) / std
    for c in range(3):
        img[:, :, c] = (img[:, :, c] - CLIP_MEAN[c]) / CLIP_STD[c]

    # HWC → CHW → BCHW
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def compute_recall_at_k(similarity_matrix, ground_truth, k=10):
    """
    Compute Recall@K for image-to-text retrieval (fractional recall).

    For each image, recall = (# of correct texts found in top-k) / (# of total correct texts).
    Final score = average recall across all images.

    This matches the competition metric: Recall@Top10.

    similarity_matrix: (num_images, num_texts) cosine similarity
    ground_truth: list of sets, gt[i] = set of correct text indices for image i
    k: retrieve top-k

    Returns: mean fractional recall across all images
    """
    num_images = similarity_matrix.shape[0]
    recalls = []
    for i in range(num_images):
        gt_set = ground_truth[i]
        topk_indices = set(similarity_matrix[i].argsort(descending=True)[:k].tolist())
        matched = len(gt_set & topk_indices)
        recalls.append(matched / len(gt_set))
    return sum(recalls) / num_images


def eval_pytorch_model(model_name, pretrained, image_size=224):
    """Evaluate a pretrained OpenCLIP model using PyTorch (ground truth)."""
    import open_clip

    print(f"\n{'='*70}")
    print(f"  PyTorch Eval: {model_name} / {pretrained}")
    print(f"{'='*70}")

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Load dataset
    image_names, all_texts, gt = load_sample_dataset()
    print(f"  Images: {len(image_names)}, Texts: {len(all_texts)}")

    # Encode images
    print("  Encoding images...")
    image_embeddings = []
    with torch.no_grad():
        for img_name in image_names:
            img_path = os.path.join(IMAGE_DIR, img_name)
            img_tensor = preprocess_image_clip(img_path, image_size)
            emb = model.encode_image(img_tensor, normalize=False)
            image_embeddings.append(emb)
    image_embeddings = torch.cat(image_embeddings, dim=0)

    # Encode texts
    print("  Encoding texts...")
    text_tokens = tokenizer(all_texts)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens, normalize=False)

    # L2 normalize (cosine similarity)
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Similarity matrix
    similarity = image_embeddings @ text_embeddings.T

    # Recall@K
    r1 = compute_recall_at_k(similarity, gt, k=1)
    r5 = compute_recall_at_k(similarity, gt, k=5)
    r10 = compute_recall_at_k(similarity, gt, k=10)

    print(f"\n  Results ({model_name} / {pretrained}):")
    print(f"    Recall@1:  {r1:.4f} ({r1*100:.1f}%)")
    print(f"    Recall@5:  {r5:.4f} ({r5*100:.1f}%)")
    print(f"    Recall@10: {r10:.4f} ({r10*100:.1f}%)")

    # Also test with OpenCLIP's own preprocessing (to compare)
    print("\n  [Comparison] Using OpenCLIP's built-in preprocess...")
    image_embeddings2 = []
    with torch.no_grad():
        for img_name in image_names:
            img_path = os.path.join(IMAGE_DIR, img_name)
            img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
            emb = model.encode_image(img, normalize=False)
            image_embeddings2.append(emb)
    image_embeddings2 = F.normalize(torch.cat(image_embeddings2, dim=0), dim=-1)
    sim2 = image_embeddings2 @ text_embeddings.T
    r10_openclip = compute_recall_at_k(sim2, gt, k=10)
    print(f"    Recall@10 (OpenCLIP preprocess): {r10_openclip:.4f} ({r10_openclip*100:.1f}%)")

    return r10


def eval_onnx_model(onnx_dir, image_size=224, context_length=77):
    """Evaluate ONNX-exported model to verify export correctness."""
    import onnxruntime as ort
    import open_clip

    print(f"\n{'='*70}")
    print(f"  ONNX Eval: {onnx_dir}")
    print(f"{'='*70}")

    # Find ONNX files
    img_onnx = None
    txt_onnx = None
    for sub in ['img_dir', '']:
        p = os.path.join(onnx_dir, sub, 'image_encoder.onnx')
        if os.path.exists(p):
            img_onnx = p
            break
    for sub in ['txt_dir', '']:
        p = os.path.join(onnx_dir, sub, 'text_encoder.onnx')
        if os.path.exists(p):
            txt_onnx = p
            break

    if not img_onnx or not txt_onnx:
        # Try root level
        img_onnx = os.path.join(onnx_dir, 'image_encoder.onnx')
        txt_onnx = os.path.join(onnx_dir, 'text_encoder.onnx')

    if not os.path.exists(img_onnx) or not os.path.exists(txt_onnx):
        print(f"  ERROR: Cannot find ONNX files in {onnx_dir}")
        return None

    print(f"  Image ONNX: {img_onnx}")
    print(f"  Text  ONNX: {txt_onnx}")

    sess_img = ort.InferenceSession(img_onnx, providers=["CPUExecutionProvider"])
    sess_txt = ort.InferenceSession(txt_onnx, providers=["CPUExecutionProvider"])

    # Check input/output names
    img_input = sess_img.get_inputs()[0]
    txt_input = sess_txt.get_inputs()[0]
    print(f"  Image input: {img_input.name} {img_input.shape} {img_input.type}")
    print(f"  Text  input: {txt_input.name} {txt_input.shape} {txt_input.type}")

    # Determine image size from ONNX model
    if img_input.shape and len(img_input.shape) == 4:
        onnx_img_size = img_input.shape[-1]
        if isinstance(onnx_img_size, int):
            image_size = onnx_img_size
            print(f"  Using image_size={image_size} from ONNX model")

    # Determine context length from ONNX model
    if txt_input.shape and len(txt_input.shape) == 2:
        onnx_ctx = txt_input.shape[-1]
        if isinstance(onnx_ctx, int):
            context_length = onnx_ctx
            print(f"  Using context_length={context_length} from ONNX model")

    # Tokenizer (always use standard CLIP tokenizer for competition)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Load dataset
    image_names, all_texts, gt = load_sample_dataset()
    print(f"  Images: {len(image_names)}, Texts: {len(all_texts)}")

    # Encode images
    print("  Encoding images (ONNX)...")
    image_embeddings = []
    for img_name in image_names:
        img_path = os.path.join(IMAGE_DIR, img_name)
        img_tensor = preprocess_image_clip(img_path, image_size).numpy()
        result = sess_img.run(None, {img_input.name: img_tensor})
        image_embeddings.append(result[0])
    image_embeddings = torch.from_numpy(np.concatenate(image_embeddings, axis=0))

    # Encode texts
    print("  Encoding texts (ONNX)...")
    text_tokens = tokenizer(all_texts).numpy().astype(np.int64)
    # Truncate/pad to context_length if needed
    if text_tokens.shape[1] != context_length:
        if text_tokens.shape[1] > context_length:
            text_tokens = text_tokens[:, :context_length]
        else:
            text_tokens = np.pad(text_tokens, ((0, 0), (0, context_length - text_tokens.shape[1])))

    text_embeddings = []
    for i in range(len(all_texts)):
        tok = text_tokens[i:i+1]
        result = sess_txt.run(None, {txt_input.name: tok})
        text_embeddings.append(result[0])
    text_embeddings = torch.from_numpy(np.concatenate(text_embeddings, axis=0))

    # L2 normalize
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Similarity
    similarity = image_embeddings @ text_embeddings.T

    # Recall
    r1 = compute_recall_at_k(similarity, gt, k=1)
    r5 = compute_recall_at_k(similarity, gt, k=5)
    r10 = compute_recall_at_k(similarity, gt, k=10)

    print(f"\n  Results (ONNX: {onnx_dir}):")
    print(f"    Recall@1:  {r1:.4f} ({r1*100:.1f}%)")
    print(f"    Recall@5:  {r5:.4f} ({r5*100:.1f}%)")
    print(f"    Recall@10: {r10:.4f} ({r10*100:.1f}%)")

    return r10


def main():
    parser = argparse.ArgumentParser(description="Local evaluation for LPCVC 2026 Track 1")
    parser.add_argument("--model", type=str, default="ViT-B-16",
                        help="OpenCLIP model name (e.g., ViT-B-16, ViT-B-32)")
    parser.add_argument("--pretrained", type=str, default="datacomp_xl_s13b_b90k",
                        help="Pretrained weights (e.g., openai, datacomp_xl_s13b_b90k)")
    parser.add_argument("--onnx_dir", type=str, default=None,
                        help="Path to ONNX export directory")
    parser.add_argument("--all", action="store_true",
                        help="Test all candidate models")
    args = parser.parse_args()

    print("=" * 70)
    print("  LPCVC 2026 Track 1 — Local Evaluation")
    print("  Sample dataset: 56 images, 211 texts")
    print("=" * 70)

    results = {}

    if args.all:
        # Test all candidates
        candidates = [
            ("ViT-B-32", "openai", 224),
            ("ViT-B-16", "openai", 224),
            ("ViT-B-16", "datacomp_xl_s13b_b90k", 224),
            ("ViT-B-16", "dfn2b", 224),
        ]
        for model_name, pretrained, img_size in candidates:
            try:
                r10 = eval_pytorch_model(model_name, pretrained, img_size)
                results[f"{model_name}/{pretrained}"] = r10
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"{model_name}/{pretrained}"] = None

        # Also test ONNX exports if they exist
        onnx_dirs = [
            "onnx_B16_datacomp_v3",
            "onnx_B16_datacomp",
            "0302-2",
        ]
        for d in onnx_dirs:
            if os.path.exists(d):
                try:
                    r10 = eval_onnx_model(d)
                    results[f"ONNX:{d}"] = r10
                except Exception as e:
                    print(f"  ERROR: {e}")

        # Summary
        print(f"\n{'='*70}")
        print(f"  SUMMARY — All Candidates")
        print(f"{'='*70}")
        for name, r10 in sorted(results.items(), key=lambda x: x[1] or 0, reverse=True):
            if r10 is not None:
                print(f"    {r10:.4f} ({r10*100:.1f}%)  {name}")
            else:
                print(f"    FAILED       {name}")

    elif args.onnx_dir:
        eval_onnx_model(args.onnx_dir)

    else:
        eval_pytorch_model(args.model, args.pretrained)


if __name__ == "__main__":
    main()
