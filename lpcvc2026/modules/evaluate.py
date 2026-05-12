"""
lpcvc2026.modules.evaluate
==========================
Evaluation utilities for LPCVC 2026 Track 1.

Exports:
  evaluate_proxy        - Proxy val set (RefCOCO/VG, ~300 imgs)
  evaluate_on_sample    - Official sample set (56 images)
  evaluate_on_gemini_val - Gemini val split evaluation
  evaluate_on_custom_eval - Custom VG-based eval set (11 images, expandable)
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import BASE_DIR, DATA_ROOT, data_path, CompetitionTransform


# ── Proxy Evaluation ───────────────────────────────────────────────────────────
def evaluate_proxy(model, tokenizer, val_images, val_texts, val_gt,
                   prompt_prefix="", device='cuda', batch_size=64):
    """Evaluate on proxy validation set. Returns R@1, R@5, R@10 (%).

    Scoring: Σ(|matched ∩ top_k| / |gt|) / n_images  (fractional, not binary)
    """
    transform = CompetitionTransform(224)
    prefixed_texts = [prompt_prefix + t for t in val_texts]

    model.eval()
    with torch.no_grad():
        text_feats = []
        for i in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[i:i + batch_size]
            tokens = tokenizer(batch).to(device)
            feats = model.encode_text(tokens)
            text_feats.append(F.normalize(feats.float(), dim=-1).cpu())
        text_feats = torch.cat(text_feats, dim=0)

        image_feats = []
        for i in range(0, len(val_images), batch_size):
            batch_imgs = []
            for p in val_images[i:i + batch_size]:
                img = Image.open(p).convert('RGB')
                batch_imgs.append(transform(img))
            batch_t = torch.stack(batch_imgs).to(device)
            feats = model.encode_image(batch_t)
            image_feats.append(F.normalize(feats.float(), dim=-1).cpu())
        image_feats = torch.cat(image_feats, dim=0)

    sims = image_feats @ text_feats.t()
    recalls = {1: 0.0, 5: 0.0, 10: 0.0}
    for i in range(len(val_images)):
        gt_indices = val_gt[i]
        n_gt = len(gt_indices)
        ranked = sims[i].argsort(descending=True).tolist()
        for k in [1, 5, 10]:
            top_k = set(ranked[:k])
            matched = len(gt_indices & top_k)
            recalls[k] += matched / n_gt

    n = len(val_images)
    return recalls[1] / n * 100, recalls[5] / n * 100, recalls[10] / n * 100


# ── Sample Evaluation ──────────────────────────────────────────────────────────
def evaluate_on_sample(model, tokenizer, prompt_prefix="", device='cuda',
                       img_size=224, text_len=77):
    """Evaluate on the official 56-image sample set.

    Returns R@1, R@5, R@10 (%).
    Reads CSVs from track1_sample/sample/.
    """
    sample_dir = data_path("track1_sample", "sample")
    img_dir = os.path.join(sample_dir, "images", "default")

    texts_map = {}
    with open(os.path.join(sample_dir, "SampleDataset_Textnums_to_Texts 1.csv"),
              'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            texts_map[int(row['Text_nums'])] = row['Unique_Texts']

    gt = {}
    with open(os.path.join(sample_dir, "SampleDataset_Image_to_Textnums 1.csv"),
              'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            img_name = row['Image_names']
            text_nums = [int(x.strip()) for x in row['Text_nums'].split(';')]
            gt[img_name] = text_nums

    all_images = sorted(gt.keys())
    all_text_ids = sorted(texts_map.keys())
    all_texts = [prompt_prefix + texts_map[tid] for tid in all_text_ids]

    transform = CompetitionTransform(img_size)
    model.eval()
    with torch.no_grad():
        text_tokens = tokenizer(all_texts).to(device)
        text_tokens = text_tokens[:, :text_len]
        text_features = F.normalize(model.encode_text(text_tokens).float(), dim=-1)

        image_features_list = []
        for img_name in all_images:
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            img_feat = F.normalize(model.encode_image(img_t).float(), dim=-1)
            image_features_list.append(img_feat)
        image_features = torch.cat(image_features_list, dim=0)

    sims = image_features @ text_features.t()
    recalls = {1: [], 5: [], 10: []}
    for i, img_name in enumerate(all_images):
        gt_ids = set(gt[img_name])
        ranked = sims[i].argsort(descending=True)
        for k in [1, 5, 10]:
            top_k_ids = {all_text_ids[ranked[j].item()] for j in range(k)}
            matched = len(gt_ids & top_k_ids)
            recalls[k].append(matched / len(gt_ids))

    return tuple(np.mean(recalls[k]) * 100 for k in [1, 5, 10])


# ── Gemini Val Evaluation ──────────────────────────────────────────────────────
def evaluate_on_gemini_val(model, tokenizer, val_entries, device,
                           batch_size=64, img_size=224):
    """Evaluate on Gemini validation split.

    val_entries: list of (image_path, [texts])
    Returns R@1, R@5, R@10 (%).
    """
    transform = CompetitionTransform(img_size)
    model.eval()

    all_texts = []
    text_to_id = {}
    gt = {}

    for img_idx, (_, texts) in enumerate(val_entries):
        gt[img_idx] = set()
        for t in texts:
            if t not in text_to_id:
                text_to_id[t] = len(all_texts)
                all_texts.append(t)
            gt[img_idx].add(text_to_id[t])

    print(f"  Val set: {len(val_entries)} images, {len(all_texts)} unique texts")

    with torch.no_grad():
        text_feats = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            tokens = tokenizer(batch).to(device)
            feats = model.encode_text(tokens)
            text_feats.append(F.normalize(feats.float(), dim=-1).cpu())
        text_feats = torch.cat(text_feats, dim=0)

        image_feats = []
        for img_path, _ in val_entries:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            feats = model.encode_image(img_t)
            image_feats.append(F.normalize(feats.float(), dim=-1).cpu())
        image_feats = torch.cat(image_feats, dim=0)

    sims = image_feats @ text_feats.t()
    recalls = {1: [], 5: [], 10: []}
    for img_idx in range(len(val_entries)):
        gt_ids = gt[img_idx]
        ranked = sims[img_idx].argsort(descending=True)
        for k in [1, 5, 10]:
            top_k = set(ranked[:k].tolist())
            recalls[k].append(len(gt_ids & top_k) / len(gt_ids))

    results = {k: np.mean(v) * 100 for k, v in recalls.items()}
    return results[1], results[5], results[10]


# ── Custom Eval Evaluation ────────────────────────────────────────────────────
def evaluate_on_custom_eval(model, tokenizer, prompt_prefix="", device='cuda',
                            img_size=224):
    """Evaluate on the custom 11-image eval set (VG images, competition-style).

    Returns R@1, R@5, R@10 (%).
    Reads CSVs from track1_custom_eval/.
    """
    eval_dir = data_path("track1_custom_eval")
    img_dir  = os.path.join(eval_dir, "images")

    texts_map = {}
    with open(os.path.join(eval_dir, "custom_texts.csv"), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            texts_map[int(row['Text_nums'])] = row['Unique_Texts']

    gt = {}
    with open(os.path.join(eval_dir, "custom_image_to_texts.csv"), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            img_name  = row['Image_names']
            text_nums = [int(x.strip()) for x in row['Text_nums'].split(';')]
            gt[img_name] = text_nums

    all_images   = sorted(gt.keys())
    all_text_ids = sorted(texts_map.keys())
    all_texts    = [prompt_prefix + texts_map[tid] for tid in all_text_ids]

    transform = CompetitionTransform(img_size)
    model.eval()
    with torch.no_grad():
        text_tokens   = tokenizer(all_texts).to(device)
        text_features = F.normalize(model.encode_text(text_tokens).float(), dim=-1)

        image_features_list = []
        for img_name in all_images:
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            img_feat = F.normalize(model.encode_image(img_t).float(), dim=-1)
            image_features_list.append(img_feat)
        image_features = torch.cat(image_features_list, dim=0)

    sims = image_features @ text_features.t()
    recalls = {1: [], 5: [], 10: []}
    for i, img_name in enumerate(all_images):
        gt_ids = set(gt[img_name])
        ranked = sims[i].argsort(descending=True)
        for k in [1, 5, 10]:
            top_k_ids = {all_text_ids[ranked[j].item()] for j in range(k)}
            matched   = len(gt_ids & top_k_ids)
            recalls[k].append(matched / len(gt_ids))

    return tuple(np.mean(recalls[k]) * 100 for k in [1, 5, 10])


def evaluate_on_test_eval(model, tokenizer, prompt_prefix="", device='cuda',
                          img_size=224):
    """Evaluate on the web-image test eval set (54 images, non-training data).

    Returns R@1, R@5, R@10 (%).
    Reads CSVs from track1_test_eval/.
    """
    eval_dir = data_path("track1_test_eval")
    img_dir  = os.path.join(eval_dir, "images")

    texts_map = {}
    with open(os.path.join(eval_dir, "test_texts.csv"), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            texts_map[int(row['Text_nums'])] = row['Unique_Texts']

    gt = {}
    with open(os.path.join(eval_dir, "test_image_to_texts.csv"), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            img_name  = row['Image_names']
            text_nums = [int(x.strip()) for x in row['Text_nums'].split(';')]
            gt[img_name] = text_nums

    all_images   = sorted(gt.keys())
    all_text_ids = sorted(texts_map.keys())
    all_texts    = [prompt_prefix + texts_map[tid] for tid in all_text_ids]

    transform = CompetitionTransform(img_size)
    model.eval()
    with torch.no_grad():
        text_tokens   = tokenizer(all_texts).to(device)
        text_features = F.normalize(model.encode_text(text_tokens).float(), dim=-1)

        image_features_list = []
        for img_name in all_images:
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            img_feat = F.normalize(model.encode_image(img_t).float(), dim=-1)
            image_features_list.append(img_feat)
        image_features = torch.cat(image_features_list, dim=0)

    sims = image_features @ text_features.t()
    recalls = {1: [], 5: [], 10: []}
    for i, img_name in enumerate(all_images):
        gt_ids = set(gt[img_name])
        ranked = sims[i].argsort(descending=True)
        for k in [1, 5, 10]:
            top_k_ids = {all_text_ids[ranked[j].item()] for j in range(k)}
            matched   = len(gt_ids & top_k_ids)
            recalls[k].append(matched / len(gt_ids))

    return tuple(np.mean(recalls[k]) * 100 for k in [1, 5, 10])


# ── Combined Evaluation (for soup search) ─────────────────────────────────────
def evaluate_all(model, tokenizer, val_images, val_texts, val_gt,
                 val_entries, device='cuda'):
    """Run GVal + Sample + Proxy and return balanced score.

    balanced = 0.4*GVal_R10 + 0.3*Sample_R10 + 0.3*Proxy_R10

    Returns: (g_r10, s_r10, p_r10, balanced)
    """
    from .evaluate import (evaluate_on_gemini_val, evaluate_on_sample,
                           evaluate_proxy)
    g_r1, g_r5, g_r10 = evaluate_on_gemini_val(model, tokenizer, val_entries, device)
    s_r1, s_r5, s_r10 = evaluate_on_sample(model, tokenizer, device=device)
    p_r1, p_r5, p_r10 = evaluate_proxy(
        model, tokenizer, val_images, val_texts, val_gt, device=device)
    balanced = 0.4 * g_r10 + 0.3 * s_r10 + 0.3 * p_r10
    return g_r10, s_r10, p_r10, balanced
