"""
Unified multi-dataset full-image training for LPCVC 2026 Track 1.

Combines RefCOCO/RefCOCO+/RefCOCOg + Visual Genome region descriptions.
ALL training uses full images (not bbox crops) with FN masking.

Key differences from previous scripts:
  - Full image only (matches competition: full scene → text retrieval)
  - Always uses grouped batch + FN masking
  - Multi-source: RefCOCO family + VG regions
  - Large proxy validation set (~300 images, ~1200 texts) to simulate test

Usage:
    python train_fullimg.py                                  # defaults
    python train_fullimg.py --epochs 5 --lr 2e-6
    python train_fullimg.py --datasets refcoco vg            # specific combo
    python train_fullimg.py --resume models/refcoco_ft/best.pt
    python train_fullimg.py --prompt ""                      # no prompt prefix
"""

import os
import sys
import json
import random
import argparse
import math
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
import open_clip

# ============================================================
# Paths & Config
# ============================================================
BASE_DIR = os.environ.get("LPCVC_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
_DATA_ROOT_CANDIDATES = [
    os.path.join(BASE_DIR, "LPCVC_DATA"),
    r"D:\LPCVC_DATA",
    BASE_DIR,
]
DATA_ROOT = os.environ.get(
    "LPCVC_DATA_ROOT",
    next((p for p in _DATA_ROOT_CANDIDATES if os.path.exists(p)), BASE_DIR),
)


def data_path(*parts):
    """Resolve dataset files from DATA_ROOT, falling back to BASE_DIR."""
    p = os.path.join(DATA_ROOT, *parts)
    if os.path.exists(p):
        return p
    return os.path.join(BASE_DIR, *parts)


def _remap_cached_image_path(path, image_dir):
    """Make cached absolute image paths survive moving the dataset root."""
    if os.path.isabs(path):
        image_dir_norm = os.path.normcase(os.path.abspath(image_dir))
        path_norm = os.path.normcase(os.path.abspath(path))
        if not path_norm.startswith(image_dir_norm):
            remapped = os.path.join(image_dir, os.path.basename(path))
            if os.path.exists(remapped):
                return remapped
    if os.path.exists(path):
        return path
    remapped = os.path.join(image_dir, os.path.basename(path))
    if os.path.exists(remapped):
        return remapped
    return path


COCO_TRAIN2014_DIR = data_path("train2014", "train2014")
VG_JSON = data_path("region_descriptions.json")
VG_IMAGE_DIR = data_path("vg_images", "VG_100K")
VG_SPLIT_CACHE = data_path("vg_cache")
REFCOCO_CACHE = data_path("refcoco_cache")

STUDENT_ARCH = "ViT-B-16-quickgelu"
STUDENT_PRETRAINED = "dfn2b"
SAVE_DIR = os.path.join(BASE_DIR, "models", "fullimg_ft_dfn2b")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

HF_DATASETS = {
    'refcoco':     'jxu124/refcoco',
    'refcocoplus': 'jxu124/refcocoplus',
    'refcocog':    'jxu124/refcocog',
}


# ============================================================
# Image Transform (competition-matching)
# ============================================================
class CompetitionTransform:
    """Direct resize to target size + /255.0 to [0,1]. No CLIP mean/std.
    Matches the official competition eval pipeline (upload_dataset.py)."""
    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, img):
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)
        return img


# ============================================================
# RefCOCO Full-Image Dataset
# ============================================================
def load_refcoco_fullimage(dataset_names, split='train', cache_dir=None,
                           min_texts_per_image=2):
    """
    Load RefCOCO data and group by COCO image (full image, not crops).
    Returns: list of (image_path, [texts], unique_image_id)
    """
    # Load raw pairs from cache or HuggingFace
    cache_name = f"refcoco_{'_'.join(sorted(dataset_names))}_{split}.json"
    cache_path = os.path.join(cache_dir, cache_name) if cache_dir else None

    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached RefCOCO: {cache_path}")
        with open(cache_path, 'r') as f:
            pairs = json.load(f)
    else:
        from datasets import load_dataset
        pairs = []
        for ds_name in dataset_names:
            hf_name = HF_DATASETS[ds_name]
            print(f"  Loading {ds_name} ({hf_name}) split={split}...")
            ds = load_dataset(hf_name, split=split, streaming=True)
            count = 0
            for sample in ds:
                image_id = sample['image_id']
                for cap in sample['captions']:
                    cap = cap.strip()
                    if len(cap) >= 2:
                        pairs.append([image_id, sample['bbox'], cap])
                        count += 1
            print(f"    {count} pairs loaded")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(pairs, f)

    # Group by image_id → full image with multiple texts
    image_to_texts = {}
    for img_id, bbox, cap in pairs:
        if img_id not in image_to_texts:
            image_to_texts[img_id] = set()
        image_to_texts[img_id].add(cap)

    # Build entries: (image_path, [texts], unique_id)
    entries = []
    for img_id, texts in image_to_texts.items():
        if len(texts) < min_texts_per_image:
            continue
        img_path = os.path.join(COCO_TRAIN2014_DIR,
                                f"COCO_train2014_{img_id:012d}.jpg")
        if not os.path.exists(img_path):
            continue
        # Use negative image_id to avoid collision with VG image_ids
        unique_id = -img_id  # negative = COCO
        # Keep deterministic order across processes (set -> sorted list)
        entries.append((img_path, sorted(texts), unique_id))

    n_texts = sum(len(e[1]) for e in entries)
    print(f"  RefCOCO full-image: {len(entries)} images, {n_texts} unique texts "
          f"(min {min_texts_per_image} texts/img)")
    return entries


# ============================================================
# VG Full-Image Dataset
# ============================================================
def load_vg_fullimage(split='train', min_phrases=4):
    """
    Load VG region descriptions grouped by image.
    Returns: list of (image_path, [texts], unique_image_id)
    """
    split_file = os.path.join(VG_SPLIT_CACHE, "vg_splits.json")
    if not os.path.exists(split_file):
        # Need to generate splits
        print(f"  VG splits not found. Loading region_descriptions.json...")
        with open(VG_JSON, 'r') as f:
            data = json.load(f)
        available = []
        for entry in data:
            img_id = entry["id"]
            img_path = os.path.join(VG_IMAGE_DIR, f"{img_id}.jpg")
            if os.path.exists(img_path):
                phrases = sorted({r["phrase"].strip() for r in entry.get("regions", [])
                                 if r.get("phrase", "").strip()})
                if len(phrases) >= min_phrases:
                    available.append([img_id, img_path, phrases])
        rng = random.Random(42)
        rng.shuffle(available)
        n = len(available)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        splits = {
            'train': available[:n_train],
            'val':   available[n_train:n_train + n_val],
            'test':  available[n_train + n_val:],
        }
        os.makedirs(VG_SPLIT_CACHE, exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump(splits, f)
    else:
        with open(split_file, 'r') as f:
            splits = json.load(f)

    split_data = splits[split]
    entries = []
    for img_id, img_path, phrases in split_data:
        if len(phrases) < min_phrases:
            continue
        img_path = _remap_cached_image_path(img_path, VG_IMAGE_DIR)
        if not os.path.exists(img_path):
            continue
        entries.append((img_path, sorted(phrases), img_id))  # VG ids are positive

    n_texts = sum(len(e[1]) for e in entries)
    print(f"  VG {split}: {len(entries)} images, {n_texts} texts")
    return entries


# ============================================================
# Unified Full-Image Dataset
# ============================================================
class FullImageDataset(Dataset):
    """
    Unified dataset: each sample = (image, text, image_id).
    A single image appears multiple times with different texts.
    """
    def __init__(self, entries, image_transform, tokenizer, prompt_prefix=""):
        """
        entries: list of (image_path, [texts], unique_image_id)
        """
        self.image_transform = image_transform
        self.prompt_prefix = prompt_prefix
        self.pairs = []       # (image_path, text, image_id)
        self.image_to_indices = {}  # image_id -> [indices]

        for img_path, texts, img_id in entries:
            for txt in texts:
                idx = len(self.pairs)
                self.pairs.append((img_path, txt, img_id))
                if img_id not in self.image_to_indices:
                    self.image_to_indices[img_id] = []
                self.image_to_indices[img_id].append(idx)

        self.image_ids = list(self.image_to_indices.keys())
        n_pairs = len(self.pairs)
        n_imgs = len(self.image_ids)
        print(f"    FullImageDataset: {n_pairs} pairs, {n_imgs} images, "
              f"avg {n_pairs/n_imgs:.1f} texts/img")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption, image_id = self.pairs[idx]
        text = self.prompt_prefix + caption
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a blank image on error
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        if self.image_transform:
            img = self.image_transform(img)
        return img, text, image_id


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images, texts, image_ids = zip(*batch)
        images = torch.stack(images, 0)
        tokens = self.tokenizer(list(texts))
        image_ids = torch.tensor(image_ids, dtype=torch.long)
        return images, tokens, image_ids


# ============================================================
# Grouped Batch Sampler
# ============================================================
class GroupedBatchSampler:
    """Groups same-image texts into batches for proper FN masking."""
    def __init__(self, dataset, images_per_batch=32, texts_per_image=4,
                 shuffle=True, drop_last=True):
        self.image_to_indices = dataset.image_to_indices
        self.image_ids = dataset.image_ids
        self.images_per_batch = images_per_batch
        self.texts_per_image = texts_per_image
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        image_ids = self.image_ids.copy()
        if self.shuffle:
            random.shuffle(image_ids)
        for start in range(0, len(image_ids), self.images_per_batch):
            batch_ids = image_ids[start:start + self.images_per_batch]
            if self.drop_last and len(batch_ids) < self.images_per_batch:
                continue
            batch_indices = []
            for iid in batch_ids:
                indices = self.image_to_indices[iid]
                if len(indices) > self.texts_per_image:
                    selected = random.sample(indices, self.texts_per_image)
                else:
                    selected = indices
                batch_indices.extend(selected)
            yield batch_indices

    def __len__(self):
        return len(self.image_ids) // self.images_per_batch


# ============================================================
# Loss (CLIP contrastive + FN masking)
# ============================================================
class CLIPContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def forward(self, img_feats, txt_feats, image_ids, hard_neg_txt_feats=None):
        img_feats = F.normalize(img_feats.float(), dim=-1)
        txt_feats = F.normalize(txt_feats.float(), dim=-1)

        n = img_feats.shape[0]
        labels = torch.arange(n, device=img_feats.device)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * (img_feats @ txt_feats.t())

        # FN masking: texts from same image are not negatives
        fn_mask = (image_ids.unsqueeze(0) == image_ids.unsqueeze(1))
        fn_mask.fill_diagonal_(False)
        logits_f32 = logits.float()
        logits_i2t = logits_f32.masked_fill(fn_mask, -1e9)
        logits_t2i = logits_f32.t().masked_fill(fn_mask.t(), -1e9)

        if hard_neg_txt_feats is not None and hard_neg_txt_feats.shape[0] > 0:
            # Expand i2t denominator with hard negatives (spatial-swapped captions).
            # Hard negs are treated as guaranteed negatives for all images — no FN masking needed
            # since spatial swaps produce semantically incorrect captions.
            hn = F.normalize(hard_neg_txt_feats.float(), dim=-1)
            logits_hn = (logit_scale * (img_feats @ hn.t())).float()
            logits_i2t = torch.cat([logits_i2t, logits_hn], dim=1)

        loss = (F.cross_entropy(logits_i2t, labels) +
                F.cross_entropy(logits_t2i, labels)) / 2.0
        return loss


class SigLIPLoss(nn.Module):
    """Pairwise sigmoid BCE loss (SigLIP, 2303.15343).

    Each (img, txt) pair is treated independently with binary label +1/−1.
    Eliminates row-softmax FN pollution from InfoNCE; a learned bias calibrates
    the prior probability of a positive pair in the batch.
    """
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))  # ≈ −log(N−1) for N≈128

    def forward(self, img_feats, txt_feats, image_ids, hard_neg_txt_feats=None):
        img_feats = F.normalize(img_feats.float(), dim=-1)
        txt_feats = F.normalize(txt_feats.float(), dim=-1)

        n = img_feats.shape[0]
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = (logit_scale * (img_feats @ txt_feats.t()) + self.logit_bias).float()

        labels = -torch.ones(n, n, device=img_feats.device)
        labels.fill_diagonal_(1.0)

        fn_mask = (image_ids.unsqueeze(0) == image_ids.unsqueeze(1))
        fn_mask.fill_diagonal_(False)

        loss_mat = -F.logsigmoid(labels * logits)
        loss_mat = loss_mat.masked_fill(fn_mask, 0.0)
        loss = loss_mat.sum() / (~fn_mask).float().sum()
        return loss


# ============================================================
# Large Proxy Validation Set
# ============================================================
def build_proxy_valset(n_images=300, texts_per_image=4, seed=42):
    """
    Build a validation set that simulates the competition test set:
    ~300 images with ~1200 texts (mixed sources).
    Uses RefCOCO validation + VG validation.
    """
    rng = random.Random(seed)
    entries = []

    # RefCOCO val (full image, grouped)
    refcoco_val = load_refcoco_fullimage(
        ['refcoco', 'refcocoplus', 'refcocog'], split='validation',
        cache_dir=REFCOCO_CACHE, min_texts_per_image=2,
    )
    entries.extend(refcoco_val)

    # VG val
    vg_val = load_vg_fullimage(split='val', min_phrases=4)
    entries.extend(vg_val)

    rng.shuffle(entries)

    # Trim to target size
    if len(entries) > n_images:
        entries = entries[:n_images]

    # For each image, sample texts_per_image texts
    val_images = []   # (image_path, image_idx)
    val_texts = []    # all texts
    val_gt = {}       # image_idx -> set of text indices

    for img_idx, (img_path, texts, img_id) in enumerate(entries):
        val_images.append(img_path)
        if len(texts) > texts_per_image:
            selected = rng.sample(texts, texts_per_image)
        else:
            selected = texts
        gt_indices = set()
        for txt in selected:
            gt_indices.add(len(val_texts))
            val_texts.append(txt)
        val_gt[img_idx] = gt_indices

    print(f"  Proxy val: {len(val_images)} images, {len(val_texts)} texts "
          f"(avg {len(val_texts)/len(val_images):.1f}/img)")
    return val_images, val_texts, val_gt


def evaluate_proxy(model, tokenizer, val_images, val_texts, val_gt,
                   prompt_prefix="", device='cuda', batch_size=64, img_size=224):
    """Evaluate on proxy validation set. Returns R@1, R@5, R@10."""
    transform = CompetitionTransform(img_size)
    prefixed_texts = [prompt_prefix + t for t in val_texts]

    model.eval()
    with torch.no_grad():
        # Encode texts
        text_feats = []
        for i in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[i:i + batch_size]
            tokens = tokenizer(batch).to(device)
            feats = model.encode_text(tokens)
            text_feats.append(F.normalize(feats.float(), dim=-1).cpu())
        text_feats = torch.cat(text_feats, dim=0)

        # Encode images
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


# ============================================================
# Sample Evaluation (56 images)
# ============================================================
def evaluate_on_sample(model, tokenizer, prompt_prefix="", device='cuda'):
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

    transform = CompetitionTransform(224)
    model.eval()
    with torch.no_grad():
        text_tokens = tokenizer(all_texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)

        image_features_list = []
        for img_name in all_images:
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_t)
            img_feat = F.normalize(img_feat.float(), dim=-1)
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


# ============================================================
# Training
# ============================================================
def train(args):
    save_dir = getattr(args, '_save_dir_override', None) or SAVE_DIR
    prompt_prefix = args.prompt
    print(f"\n{'='*70}")
    print(f"  Full-Image Multi-Dataset Training")
    print(f"  Datasets: {args.datasets}")
    print(f"  Prompt: \"{prompt_prefix}\"")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = CompetitionTransform(224)
    tokenizer = open_clip.get_tokenizer(STUDENT_ARCH)

    # ---- Load training data ----
    print(f"\n  Loading training data...")
    all_entries = []  # list of (image_path, [texts], unique_image_id)

    if 'refcoco' in args.datasets:
        refcoco_entries = load_refcoco_fullimage(
            ['refcoco', 'refcocoplus', 'refcocog'], split='train',
            cache_dir=REFCOCO_CACHE, min_texts_per_image=args.min_texts,
        )
        all_entries.extend(refcoco_entries)

    if 'vg' in args.datasets:
        vg_entries = load_vg_fullimage(
            split='train', min_phrases=args.min_texts,
        )
        all_entries.extend(vg_entries)

    print(f"\n  Total: {len(all_entries)} images")

    # ---- Build dataset ----
    train_dataset = FullImageDataset(
        all_entries, transform, tokenizer, prompt_prefix=prompt_prefix,
    )

    ipb = max(4, args.batch_size // args.texts_per_image)
    sampler = GroupedBatchSampler(
        train_dataset, images_per_batch=ipb,
        texts_per_image=args.texts_per_image,
        shuffle=True, drop_last=True,
    )
    dataloader = DataLoader(
        train_dataset, batch_sampler=sampler,
        num_workers=0, collate_fn=CollateFn(tokenizer),
        pin_memory=True,
    )
    print(f"  GroupedBatch: {ipb} images x {args.texts_per_image} texts "
          f"= ~{ipb * args.texts_per_image}/batch")

    # ---- Build proxy validation set ----
    print(f"\n  Building proxy validation set...")
    val_images, val_texts, val_gt = build_proxy_valset(
        n_images=args.val_images, texts_per_image=args.val_texts_per_image,
        seed=42,
    )

    # ---- Load model ----
    print(f"\n  Loading model: {STUDENT_ARCH}/{STUDENT_PRETRAINED}...")
    model, _, _ = open_clip.create_model_and_transforms(
        STUDENT_ARCH, pretrained=STUDENT_PRETRAINED,
    )
    model = model.to(device)

    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {trainable / 1e6:.0f}M trainable")

    # ---- Baseline eval ----
    print(f"\n  Baseline evaluation...")
    s_r1, s_r5, s_r10 = evaluate_on_sample(model, tokenizer, prompt_prefix, device)
    print(f"    Sample (56img/211txt):  R@1={s_r1:.1f}%  R@5={s_r5:.1f}%  R@10={s_r10:.1f}%")

    p_r1, p_r5, p_r10 = evaluate_proxy(
        model, tokenizer, val_images, val_texts, val_gt,
        prompt_prefix=prompt_prefix, device=device,
    )
    print(f"    Proxy  ({len(val_images)}img/{len(val_texts)}txt): "
          f"R@1={p_r1:.1f}%  R@5={p_r5:.1f}%  R@10={p_r10:.1f}%")

    best_sample_r10 = s_r10
    best_proxy_r10 = p_r10

    # ---- Loss & Optimizer ----
    criterion = CLIPContrastiveLoss().to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd,
        betas=(0.9, 0.98), eps=1e-6,
    )

    n_batches = len(dataloader)
    opt_steps_per_epoch = n_batches // args.grad_accum
    total_steps = args.epochs * opt_steps_per_epoch
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n  Training: {args.epochs} epochs x {n_batches} batches = {total_steps} opt steps")
    print(f"  LR={args.lr}, WD={args.wd}, Warmup={warmup_steps}, GradAccum={args.grad_accum}")

    # ---- Training loop ----
    grad_accum = args.grad_accum
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        t0 = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)
        optimizer.zero_grad(set_to_none=True)
        for step_in_epoch, (images, tokens, image_ids) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            image_ids = image_ids.to(device)

            with autocast('cuda', dtype=torch.float16):
                img_feats = model.encode_image(images)
                txt_feats = model.encode_text(tokens)
                loss = criterion(img_feats, txt_feats, image_ids)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step_in_epoch + 1) % grad_accum == 0 or (step_in_epoch + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum
            n_batch += 1
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item()*grad_accum:.3f}", lr=f"{lr_now:.1e}")

        avg_loss = epoch_loss / max(n_batch, 1)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch+1}: loss={avg_loss:.4f}  time={elapsed:.0f}s")

        # ---- Evaluation ----
        model.eval()
        s_r1, s_r5, s_r10 = evaluate_on_sample(
            model, tokenizer, prompt_prefix, device,
        )
        print(f"    Sample (56/211):  R@1={s_r1:.1f}%  R@5={s_r5:.1f}%  R@10={s_r10:.1f}%")

        p_r1, p_r5, p_r10 = evaluate_proxy(
            model, tokenizer, val_images, val_texts, val_gt,
            prompt_prefix=prompt_prefix, device=device,
        )
        print(f"    Proxy  ({len(val_images)}/{len(val_texts)}): "
              f"R@1={p_r1:.1f}%  R@5={p_r5:.1f}%  R@10={p_r10:.1f}%")

        # Save best (prioritize proxy R@10 as it's more realistic)
        improved = False
        reason = []
        if p_r10 > best_proxy_r10:
            best_proxy_r10 = p_r10
            improved = True
            reason.append(f"proxy_R@10={p_r10:.1f}%")
        if s_r10 > best_sample_r10:
            best_sample_r10 = s_r10
            if not improved:
                improved = True
            reason.append(f"sample_R@10={s_r10:.1f}%")

        if improved:
            save_path = os.path.join(save_dir, "best.pt")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'sample_r10': s_r10,
                'sample_r1': s_r1,
                'proxy_r10': p_r10,
                'proxy_r1': p_r1,
                'arch': STUDENT_ARCH,
                'pretrained': STUDENT_PRETRAINED,
                'avg_loss': avg_loss,
                'prompt_prefix': prompt_prefix,
                'datasets': args.datasets,
            }, save_path)
            print(f"    ** SAVED best ({', '.join(reason)}) **")

        # Periodic checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'sample_r10': s_r10,
            'proxy_r10': p_r10,
            'avg_loss': avg_loss,
        }, ckpt_path)

    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best Sample R@10: {best_sample_r10:.1f}%")
    print(f"  Best Proxy  R@10: {best_proxy_r10:.1f}%")
    print(f"  Model: {os.path.join(save_dir, 'best.pt')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-image multi-dataset CLIP FT")

    # Data
    parser.add_argument("--datasets", nargs='+', default=['refcoco', 'vg'],
                        choices=['refcoco', 'vg'],
                        help="Data sources to use")
    parser.add_argument("--min_texts", type=int, default=4,
                        help="Min texts per image to include")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Approx batch size (images_per_batch * texts_per_image)")
    parser.add_argument("--texts_per_image", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")

    # Prompt
    parser.add_argument("--prompt", type=str, default="none",
                        help="Prompt prefix ('none' = no prompt)")
    parser.add_argument("--no_prompt", action="store_true",
                        help="Disable prompt prefix")

    # Validation
    parser.add_argument("--val_images", type=int, default=300,
                        help="Number of images in proxy validation set")
    parser.add_argument("--val_texts_per_image", type=int, default=4,
                        help="Texts per image in proxy val (total ~val_images*this)")

    # Resume
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory")

    args = parser.parse_args()
    if args.no_prompt or args.prompt == "none":
        args.prompt = ""
    if args.save_dir:
        args._save_dir_override = os.path.join(BASE_DIR, "models", args.save_dir)
    else:
        args._save_dir_override = None
    train(args)
