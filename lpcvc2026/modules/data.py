"""
lpcvc2026.modules.data
======================
Data loading utilities for LPCVC 2026 Track 1.

Exports:
  CompetitionTransform    - /255 resize, no CLIP mean/std (matches competition pipeline)
  load_refcoco_fullimage  - RefCOCO/+ /g grouped by COCO image
  load_vg_fullimage       - Visual Genome region descriptions grouped by image
  load_coco_captions      - COCO original human captions
  load_and_split_gemini   - Load Gemini captions JSON and train/val split
  FullImageDataset        - Dataset for RefCOCO/VG entries
  CollateFn               - Collate for FullImageDataset
  GroupedBatchSampler     - Groups same-image texts for proper FN masking
  GeminiDataset           - Dataset for Gemini caption entries
  GeminiCollateFn         - Collate for GeminiDataset (samples N texts per image)
  build_proxy_valset      - Build proxy validation set (RefCOCO val + VG val)
  CLIPContrastiveLoss     - CLIP contrastive loss with FN masking
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
# Auto-detect BASE_DIR: use environment variable or auto-detect from module location
if 'LPCVC_BASE_DIR' in os.environ:
    BASE_DIR = os.environ['LPCVC_BASE_DIR']
else:
    # Auto-detect: go up 2 levels from modules/data.py
    # modules/data.py -> modules/ -> lpcvc2026/ -> IEEE_cv_challenge/
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(_MODULE_DIR))

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
VG_JSON            = data_path("region_descriptions.json")
VG_IMAGE_DIR       = data_path("vg_images", "VG_100K")
VG_SPLIT_CACHE     = data_path("vg_cache")
REFCOCO_CACHE      = data_path("refcoco_cache")

print(f"[lpcvc2026] BASE_DIR: {BASE_DIR}")
print(f"[lpcvc2026] DATA_ROOT: {DATA_ROOT}")

HF_DATASETS = {
    'refcoco':     'jxu124/refcoco',
    'refcocoplus': 'jxu124/refcocoplus',
    'refcocog':    'jxu124/refcocog',
}


# ── Image Transform ────────────────────────────────────────────────────────────
class CompetitionTransform:
    """Resize to target size + /255.0 to [0,1].
    Matches the official competition eval pipeline (upload_dataset.py).
    Do NOT use CLIP mean/std — the competition doesn't.
    """
    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, img):
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)
        return img


# ── RefCOCO Full-Image ─────────────────────────────────────────────────────────
def load_refcoco_fullimage(dataset_names, split='train', cache_dir=None,
                           min_texts_per_image=2):
    """Load RefCOCO data grouped by COCO image (full image, not bbox crops).

    Returns: list of (image_path, [texts], unique_image_id)
      unique_image_id is negative (to avoid collision with VG ids).
    """
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

    image_to_texts = {}
    for img_id, bbox, cap in pairs:
        if img_id not in image_to_texts:
            image_to_texts[img_id] = set()
        image_to_texts[img_id].add(cap)

    entries = []
    for img_id, texts in image_to_texts.items():
        if len(texts) < min_texts_per_image:
            continue
        img_path = os.path.join(COCO_TRAIN2014_DIR,
                                f"COCO_train2014_{img_id:012d}.jpg")
        if not os.path.exists(img_path):
            continue
        entries.append((img_path, sorted(texts), -img_id))  # negative = COCO

    n_texts = sum(len(e[1]) for e in entries)
    print(f"  RefCOCO full-image: {len(entries)} images, {n_texts} unique texts "
          f"(min {min_texts_per_image} texts/img)")
    return entries


# ── Visual Genome Full-Image ───────────────────────────────────────────────────
def load_vg_fullimage(split='train', min_phrases=4):
    """Load VG region descriptions grouped by image.

    Returns: list of (image_path, [texts], unique_image_id)
    """
    split_file = os.path.join(VG_SPLIT_CACHE, "vg_splits.json")
    if not os.path.exists(split_file):
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
        entries.append((img_path, sorted(phrases), img_id))

    n_texts = sum(len(e[1]) for e in entries)
    print(f"  VG {split}: {len(entries)} images, {n_texts} texts")
    return entries


# ── VG Structured Full-Image (calibration-style spatial captions) ──────────────
def load_vg_structured_fullimage(split='train', min_phrases=4):
    """Load pre-generated structured captions (calibration-style spatial format).

    Format: 'The [phrase] on the [side] side' — generated by generate_structured_captions.py.
    Returns: list of (img_path, [structured_phrases], image_id) — same as load_vg_fullimage.
    """
    cache_path = os.path.join(VG_SPLIT_CACHE, f"vg_structured_fullimage_{split}.json")
    assert os.path.exists(cache_path), (
        f"Structured captions not found: {cache_path}\n"
        f"Run: python generate_structured_captions.py --split {split}"
    )
    with open(cache_path) as f:
        raw = json.load(f)
    entries = []
    for img_path, phrases, img_id in raw:
        if len(phrases) < min_phrases:
            continue
        img_path = _remap_cached_image_path(img_path, VG_IMAGE_DIR)
        if os.path.exists(img_path):
            entries.append((img_path, phrases, img_id))
    n_texts = sum(len(e[1]) for e in entries)
    print(f"  VG structured ({split}): {len(entries)} images, {n_texts} texts")
    return entries


# ── COCO Original Captions ─────────────────────────────────────────────────────
def load_coco_captions(ann_path, coco_dir=None):
    """Load COCO original human-annotated captions.

    Returns: list of (image_path, [captions])
    """
    if coco_dir is None:
        coco_dir = COCO_TRAIN2014_DIR
    print(f"Loading COCO captions from {ann_path}...")
    with open(ann_path, encoding='utf-8') as f:
        data = json.load(f)

    id_to_file = {img['id']: img['file_name'] for img in data['images']}

    from collections import defaultdict
    img_to_caps = defaultdict(list)
    for ann in data['annotations']:
        img_to_caps[ann['image_id']].append(ann['caption'])

    entries = []
    for img_id, caps in img_to_caps.items():
        img_name = id_to_file.get(img_id)
        if img_name is None:
            continue
        img_path = os.path.join(coco_dir, img_name)
        if os.path.exists(img_path):
            entries.append((img_path, caps))

    print(f"  COCO captions: {len(entries)} images, "
          f"{sum(len(c) for _, c in entries)} captions")
    return entries


# ── Gemini Caption Loading ─────────────────────────────────────────────────────
def load_and_split_gemini(json_path, val_ratio=0.1, seed=42):
    """Load Gemini captions JSON and split into train/val.

    Returns: (train_entries, val_entries)
      Each entry: (image_path, [texts])
    """
    print(f"Loading Gemini captions from {json_path}...")
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    for img_name, texts in data.items():
        img_name = os.path.basename(os.path.normpath(str(img_name)))
        img_path = os.path.join(COCO_TRAIN2014_DIR, img_name)
        if os.path.exists(img_path):
            entries.append((img_path, texts))

    print(f"Total valid entries: {len(entries)}")

    random.seed(seed)
    random.shuffle(entries)

    val_size = int(len(entries) * val_ratio)
    val_entries = entries[:val_size]
    train_entries = entries[val_size:]

    print(f"Train: {len(train_entries)} images")
    print(f"Val:   {val_size} images")
    return train_entries, val_entries


# ── Datasets ───────────────────────────────────────────────────────────────────
class FullImageDataset(Dataset):
    """Dataset for RefCOCO/VG entries.

    Each __getitem__ returns (image_tensor, text, image_id).
    Same image appears multiple times with different texts.
    """
    def __init__(self, entries, image_transform, tokenizer, prompt_prefix=""):
        """entries: list of (image_path, [texts], unique_image_id)"""
        self.image_transform = image_transform
        self.prompt_prefix = prompt_prefix
        self.pairs = []
        self.image_to_indices = {}

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


class GeminiDataset(Dataset):
    """Dataset for Gemini-generated captions.

    Each __getitem__ returns (image_tensor, [texts], image_idx).
    """
    def __init__(self, entries, transform):
        """entries: [(image_path, [texts]), ...]"""
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, texts = self.entries[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, texts, idx


class GeminiCollateFn:
    """Collate for GeminiDataset — samples texts_per_image texts per image."""
    def __init__(self, tokenizer, texts_per_image=4):
        self.tokenizer = tokenizer
        self.texts_per_image = texts_per_image

    def __call__(self, batch):
        images, all_texts, image_ids = zip(*batch)
        images = torch.stack(images, 0)

        selected_texts = []
        selected_ids = []
        for img_idx, texts in enumerate(all_texts):
            if len(texts) >= self.texts_per_image:
                sampled = random.sample(texts, self.texts_per_image)
            else:
                sampled = texts * (self.texts_per_image // len(texts) + 1)
                sampled = sampled[:self.texts_per_image]
            selected_texts.extend(sampled)
            selected_ids.extend([img_idx] * len(sampled))

        tokens = self.tokenizer(selected_texts)
        return images, tokens, torch.tensor(selected_ids)


# ── Proxy Val Set ──────────────────────────────────────────────────────────────
def build_proxy_valset(n_images=300, texts_per_image=4, seed=42):
    """Build proxy validation set from RefCOCO val + VG val.

    Simulates the competition test set: ~300 images, ~1200 texts.
    Returns: (val_images, val_texts, val_gt)
      val_images: list of image paths
      val_texts:  list of all text strings
      val_gt:     dict {img_idx -> set of text_indices}
    """
    rng = random.Random(seed)

    refcoco_val = load_refcoco_fullimage(
        ['refcoco', 'refcocoplus', 'refcocog'], split='validation',
        cache_dir=REFCOCO_CACHE, min_texts_per_image=2)
    vg_val = load_vg_fullimage(split='val', min_phrases=4)

    entries = refcoco_val + vg_val
    rng.shuffle(entries)
    if len(entries) > n_images:
        entries = entries[:n_images]

    val_images = []
    val_texts = []
    val_gt = {}

    for img_idx, (img_path, texts, _) in enumerate(entries):
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


# ── Visual Genome Region Crops ────────────────────────────────────────────────
def load_vg_regions(split='train', min_box_px=32, min_phrase_len=5,
                    max_per_image=10, seed=42):
    """Load VG region bbox + phrase pairs for crop-contrastive training.

    Uses the same train/val/test split as load_vg_fullimage (from vg_splits.json).
    Returns: list of [img_path, x, y, w, h, phrase, image_id]
    Caches results to vg_cache/vg_region_pairs_{split}.json.
    """
    cache_path = os.path.join(VG_SPLIT_CACHE, f"vg_region_pairs_{split}.json")
    if os.path.exists(cache_path):
        print(f"  Loading cached VG regions: {cache_path}")
        with open(cache_path, 'r') as f:
            pairs = json.load(f)
        print(f"  VG region pairs ({split}): {len(pairs)}")
        return pairs

    # Load the split to know which image ids are in this split
    split_file = os.path.join(VG_SPLIT_CACHE, "vg_splits.json")
    if not os.path.exists(split_file):
        # Force creation via load_vg_fullimage
        load_vg_fullimage(split='train', min_phrases=4)
    with open(split_file, 'r') as f:
        splits_data = json.load(f)
    split_ids = {int(entry[0]) for entry in splits_data[split]}

    print(f"  Parsing region_descriptions.json for {split} split ({len(split_ids)} images)...")
    with open(VG_JSON, 'r') as f:
        data = json.load(f)

    rng = random.Random(seed)
    pairs = []
    for entry in data:
        img_id = int(entry["id"])
        if img_id not in split_ids:
            continue
        img_path = os.path.join(VG_IMAGE_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        valid = []
        for r in entry.get("regions", []):
            phrase = r.get("phrase", "").strip()
            x = r.get("x", 0); y = r.get("y", 0)
            w = r.get("width", 0); h = r.get("height", 0)
            if len(phrase) < min_phrase_len:
                continue
            if min(w, h) < min_box_px:
                continue
            valid.append([img_path, int(x), int(y), int(w), int(h), phrase, img_id])
        if not valid:
            continue
        if len(valid) > max_per_image:
            rng.shuffle(valid)
            valid = valid[:max_per_image]
        pairs.extend(valid)

    print(f"  VG region pairs ({split}): {len(pairs)} from {len(split_ids)} images")
    os.makedirs(VG_SPLIT_CACHE, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(pairs, f)
    return pairs


class VGRegionDataset(Dataset):
    """Crops VG bounding box regions and pairs with region phrases.

    Returns (crop_tensor, [phrase], image_id) — compatible with GeminiCollateFn
    (texts_per_image=1).  image_id is the parent VG image id for FN masking.
    """
    def __init__(self, region_pairs, transform, crop_margin=0.10):
        """region_pairs: list of [img_path, x, y, w, h, phrase, image_id]"""
        self.pairs = region_pairs
        self.transform = transform
        self.crop_margin = crop_margin

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, x, y, w, h, phrase, image_id = self.pairs[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            iw, ih = img.size
            mx = w * self.crop_margin
            my = h * self.crop_margin
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(iw, x + w + mx)
            y2 = min(ih, y + h + my)
            img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        img = self.transform(img)
        return img, [phrase], image_id


# ── Loss ───────────────────────────────────────────────────────────────────────
class CLIPContrastiveLoss(nn.Module):
    """CLIP contrastive loss with False-Negative masking.

    Texts from the same image are masked out as negatives.
    """
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def forward(self, img_feats, txt_feats, image_ids):
        img_feats = F.normalize(img_feats.float(), dim=-1)
        txt_feats = F.normalize(txt_feats.float(), dim=-1)

        n = img_feats.shape[0]
        labels = torch.arange(n, device=img_feats.device)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * (img_feats @ txt_feats.t())

        fn_mask = (image_ids.unsqueeze(0) == image_ids.unsqueeze(1))
        fn_mask.fill_diagonal_(False)
        logits_f32 = logits.float()
        logits_i2t = logits_f32.masked_fill(fn_mask, -1e9)
        logits_t2i = logits_f32.t().masked_fill(fn_mask.t(), -1e9)

        loss = (F.cross_entropy(logits_i2t, labels) +
                F.cross_entropy(logits_t2i, labels)) / 2.0
        return loss
