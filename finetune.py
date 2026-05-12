"""
Train with Gemini captions - custom train/val split.

Key difference from freeze_img:
  - Uses Gemini captions as PRIMARY data source (68K images)
  - Splits into train/val (e.g., 90/10) from Gemini itself
  - Trains BOTH image and text encoder (full fine-tune)
  - Validation set matches the text style of actual competition

Usage:
    python train_gemini_full.py --val_ratio 0.1 --epochs 5 --lr 5e-6
"""
import os, sys, math, time, argparse, random, json, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import open_clip

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_utils import (
    CompetitionTransform, CLIPContrastiveLoss, SigLIPLoss,
    BASE_DIR, DATA_ROOT, data_path, COCO_TRAIN2014_DIR,
    load_refcoco_fullimage, load_vg_fullimage,
    build_proxy_valset, evaluate_proxy,
)
from npu_utils import apply_no_cls_gap, load_no_cls_checkpoint, verify_no_cls, apply_txt_gap, verify_txt_gap
from ptqat_utils import apply_ptqat_text, remove_ptqat_text

COCO_VAL2014_DIR = data_path("val2014", "val2014")

# ── Spatial hard negative generation ──────────────────────────────────────────
_SPATIAL_OPPOSITE = {"left": "right", "right": "left", "upper": "lower", "lower": "upper", "center": "left"}
_SPATIAL_RE = re.compile(r'\b(left|right|upper|lower|center)\b', re.IGNORECASE)

def _get_spatial_hard_neg(text):
    """Return one hard negative by swapping the spatial direction, or None if not a structured caption."""
    m = _SPATIAL_RE.search(text)
    if m is None:
        return None
    opposite = _SPATIAL_OPPOSITE[m.group(1).lower()]
    return text[:m.start()] + opposite + text[m.end():]

# Defaults (can be overridden via --arch / --pretrained)
DEFAULT_ARCH = "ViT-B-16-quickgelu"
DEFAULT_PRETRAINED = "dfn2b"
DEFAULT_IMG_SIZE = 224
SAVE_DIR = os.path.join(BASE_DIR, "models", "gemini_full_ft")


# ============================================================
# Dataset for Gemini Captions
# ============================================================
class GeminiDataset(Dataset):
    """Dataset for Gemini-generated referring expressions."""
    def __init__(self, entries, transform):
        """
        entries: [(image_path, [texts]), ...]
        """
        self.entries = entries
        self.transform = transform
        self.image_to_texts = {i: texts for i, (_, texts) in enumerate(entries)}
        self.image_paths = [path for path, _ in entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, texts = self.entries[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # Return all texts for this image
        return img, texts, idx


class GeminiCollateFn:
    """Collate for Gemini dataset - samples texts_per_image texts."""
    def __init__(self, tokenizer, texts_per_image=4, hard_neg=False):
        self.tokenizer = tokenizer
        self.texts_per_image = texts_per_image
        self.hard_neg = hard_neg

    def __call__(self, batch):
        images, all_texts, image_ids = zip(*batch)
        images = torch.stack(images, 0)

        selected_texts = []
        selected_ids = []
        hard_neg_texts = []
        for img_idx, texts in enumerate(all_texts):
            if len(texts) >= self.texts_per_image:
                sampled = random.sample(texts, self.texts_per_image)
            else:
                sampled = texts * (self.texts_per_image // len(texts) + 1)
                sampled = sampled[:self.texts_per_image]
            selected_texts.extend(sampled)
            selected_ids.extend([img_idx] * len(sampled))
            if self.hard_neg:
                for t in sampled:
                    neg = _get_spatial_hard_neg(t)
                    if neg is not None:
                        hard_neg_texts.append(neg)

        tokens = self.tokenizer(selected_texts)
        hard_neg_tokens = self.tokenizer(hard_neg_texts) if hard_neg_texts else None
        return images, tokens, torch.tensor(selected_ids), hard_neg_tokens


# ============================================================
# Load and Split Gemini Data
# ============================================================
def load_and_split_gemini(json_path, val_ratio=0.1, seed=42):
    """Load Gemini captions and split into train/val."""
    print(f"Loading Gemini captions from {json_path}...")
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    # Build entries list
    entries = []
    for img_name, texts in data.items():
        img_name = os.path.basename(os.path.normpath(str(img_name)))
        img_path = os.path.join(COCO_TRAIN2014_DIR, img_name)
        if os.path.exists(img_path):
            entries.append((img_path, texts))

    print(f"Total valid entries: {len(entries)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(entries)

    val_size = int(len(entries) * val_ratio)
    val_entries = entries[:val_size]
    train_entries = entries[val_size:]

    print(f"Train: {len(train_entries)} images")
    print(f"Val:   {val_size} images")

    return train_entries, val_entries


def _entry_path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def dedup_entries(entries):
    """Deduplicate entries by image path and merge captions."""
    merged = {}
    caption_sets = {}
    for path, texts in entries:
        key = _entry_path_key(path)
        if key not in merged:
            merged[key] = [path, []]
            caption_sets[key] = set()
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        for t in texts:
            if t is None:
                continue
            t = str(t).strip()
            if not t:
                continue
            if t not in caption_sets[key]:
                caption_sets[key].add(t)
                merged[key][1].append(t)
    return [(path, caps) for path, caps in merged.values() if len(caps) > 0]


def filter_entries_by_keys(entries, blocked_keys):
    kept = []
    removed = 0
    for path, caps in entries:
        if _entry_path_key(path) in blocked_keys:
            removed += 1
        else:
            kept.append((path, caps))
    return kept, removed


def load_coco_captions(ann_path, coco_dir=None):
    """Load COCO original human-annotated captions.
    Returns list of (image_path, [captions]) entries.
    """
    if coco_dir is None:
        coco_dir = COCO_TRAIN2014_DIR
    print(f"Loading COCO captions from {ann_path}...")
    with open(ann_path, encoding='utf-8') as f:
        data = json.load(f)

    # Build image_id → filename map
    id_to_file = {img['id']: img['file_name'] for img in data['images']}

    # Group captions by image_id
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

    print(f"  COCO captions: {len(entries)} images, {sum(len(c) for _, c in entries)} captions")
    return entries


def load_extra_json_captions(json_path, img_dir):
    """Load captions from a JSON file with {filename: [captions]} format.
    img_dir: directory where the images live.
    Returns list of (image_path, [captions]) entries.
    """
    print(f"Loading extra captions from {json_path}...")
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    entries = []
    for img_name, texts in data.items():
        img_name = os.path.basename(os.path.normpath(str(img_name)))
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            entries.append((img_path, texts))
    print(f"  Valid entries: {len(entries)} / {len(data)}")
    return entries


def load_localized_narratives(split='train', coco_dir=None, cache_path=None):
    """Load Localized Narratives COCO subset.

    Each COCO image has one long narrative caption (~100-200 words).
    Returns list of (image_path, [caption]) entries.

    Downloads JSONL from Google Storage on first run, then caches locally.
    Official source: https://google.github.io/localized-narratives/
    """
    if coco_dir is None:
        coco_dir = COCO_TRAIN2014_DIR

    _cache = cache_path or data_path(f"ln_cache_coco_{split}.json")
    if os.path.exists(_cache):
        print(f"  Loading cached Localized Narratives: {_cache}")
        with open(_cache, encoding='utf-8') as f:
            raw = json.load(f)
        entries = [(p, caps) for p, caps in raw if os.path.exists(p)]
        print(f"  Localized Narratives {split}: {len(entries)} images")
        return entries

    # Download JSONL from official Google Storage
    split_map = {
        'train': 'coco_train_captions.jsonl',
        'val':   'coco_val_captions.jsonl',
    }
    filename = split_map.get(split)
    if filename is None:
        raise ValueError(f"Unknown split: {split}")

    jsonl_path = data_path(filename)
    if not os.path.exists(jsonl_path):
        import urllib.request
        url = f"https://storage.googleapis.com/localized-narratives/annotations/{filename}"
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, jsonl_path)
        print(f"  Saved to {jsonl_path}")

    print(f"  Parsing {jsonl_path} ...")
    entries = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            img_id = sample.get('image_id')
            caption = sample.get('caption', '').strip()
            if img_id is None or len(caption) < 10:
                continue
            # COCO train2014 filename format
            img_path = os.path.join(coco_dir, f"COCO_train2014_{int(img_id):012d}.jpg")
            if not os.path.exists(img_path):
                continue
            entries.append((img_path, [caption]))

    with open(_cache, 'w', encoding='utf-8') as f:
        json.dump(entries, f)
    print(f"  Localized Narratives {split}: {len(entries)} images (cached to {_cache})")
    return entries


def load_cocoval_captions(json_path, coco_val_dir=None):
    """Load Gemini/OpenRouter captions for COCO val2014 images."""
    if coco_val_dir is None:
        coco_val_dir = COCO_VAL2014_DIR
    print(f"Loading COCO val2014 captions from {json_path}...")
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    for img_name, texts in data.items():
        img_path = os.path.join(coco_val_dir, os.path.basename(img_name))
        if os.path.exists(img_path) and isinstance(texts, list) and len(texts) >= 2:
            # Filter out safety-blocked placeholder entries
            if any('safety blocked' in t.lower() for t in texts):
                continue
            entries.append((img_path, texts))

    print(f"  COCO val2014 captions: {len(entries)} images")
    return entries


# ============================================================
# Image Resolution Resize (pos_embed interpolation)
# ============================================================
def resize_pos_embed(model, new_img_size):
    """Interpolate positional embedding for a new image resolution.

    MobileCLIP-B uses HybridEmbed (ConvStem, stride=16):
      - pos_embed: (1, num_patches, dim) where num_patches = (img_size/16)^2
      - cls_token: (1, 1, dim) — separate, no change needed

    For 224→256: (1, 196, 768) → (1, 256, 768)
    """
    trunk = model.visual.trunk
    old_pos = trunk.pos_embed.data  # (1, old_n, dim)
    old_n = old_pos.shape[1]
    dim = old_pos.shape[2]
    old_grid = int(old_n ** 0.5)  # 14 for 224
    new_grid = new_img_size // 16  # 16 for 256
    new_n = new_grid * new_grid    # 256

    if old_n == new_n:
        print(f"  pos_embed already matches {new_img_size}x{new_img_size}, no change needed")
        return

    print(f"  Interpolating pos_embed: ({old_grid}x{old_grid}={old_n}) → ({new_grid}x{new_grid}={new_n})")

    # Reshape to 2D grid: (1, old_n, dim) → (1, dim, old_grid, old_grid)
    pos_2d = old_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)

    # Bicubic interpolation to new grid size
    pos_2d_new = F.interpolate(pos_2d.float(), size=(new_grid, new_grid),
                                mode='bicubic', align_corners=False)

    # Reshape back: (1, dim, new_grid, new_grid) → (1, new_n, dim)
    new_pos = pos_2d_new.permute(0, 2, 3, 1).reshape(1, new_n, dim)
    new_pos = new_pos.to(old_pos.dtype)

    trunk.pos_embed = nn.Parameter(new_pos)

    # Update num_patches in patch_embed if it exists
    if hasattr(trunk.patch_embed, 'num_patches'):
        trunk.patch_embed.num_patches = new_n

    # Verify
    dummy = torch.rand(1, 3, new_img_size, new_img_size, device=old_pos.device)
    with torch.no_grad():
        out = model.visual(dummy)
    print(f"  pos_embed: (1, {old_n}, {dim}) → (1, {new_n}, {dim})")
    print(f"  Verification: {new_img_size}x{new_img_size} → output {out.shape} [OK]")


# ============================================================
# Evaluation on Gemini Val Set
# ============================================================
def evaluate_on_sample_flex(model, tokenizer, device, img_size=224, text_len=77):
    """Evaluate on sample set with configurable image size."""
    import csv
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
    all_texts = [texts_map[tid] for tid in all_text_ids]

    transform = CompetitionTransform(img_size)
    model.eval()
    with torch.no_grad():
        text_tokens = tokenizer(all_texts).to(device)
        if text_len < 77:
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


def evaluate_on_gemini_val(model, tokenizer, val_entries, device, batch_size=64, img_size=224):
    """Evaluate on Gemini validation set."""
    transform = CompetitionTransform(img_size)
    model.eval()

    # Build text pool from val set
    all_texts = []
    text_to_id = {}
    gt = {}  # image_idx -> set of text_ids

    for img_idx, (_, texts) in enumerate(val_entries):
        gt[img_idx] = set()
        for t in texts:
            if t not in text_to_id:
                text_to_id[t] = len(all_texts)
                all_texts.append(t)
            gt[img_idx].add(text_to_id[t])

    print(f"  Val set: {len(val_entries)} images, {len(all_texts)} unique texts")

    with torch.no_grad():
        # Encode texts
        text_feats = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            tokens = tokenizer(batch).to(device)
            feats = model.encode_text(tokens)
            text_feats.append(F.normalize(feats.float(), dim=-1).cpu())
        text_feats = torch.cat(text_feats, dim=0)

        # Encode images
        image_feats = []
        for img_path, _ in val_entries:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            feats = model.encode_image(img_t)
            image_feats.append(F.normalize(feats.float(), dim=-1).cpu())
        image_feats = torch.cat(image_feats, dim=0)

    # Compute similarities
    sims = image_feats @ text_feats.t()

    # Recall metrics
    recalls = {1: [], 5: [], 10: []}
    for img_idx in range(len(val_entries)):
        gt_ids = gt[img_idx]
        ranked = sims[img_idx].argsort(descending=True)

        for k in [1, 5, 10]:
            top_k = set(ranked[:k].tolist())
            recalls[k].append(len(gt_ids & top_k) / len(gt_ids))

    results = {k: np.mean(v) * 100 for k, v in recalls.items()}
    return results[1], results[5], results[10]


# ============================================================
# Main Training
# ============================================================
def train(args):
    save_dir = args.save_dir or SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda')

    arch = args.arch
    pretrained = args.pretrained
    img_size = args.img_size

    tokenizer = open_clip.get_tokenizer(arch)

    print(f"\n{'='*70}")
    print(f"  Gemini Full Training (Image + Text)")
    print(f"  Model: {arch}/{pretrained} (img_size={img_size})")
    print(f"  Checkpoint: {args.checkpoint or 'Pretrained'}")
    print(f"  Val ratio: {args.val_ratio}, Epochs: {args.epochs}, LR: {args.lr}")
    print(f"{'='*70}")

    # Load model
    use_no_cls = getattr(args, 'no_cls', False)
    print(f"\nLoading model: {arch}/{pretrained}...")
    model, _, _ = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained)

    transform = CompetitionTransform(img_size)

    # NoCLS: remove cls_token, switch to GAP (seq_len 257→256, NPU-friendly)
    if use_no_cls:
        apply_no_cls_gap(model)
        
    if getattr(args, 'npu_img_192', False):
        if not use_no_cls:
            raise ValueError("--npu_img_192 requires --no_cls during training")
        print("  [NPU-IMG] Applying 192-token truncation during training...")
        # Override the visual forward to truncate to 192 tokens before GAP
        original_forward_features = model.visual.trunk.forward_features
        def forward_features_192(self, x, **kwargs):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.norm_pre(x)
            # MobileCLIP-B's blocks is a Sequential or list of blocks, usually doesn't take attn_mask directly
            # but we pass it along if blocks expects it, else ignore
            if 'attn_mask' in kwargs and kwargs['attn_mask'] is not None:
                # timm pass attn_mask if supported
                try:
                    x = self.blocks(x, **kwargs)
                except TypeError:
                    x = self.blocks(x)
            else:
                x = self.blocks(x)
            # Truncate to first 192 tokens before GAP pooling
            x = x[:, :192, :]
            x = self.norm(x)
            return x
        import types
        model.visual.trunk.forward_features = types.MethodType(forward_features_192, model.visual.trunk)
        
    if getattr(args, 'drop_image_layers', 0) > 0:
        drop_n = args.drop_image_layers
        print(f"  [DropLayer] Dropping last {drop_n} layers from Image Encoder...")
        if hasattr(model.visual.trunk, 'blocks'):
            original_len = len(model.visual.trunk.blocks)
            model.visual.trunk.blocks = model.visual.trunk.blocks[:-drop_n]
            print(f"  [DropLayer] Image blocks reduced: {original_len} -> {len(model.visual.trunk.blocks)}")
        else:
            print("  [DropLayer] WARNING: Could not find 'blocks' in visual.trunk")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        if use_no_cls:
            load_no_cls_checkpoint(model, args.checkpoint)
        else:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            missing, unexpected = model.load_state_dict(
                ckpt.get('state_dict', ckpt), strict=False)
            if missing:
                print(f"  [ckpt] Missing keys (expected if loading N-layer ckpt into full model): {len(missing)}")
            if unexpected:
                print(f"  [ckpt] Unexpected keys: {unexpected}")

    # Interpolate pos_embed AFTER loading checkpoint so checkpoint weights (224px) load cleanly
    if img_size != 224 and hasattr(model, 'visual') and hasattr(model.visual, 'trunk'):
        resize_pos_embed(model, img_size)

    if use_no_cls:
        verify_no_cls(model, img_size=img_size)

    model = model.to(device)

    if getattr(args, 'sigmoid_gelu', False):
        class _SigmoidGELU(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(x * 1.702)
        def _replace_gelu(m):
            for name, child in m.named_children():
                if isinstance(child, nn.GELU):
                    setattr(m, name, _SigmoidGELU())
                else:
                    _replace_gelu(child)
        _replace_gelu(model)
        n = sum(1 for m in model.modules() if isinstance(m, _SigmoidGELU))
        print(f"Replaced {n} nn.GELU -> sigmoid-GELU")

    if getattr(args, 'drop_text_layers', 0) > 0:
        n_drop = args.drop_text_layers
        text_model = getattr(model, 'text', model)
        resblocks = text_model.transformer.resblocks
        n_orig = len(resblocks)
        if n_drop >= n_orig:
            raise ValueError(f"--drop_text_layers {n_drop} >= total layers {n_orig}")
        text_model.transformer.resblocks = nn.ModuleList(list(resblocks)[:n_orig - n_drop])
        print(f"Text transformer: {n_orig} → {n_orig - n_drop} layers (dropped last {n_drop})")

    if getattr(args, 'txt_gap', False):
        apply_txt_gap(model)
        verify_txt_gap(model, text_len=getattr(args, 'text_len', 77))

    # PTQAT: inject INT8 fake-quantization into text encoder
    _ptqat_hooks = []
    if getattr(args, 'ptqat', False):
        _ptqat_hooks = apply_ptqat_text(model)

    if getattr(args, 'freeze_image', False):
        for p in model.visual.parameters():
            p.requires_grad = False
        print("Image encoder FROZEN")

    if getattr(args, 'freeze_text', False):
        text_module = getattr(model, 'text', None) or getattr(model, 'transformer', None)
        if text_module is not None:
            for p in text_module.parameters():
                p.requires_grad = False
        # Also freeze logit_scale
        if hasattr(model, 'logit_scale'):
            model.logit_scale.requires_grad = False
        print("Text encoder FROZEN")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if getattr(args, 'freeze_image', False):
        mode = "text-only"
    elif getattr(args, 'freeze_text', False):
        mode = "image-only"
    else:
        mode = "full model"
    print(f"Params: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total ({mode})")

    # Load and split data
    gemini_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gemini_captions.json")
    train_entries, val_entries = load_and_split_gemini(
        gemini_json, val_ratio=args.val_ratio, seed=42)

    # Global dedup + split guard
    train_entries = dedup_entries(train_entries)
    val_entries = dedup_entries(val_entries)
    gemini_val_keys = {_entry_path_key(p) for p, _ in val_entries}
    train_seen_keys = {_entry_path_key(p) for p, _ in train_entries}
    split_overlap = len(train_seen_keys & gemini_val_keys)
    print(f"  Gemini split overlap after dedup: {split_overlap}")

    # Apply Gemini weight (independent of extra_data)
    if getattr(args, 'no_gemini', False) or args.gemini_weight == 0:
        print("  Gemini: excluded from training (--no_gemini)")
        train_entries = []
    else:
        gemini_repeat = args.gemini_weight
        if gemini_repeat > 1:
            n_gemini = len(train_entries)
            train_entries = train_entries * gemini_repeat
            print(f"  Gemini: {n_gemini} x{gemini_repeat} = {len(train_entries)} images")

    def add_source_entries(source_name, entries, weight=1):
        nonlocal train_entries, train_seen_keys
        entries = dedup_entries(entries)
        entries, removed_val = filter_entries_by_keys(entries, gemini_val_keys)

        unique_entries = []
        removed_seen = 0
        for path, caps in entries:
            key = _entry_path_key(path)
            if key in train_seen_keys:
                removed_seen += 1
                continue
            train_seen_keys.add(key)
            unique_entries.append((path, caps))

        weighted_entries = unique_entries * max(weight, 1)
        train_entries = train_entries + weighted_entries

        print(
            f"  {source_name} added: {len(unique_entries)} unique images "
            f"(weight x{weight} => +{len(weighted_entries)}), "
            f"removed val overlap={removed_val}, removed cross-source overlap={removed_seen}"
        )

    # Optionally add RefCOCO + VG training data
    if args.extra_data:
        print(f"\nLoading extra datasets (RefCOCO + VG)...")
        refcoco_entries = load_refcoco_fullimage(
            ['refcoco', 'refcocoplus', 'refcocog'], split='train',
            cache_dir=data_path("refcoco_cache"), min_texts_per_image=2)
        if getattr(args, 'structured_captions', False):
            from lpcvc2026.modules.data import load_vg_structured_fullimage
            vg_entries = load_vg_structured_fullimage(split='train', min_phrases=4)
            print(f"  VG: using structured (calibration-style) captions")
        else:
            vg_entries = load_vg_fullimage(split='train', min_phrases=4)
        # Convert to (image_path, [texts]) format (drop unique_id)
        extra = [(path, texts) for path, texts, _ in refcoco_entries + vg_entries]
        print(f"  Extra data: {len(extra)} images (RefCOCO+VG)")
        add_source_entries("Extra data", extra, weight=1)
        print(f"  Combined: {len(train_entries)} images")

    # Optionally add Qwen-style short referring expression captions
    if getattr(args, 'qwen_captions', False):
        qwen_json = data_path("vllm_captions", "captions_qwen25_7b.json")
        with open(qwen_json, encoding='utf-8') as f:
            qwen_data = json.load(f)
        qwen_entries = []
        for img_name, texts in qwen_data.items():
            img_name = os.path.basename(os.path.normpath(str(img_name)))
            img_path = os.path.join(COCO_TRAIN2014_DIR, img_name)
            if os.path.exists(img_path) and texts:
                qwen_entries.append((img_path, texts))
        qwen_weight = getattr(args, 'qwen_weight', 2)
        add_source_entries(f"Qwen captions (x{qwen_weight})", qwen_entries, weight=qwen_weight)
        print(f"  Qwen captions added: {len(qwen_entries)} images (weight x{qwen_weight})")

    # Optionally add Localized Narratives (COCO subset)
    if getattr(args, 'localized_narratives', False):
        ln_entries = load_localized_narratives(split='train', coco_dir=COCO_TRAIN2014_DIR)
        ln_weight = getattr(args, 'ln_weight', 1)
        add_source_entries("Localized Narratives", ln_entries, weight=ln_weight)

    # Optionally add COCO original human-annotated captions
    if getattr(args, 'coco_captions', False):
        coco_ann = data_path("annotations_trainval2014", "annotations", "captions_train2014.json")
        coco_entries = load_coco_captions(coco_ann)
        coco_weight = getattr(args, 'coco_weight', 1)
        add_source_entries("COCO captions", coco_entries, weight=coco_weight)

    # Optionally add COCO val2014 captions (Gemini + OpenRouter)
    if getattr(args, 'cocoval_captions', False):
        cocoval_json = data_path("vllm_captions", "captions_cocoval_all.json")
        cocoval_entries = load_cocoval_captions(cocoval_json)
        cocoval_weight = getattr(args, 'cocoval_weight', 1)
        add_source_entries("COCO val2014 captions", cocoval_entries, weight=cocoval_weight)

    # Optionally add GQA scene-graph captions + filtered COCO val captions
    if getattr(args, 'gqa_captions', False):
        gqa_json = data_path("vllm_captions", "captions_gqa_sg.json")
        gqa_entries = load_extra_json_captions(gqa_json, data_path("images"))
        val_filt_json = data_path("vllm_captions", "captions_val_filtered.json")
        val_filt_entries = load_extra_json_captions(val_filt_json, data_path("val2014", "val2014"))
        gqa_all = gqa_entries + val_filt_entries
        gqa_weight = getattr(args, 'gqa_weight', 1)
        add_source_entries("GQA+val filtered captions", gqa_all, weight=gqa_weight)

    final_train_unique = {_entry_path_key(p) for p, _ in train_entries}
    final_overlap = len(final_train_unique & gemini_val_keys)
    print(f"  Final train unique images: {len(final_train_unique)}")
    print(f"  Final train-val overlap: {final_overlap}")

    # Build proxy valset for evaluation
    print(f"\nBuilding proxy validation set...")
    proxy_images, proxy_texts, proxy_gt = build_proxy_valset(
        n_images=300, texts_per_image=4, seed=42)

    # Build datasets
    train_dataset = GeminiDataset(train_entries, transform)
    collate_fn = GeminiCollateFn(tokenizer, texts_per_image=args.texts_per_image, hard_neg=getattr(args, 'hard_neg', False))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True)

    print(f"\nBatch size: {args.batch_size} images")
    print(f"Texts per image: {args.texts_per_image}")
    print(f"Effective batch: {args.batch_size * args.texts_per_image} text-image pairs")

    # Loss and optimizer
    criterion = (SigLIPLoss() if getattr(args, 'siglip', False) else CLIPContrastiveLoss()).to(device)
    print(f"Loss: {'SigLIP (sigmoid BCE)' if getattr(args, 'siglip', False) else 'InfoNCE (softmax CE)'}")
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.98), eps=1e-6)

    if args.grad_accum < 1:
        raise ValueError("--grad_accum must be >= 1")

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = getattr(args, 'warmup_steps', None)
    if warmup_steps is None:
        warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    print(f"\nTraining: {args.epochs} epochs x {len(train_loader)} batches")
    print(f"LR: {args.lr}, Warmup: {warmup_steps}, Optimizer steps/epoch: {updates_per_epoch}")

    best_val_r10 = 0.0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        optimizer.zero_grad(set_to_none=True)

        _text_len = getattr(args, 'text_len', 77)
        for step, (images, tokens, image_ids, hard_neg_tokens) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            if _text_len < 77:
                tokens = tokens[:, :_text_len]
            image_ids = image_ids.to(device)

            with autocast('cuda', dtype=torch.float16):
                img_feats = model.encode_image(images)
                txt_feats = model.encode_text(tokens)
                img_feats = img_feats.repeat_interleave(args.texts_per_image, dim=0)
                hn_feats = None
                if hard_neg_tokens is not None:
                    hn = hard_neg_tokens.to(device, non_blocking=True)
                    if _text_len < 77:
                        hn = hn[:, :_text_len]
                    hn_feats = model.encode_text(hn)
                raw_loss = criterion(img_feats, txt_feats, image_ids, hn_feats)
                loss = raw_loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += raw_loss.item()
            pbar.set_postfix(loss=f"{raw_loss.item():.3f}",
                           lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1}: loss={avg_loss:.4f}  time={elapsed:.0f}s")

        # Validation
        print("  Evaluating on Gemini val set...")
        v_r1, v_r5, v_r10 = evaluate_on_gemini_val(
            model, tokenizer, val_entries, device, batch_size=args.batch_size, img_size=img_size)
        print(f"    Gemini Val: R@1={v_r1:.1f}%  R@5={v_r5:.1f}%  R@10={v_r10:.1f}%")

        # Also evaluate on original sample set
        s_r1, s_r5, s_r10 = evaluate_on_sample_flex(model, tokenizer, device, img_size=img_size, text_len=getattr(args, 'text_len', 77))
        print(f"    Sample:     R@1={s_r1:.1f}%  R@5={s_r5:.1f}%  R@10={s_r10:.1f}%")

        # Evaluate on proxy set
        p_r1, p_r5, p_r10 = evaluate_proxy(
            model, tokenizer, proxy_images, proxy_texts, proxy_gt,
            device=device, img_size=img_size)
        print(f"    Proxy:      R@1={p_r1:.1f}%  R@5={p_r5:.1f}%  R@10={p_r10:.1f}%")

        # Strip PTQAT wrappers before saving so checkpoints are standard format
        if _ptqat_hooks:
            remove_ptqat_text(model, _ptqat_hooks, verbose=False)

        star = ""
        if v_r10 > best_val_r10:
            best_val_r10 = v_r10
            star = " ★"
            torch.save({
                'state_dict': model.state_dict(),
                'gemini_val_r10': v_r10, 'gemini_val_r5': v_r5,
                'sample_r10': s_r10, 'proxy_r10': p_r10,
                'epoch': epoch + 1,
                'arch': arch, 'pretrained': pretrained, 'img_size': img_size,
            }, os.path.join(save_dir, "best.pt"))
            print(f"    ** SAVED best (Gemini val R@10={v_r10:.1f}%) **")

        torch.save({
            'state_dict': model.state_dict(),
            'gemini_val_r10': v_r10, 'gemini_val_r5': v_r5,
            'sample_r10': s_r10, 'proxy_r10': p_r10,
            'epoch': epoch + 1, 'avg_loss': avg_loss,
            'arch': arch, 'pretrained': pretrained, 'img_size': img_size,
        }, os.path.join(save_dir, f"epoch_{epoch+1}.pt"))
        print(f"    Saved epoch_{epoch+1}.pt{star}")

        # Re-apply PTQAT for next epoch
        if _ptqat_hooks:
            _ptqat_hooks = apply_ptqat_text(model, verbose=False)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Gemini Val R@10: {best_val_r10:.1f}%")
    print(f"Save dir: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default=DEFAULT_ARCH,
                       help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default=DEFAULT_PRETRAINED,
                       help="Pretrained weights name")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE,
                       help="Image size (224 for dfn2b, 256 for MobileCLIP2-S4)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Resume from checkpoint (None = use pretrained)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation set ratio (0.1 = 10%)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Images per batch")
    parser.add_argument("--texts_per_image", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--npu_img_192", action="store_true", help="Truncate patches to 192 before GAP (requires --no_cls)")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--extra_data", action="store_true",
                       help="Add RefCOCO + VG training data alongside Gemini captions")
    parser.add_argument("--gemini_weight", type=int, default=1,
                       help="Repeat Gemini data N times when using --extra_data (default: 1). Use 0 to exclude Gemini entirely.")
    parser.add_argument("--no_gemini", action="store_true",
                       help="Exclude Gemini captions from training (use with --extra_data for VG+RefCOCO only)")
    parser.add_argument("--drop_image_layers", type=int, default=0,
                       help="Drop the last N layers of the image encoder transformer (e.g. 1)")
    parser.add_argument("--qwen_captions", action="store_true",
                       help="Add Qwen 7B short referring-expression captions (captions_qwen25_7b.json)")
    parser.add_argument("--qwen_weight", type=int, default=2,
                       help="Repeat Qwen captions N times (default: 2)")
    parser.add_argument("--freeze_image", action="store_true",
                       help="Freeze image encoder, only train text encoder")
    parser.add_argument("--freeze_text", action="store_true",
                       help="Freeze text encoder, only train image encoder + projection. "
                            "Use with --no_cls to adapt GAP projection while keeping text space fixed.")
    parser.add_argument("--no_cls", action="store_true",
                       help="Remove cls_token, use GAP pooling (seq_len 257→256, NPU-friendly). "
                            "Requires fine-tuning to adapt projection head before export.")
    parser.add_argument("--coco_captions", action="store_true",
                       help="Add COCO original human-annotated captions to training data")
    parser.add_argument("--coco_weight", type=int, default=1,
                       help="Repeat COCO captions N times (default: 1)")
    parser.add_argument("--localized_narratives", action="store_true",
                       help="Add Localized Narratives COCO subset (~134K images, 1 long caption each)")
    parser.add_argument("--ln_weight", type=int, default=1,
                       help="Repeat Localized Narratives N times (default: 1)")
    parser.add_argument("--cocoval_captions", action="store_true",
                       help="Add COCO val2014 captions (Gemini+OpenRouter merged, ~12K images)")
    parser.add_argument("--cocoval_weight", type=int, default=1,
                       help="Repeat COCO val2014 captions N times (default: 1)")
    parser.add_argument("--gqa_captions", action="store_true",
                       help="Add GQA scene-graph captions (9,863) + filtered COCO val captions (1,115)")
    parser.add_argument("--gqa_weight", type=int, default=1,
                       help="Repeat GQA+val filtered captions N times (default: 1)")
    parser.add_argument("--structured_captions", action="store_true",
                       help="Use calibration-style structured captions instead of VG region descriptions (requires generate_structured_captions.py to have been run)")
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="Warmup steps (default: min(500, total//10)). Set 0 to disable warmup.")
    parser.add_argument("--sigmoid_gelu", action="store_true",
                       help="Replace nn.GELU with x*sigmoid(1.702x) before training (closes train-deploy gap for QAIRT 2.45)")
    parser.add_argument("--drop_text_layers", type=int, default=0,
                       help="Drop last N layers from text transformer before fine-tuning. "
                            "Reduces text encoder latency ~N/12 * text_latency. "
                            "12→9 layers: ~25%% faster text (~2876 μs vs 3835 μs).")
    parser.add_argument("--txt_gap", action="store_true",
                       help="Replace text encoder argmax(EOS) pooling with global average pooling (GAP). "
                            "Use together with --text_len 24 for fast inference (2642 μs).")
    parser.add_argument("--text_len", type=int, default=77,
                       help="Truncate text tokens to this length during training and eval. "
                            "Must be multiple of 8 for NPU. Use 24 for fast text inference.")
    parser.add_argument("--hard_neg", action="store_true",
                       help="Spatial hard negatives: for each structured caption swap the spatial direction (left↔right, upper↔lower) and add as extra negatives in i2t softmax. Best used with --structured_captions.")
    parser.add_argument("--siglip", action="store_true",
                       help="Use SigLIP loss (pairwise sigmoid BCE) instead of InfoNCE softmax. Each pair is independent; no row-softmax FN pollution. Adds a learned logit_bias.")
    parser.add_argument("--ptqat", action="store_true",
                       help="INT8 fake-quantization during training (PTQAT). Adds STE fake-quant on "
                            "text transformer activations + Linear weights so QNN INT8 calibration "
                            "is more stable at deploy time. Checkpoints are saved without wrappers.")
    args = parser.parse_args()

    if args.save_dir:
        args.save_dir = os.path.join(BASE_DIR, "models", args.save_dir)
    train(args)
