"""
NoCLS GAP utilities for MobileCLIP-B.

Problem:
  256x256 input → 16x16 = 256 patches + 1 cls_token = 257 tokens
  257 is prime → Qualcomm Hexagon NPU crashes during tiling

Solution:
  Remove cls_token, replace pool with Global Average Pooling (GAP)
  Sequence length = 256 (power of 2) → NPU-friendly

MobileCLIP-B specifics (no_embed_class=True):
  - pos_embed covers patch positions only: (1, N_patches, 768)
  - cls_token is separate: (1, 1, 768)
  - Forward: patches + pos_embed → cat([cls, patches]) → transformer → x[:,0]

After this patch:
  - Forward: patches + pos_embed → transformer → x.mean(dim=1) → head (768→512)
  - pos_embed shape is UNCHANGED (already patch-only)

Usage:
    clip_model = open_clip.create_model(...)
    apply_no_cls_gap(clip_model)              # patch in-place
    load_no_cls_checkpoint(clip_model, path)  # load 256 or 224 checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_no_cls_gap(clip_model):
    """Patch MobileCLIP-B trunk in-place: remove cls_token, use GAP.

    Safe to call on both freshly created models and models with loaded weights.
    After this call:
      - trunk.cls_token = None (no longer prepended)
      - trunk.num_prefix_tokens = 0
      - trunk.global_pool = 'avg' (GAP over all patch tokens)
      - pos_embed unchanged (already patch-only in MobileCLIP-B)
    """
    trunk = clip_model.visual.trunk

    # Remove cls_token
    trunk.cls_token = None
    trunk.num_prefix_tokens = 0

    # Switch pool from 'token' (x[:,0]) to 'avg' (x.mean(dim=1))
    trunk.global_pool = 'avg'

    # fc_norm is Identity by default, but set it to norm for GAP stability
    # (timm convention: use fc_norm with GAP, norm+fc_norm=Identity when using cls)
    # For MobileCLIP-B both are Identity so no change needed, but set explicitly
    # trunk.fc_norm is already Identity — leave as-is

    print(f"  [NoCLS] cls_token removed, global_pool → 'avg', seq_len = {trunk.pos_embed.shape[1]}")


def load_no_cls_checkpoint(clip_model, checkpoint_path):
    """Load a checkpoint into a no-CLS model.

    Handles two cases:
    1. Checkpoint trained WITH cls_token (224 pretrained or 256 trained normally):
       - pos_embed shape (1, N, 768): loaded as-is if N matches, or interpolated
       - cls_token key: silently discarded
    2. Checkpoint already no-CLS: loaded directly

    Args:
        clip_model: model already patched with apply_no_cls_gap()
        checkpoint_path: path to .pt file (with 'state_dict' key or raw state_dict)
    """
    trunk = clip_model.visual.trunk
    target_n = trunk.pos_embed.shape[1]  # e.g. 256 for 256-model
    target_dim = trunk.pos_embed.shape[2]

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)

    # ── Handle pos_embed ──────────────────────────────────────────
    pe_key = 'visual.trunk.pos_embed'
    if pe_key in sd:
        old_pe = sd[pe_key]  # (1, old_n, dim)
        old_n = old_pe.shape[1]
        if old_n != target_n:
            old_grid = int(old_n ** 0.5)
            new_grid = int(target_n ** 0.5)
            pe_2d = old_pe.reshape(1, old_grid, old_grid, target_dim).permute(0, 3, 1, 2).float()
            pe_2d_new = F.interpolate(pe_2d, size=(new_grid, new_grid),
                                      mode='bicubic', align_corners=False)
            sd[pe_key] = pe_2d_new.permute(0, 2, 3, 1).reshape(1, target_n, target_dim).to(old_pe.dtype)
            print(f"  [NoCLS] pos_embed interpolated: ({old_grid}x{old_grid}={old_n}) → ({new_grid}x{new_grid}={target_n})")
        else:
            print(f"  [NoCLS] pos_embed shape matches ({target_n}), loaded as-is")

    # ── Discard cls_token ─────────────────────────────────────────
    cls_key = 'visual.trunk.cls_token'
    if cls_key in sd:
        del sd[cls_key]
        print(f"  [NoCLS] cls_token discarded")

    # ── Load with strict=False (safe for any extra keys) ──────────
    missing, unexpected = clip_model.load_state_dict(sd, strict=False)
    # Filter out known-OK missing keys (cls_token itself if re-introduced)
    real_missing = [k for k in missing if 'cls_token' not in k]
    if real_missing:
        print(f"  [NoCLS] WARNING missing keys: {real_missing[:5]}")
    if unexpected:
        print(f"  [NoCLS] WARNING unexpected keys: {unexpected[:5]}")

    print(f"  [NoCLS] Checkpoint loaded: {checkpoint_path}")
    return clip_model


def verify_no_cls(clip_model, img_size=256):
    """Verify the patched model runs correctly and outputs shape (1, 512)."""
    clip_model.eval()
    with torch.no_grad():
        dummy = torch.rand(1, 3, img_size, img_size)
        out = clip_model.encode_image(dummy)
    assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"
    trunk = clip_model.visual.trunk
    seq_len = trunk.pos_embed.shape[1]
    print(f"  [NoCLS] Verify OK: input ({img_size}x{img_size}) → seq_len={seq_len} → output {out.shape}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Text GAP utilities
# ─────────────────────────────────────────────────────────────────────────────

def apply_txt_gap(clip_model):
    """Patch MobileCLIP-B text encoder: replace argmax(EOS) pooling with GAP.

    Standard MobileCLIP-B text:  x[i, argmax(token_ids[i])] → text embedding
    After this patch:             x.mean(dim=1)               → text embedding

    Using mean over ALL text_len positions (including BOS, EOS, padding) so
    the operation is a static-shape mean — NPU-friendly, no masking required.
    The model learns to encode information distributed across all positions.
    """
    import types

    text_model = getattr(clip_model, 'text', clip_model)

    def _gap_forward(self, text):
        x, attn_mask = self._embeds(text)
        # _embeds may return full-context attn_mask (e.g. 77×77) when use_pad_mask=False.
        # Slice it to the actual sequence length so multi_head_attention doesn't error.
        seq_len = x.shape[1]  # NLD: (B, seq_len, D)
        if attn_mask is not None and attn_mask.dim() == 2 and attn_mask.shape[0] != seq_len:
            attn_mask = attn_mask[:seq_len, :seq_len]
        x = self.transformer(x, attn_mask=attn_mask)
        x = self.ln_final(x)
        pooled = x.mean(dim=1)                        # GAP over all seq positions
        if self.text_projection is not None:
            if isinstance(self.text_projection, torch.nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        return pooled

    text_model.forward = types.MethodType(_gap_forward, text_model)
    text_model._pool_type_override = 'gap_mean'
    print(f"  [TxtGAP] Text pooling: argmax(EOS) → mean(all positions)")


def apply_txt_gap_from_checkpoint(clip_model, checkpoint_path):
    """Apply txt_gap patch and load checkpoint (strict=False for safety)."""
    apply_txt_gap(clip_model)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    missing, unexpected = clip_model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if 'cls_token' not in k]
    if real_missing:
        print(f"  [TxtGAP] WARNING missing keys: {real_missing[:5]}")
    if unexpected:
        print(f"  [TxtGAP] WARNING unexpected keys: {unexpected[:5]}")
    print(f"  [TxtGAP] Checkpoint loaded: {checkpoint_path}")
    return clip_model


def verify_txt_gap(clip_model, text_len=77):
    """Verify text GAP model produces (1, 512) embeddings."""
    import open_clip
    clip_model.eval()
    dev = next(clip_model.parameters()).device
    tok = open_clip.get_tokenizer('MobileCLIP-B')
    dummy_text = ["a photo of a cat"]
    tokens = tok(dummy_text).to(dev)
    if text_len != 77:
        tokens = tokens[:, :text_len]
    with torch.no_grad():
        out = clip_model.encode_text(tokens)
    assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"
    print(f"  [TxtGAP] Verify OK: text_len={text_len} → output {out.shape}")
