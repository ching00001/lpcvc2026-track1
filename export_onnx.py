"""
Export prompt-baked ViT-B-16 datacomp_xl for LPCVC 2026.

Bakes the prompt template "a photograph of {}" into the TextEncoderONNX
by prepending prompt tokens [320, 8853, 539] before the input text tokens.

Input to text encoder:  [SOT, t1, t2, ..., tn, EOT, 0, 0, ...]  (77 tokens)
Inside the model:       [SOT, 320, 8853, 539, t1, ..., tn, EOT, 0, ...] (77 tokens)

This is transparent to the competition eval pipeline (same input/output shapes).

Usage:
    python export_prompt_baked.py                   # export + compile + profile
    python export_prompt_baked.py --export_only     # export ONNX only
    python export_prompt_baked.py --local_eval      # export + local eval only
"""

import os
import sys
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import (
    load_sample_dataset, compute_recall_at_k, IMAGE_DIR
)
from npu_utils import apply_no_cls_gap, load_no_cls_checkpoint, verify_no_cls


def preprocess_image_raw(image_path, image_size=224):
    """Competition-style preprocessing: resize + /255.0 only, NO CLIP mean/std.
    Use this with ImageEncoderONNX which has normalization baked in."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    return torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Image encoder pos_embed interpolation (224→256 etc.)
# ─────────────────────────────────────────────────────────────────────────────
def resize_image_pos_embed(clip_model, new_img_size):
    """Interpolate image encoder pos_embed for a new resolution.
    MobileCLIP-B: pos_embed (1, N, dim), stride=16, cls_token separate."""
    trunk = clip_model.visual.trunk
    old_pos = trunk.pos_embed.data  # (1, old_n, dim)
    old_n = old_pos.shape[1]
    dim = old_pos.shape[2]
    old_grid = int(old_n ** 0.5)
    new_grid = new_img_size // 16
    new_n = new_grid * new_grid
    if old_n == new_n:
        return
    pos_2d = old_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    pos_2d_new = F.interpolate(pos_2d.float(), size=(new_grid, new_grid),
                                mode='bicubic', align_corners=False)
    new_pos = pos_2d_new.permute(0, 2, 3, 1).reshape(1, new_n, dim).to(old_pos.dtype)
    trunk.pos_embed = nn.Parameter(new_pos)
    if hasattr(trunk.patch_embed, 'num_patches'):
        trunk.patch_embed.num_patches = new_n
    print(f"  Image pos_embed interpolated: ({old_grid}x{old_grid}={old_n}) → ({new_grid}x{new_grid}={new_n})")


# ─────────────────────────────────────────────────────────────────────────────
#  Text encoder positional embedding / attn_mask expansion
# ─────────────────────────────────────────────────────────────────────────────
def expand_text_pos_and_mask(text_model, target_len: int):
    """Resize positional_embedding and attn_mask to target_len.

    target_len < old_len (e.g. 64): TRUNCATE — slice first target_len positions.
    target_len > old_len (e.g. 80): EXPAND  — copy last position for PAD slots.
    target_len == old_len (77):     no-op.

    Returns (new_pos_embed, new_attn_mask) as Parameters/Tensors.
    """
    old_pos = text_model.positional_embedding  # (77, D)
    old_len = old_pos.shape[0]

    if target_len == old_len:
        return old_pos, text_model.attn_mask.clone()

    if target_len < old_len:
        # Truncate: slice to first target_len positions
        new_pos = nn.Parameter(old_pos.data[:target_len].clone())
        new_mask = text_model.attn_mask[:target_len, :target_len].clone()
        return new_pos, new_mask

    # Expand: target_len > old_len
    dim = old_pos.shape[1]
    new_pos = torch.zeros(target_len, dim, dtype=old_pos.dtype)
    new_pos[:old_len] = old_pos.data
    new_pos[old_len:] = old_pos.data[-1]  # copy last position for PAD slots

    old_mask = text_model.attn_mask  # (77, 77), upper-tri = -inf
    new_mask = torch.full((target_len, target_len), float('-inf'), dtype=old_mask.dtype)
    new_mask[:old_len, :old_len] = old_mask
    for i in range(old_len, target_len):
        new_mask[i, :i+1] = 0.0

    return nn.Parameter(new_pos), new_mask


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX Wrappers
# ─────────────────────────────────────────────────────────────────────────────
class ImageEncoderONNX(nn.Module):
    """Image encoder matching competition eval pipeline.

    Default: no normalization (new training uses [0,1] directly).
    With bake_norm=True: bakes CLIP mean/std for old checkpoints trained with CLIP normalization.
    """
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, clip_model, bake_norm=False):
        super().__init__()
        self.visual = clip_model.visual
        self.bake_norm = bake_norm
        if bake_norm:
            self.register_buffer('mean', torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(self.CLIP_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.bake_norm:
            image = (image - self.mean) / self.std
        return self.visual(image)


class NPUImageEncoderWrapper(nn.Module):
    """NPU-friendly image encoder wrapper with fixed 192-token sequence.

    Pipeline:
      1) patch embedding
      2) keep first 192 tokens
      3) add sliced positional embedding
      4) transformer + ln_post
      5) global average pooling over tokens
      6) projection head

    This avoids sequence lengths like 196 that can be problematic on some DSP tiling paths.
    """
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, clip_model, keep_tokens: int = 192, bake_norm: bool = False):
        super().__init__()
        self.keep_tokens = keep_tokens
        self.bake_norm = bake_norm

        visual = clip_model.visual
        self.transformer_blocks = None
        self.transformer = None

        # Path 1: TimmModel visual wrapper (used by MobileCLIP-B in this repo).
        if hasattr(visual, 'trunk'):
            trunk = visual.trunk
            self.patch_embed = trunk.patch_embed
            self.positional_embedding = trunk.pos_embed
            self.transformer_blocks = trunk.blocks
            self.ln_post = trunk.norm
            self.image_projection = trunk.head
            self.impl = 'timm'
        else:
            # Path 2: generic ViT-like naming requested by the user.
            self.patch_embed = getattr(visual, 'patch_embed', None)
            if self.patch_embed is None:
                self.patch_embed = getattr(visual, 'conv1', None)
            self.positional_embedding = getattr(visual, 'positional_embedding', None)
            if self.positional_embedding is None:
                self.positional_embedding = getattr(visual, 'pos_embed', None)
            self.transformer = getattr(visual, 'transformer', None)
            self.ln_post = getattr(visual, 'ln_post', None)
            if self.ln_post is None:
                self.ln_post = getattr(visual, 'norm', None)
            self.image_projection = getattr(visual, 'image_projection', None)
            if self.image_projection is None:
                self.image_projection = getattr(visual, 'head', None)
            self.impl = 'generic'

        if self.patch_embed is None:
            raise RuntimeError("NPUImageEncoderWrapper requires patch_embed/conv1 on visual model")
        if self.positional_embedding is None:
            raise RuntimeError("NPUImageEncoderWrapper requires positional embedding on visual model")
        if self.ln_post is None:
            raise RuntimeError("NPUImageEncoderWrapper requires ln_post/norm on visual model")
        if self.transformer_blocks is None and self.transformer is None:
            raise RuntimeError("NPUImageEncoderWrapper requires transformer blocks or transformer module")

        if bake_norm:
            self.register_buffer('mean', torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(self.CLIP_STD).view(1, 3, 1, 1))

        if self.positional_embedding.dim() == 3:
            pos_len = int(self.positional_embedding.shape[1])
        else:
            pos_len = int(self.positional_embedding.shape[0])
        if pos_len < self.keep_tokens:
            raise RuntimeError(
                f"NPUImageEncoderWrapper needs at least {self.keep_tokens} positional tokens, got {pos_len}"
            )

    def _slice_pos(self, token_count: int, x: torch.Tensor) -> torch.Tensor:
        if self.positional_embedding.dim() == 3:
            pos_full = self.positional_embedding
        else:
            pos_full = self.positional_embedding.unsqueeze(0)

        # If pos_embed includes cls position, skip index 0.
        if pos_full.shape[1] == token_count + 1:
            pos = pos_full[:, 1:1 + self.keep_tokens, :]
        else:
            pos = pos_full[:, :self.keep_tokens, :]
        return pos.to(dtype=x.dtype, device=x.device)

    def _apply_projection(self, pooled: torch.Tensor) -> torch.Tensor:
        proj = self.image_projection
        if proj is None:
            return pooled
        if isinstance(proj, nn.Identity):
            return pooled
        if isinstance(proj, nn.Module):
            return proj(pooled)
        return pooled @ proj

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.bake_norm:
            image = (image - self.mean) / self.std

        x = self.patch_embed(image)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BND

        token_count = int(x.shape[1])
        x = x[:, :self.keep_tokens, :]
        x = x + self._slice_pos(token_count, x)

        if self.transformer_blocks is not None:
            for blk in self.transformer_blocks:
                x = blk(x)
        else:
            # Generic transformer path. If transformer expects LND, flip once.
            if getattr(self.transformer, 'batch_first', True):
                x = self.transformer(x)
            else:
                x = self.transformer(x.transpose(0, 1)).transpose(0, 1)

        x = self.ln_post(x)
        pooled = x.mean(dim=1)
        return self._apply_projection(pooled)



class PaddedImageEncoderWrapper(nn.Module):
    """NoCLS image encoder with zero-padding to a fixed token count.

    Pipeline:
      1. patch_embed: (1,3,224,224) → (1, N, D)  where N=196
      2. add pos_embed (patch-only, no CLS)
      3. zero-pad to (1, pad_to, D)  — extra tokens are all-zero
      4. transformer blocks (all tokens attend each other)
      5. slice back first N real tokens, ln_post, GAP, projection

    Requires apply_no_cls_gap() to have been called beforehand.
    """
    def __init__(self, clip_model, pad_to: int = 200, bake_norm: bool = False):
        super().__init__()
        CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

        trunk = clip_model.visual.trunk
        self.patch_embed = trunk.patch_embed
        # pos_embed is patch-only after apply_no_cls_gap: (1, N, D)
        pos = trunk.pos_embed.data.clone().detach()
        self.register_buffer('pos_embed', pos)
        self.transformer_blocks = trunk.blocks
        self.ln_post = trunk.norm
        self.image_projection = trunk.head
        self.n_real = int(pos.shape[1])   # 196 for 224-px input
        self.pad_to  = pad_to
        self.bake_norm = bake_norm
        if bake_norm:
            self.register_buffer('mean', torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std',  torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        # Pre-build static zero pad (constant, exported as initializer)
        D = int(pos.shape[2])
        n_pad = pad_to - self.n_real
        if n_pad <= 0:
            raise ValueError(f"pad_to={pad_to} must be larger than n_real={self.n_real}")
        self.register_buffer('zero_pad', torch.zeros(1, n_pad, D))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.bake_norm:
            image = (image - self.mean) / self.std
        x = self.patch_embed(image)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)   # (1, 196, D)
        x = x + self.pos_embed                  # add positional embedding
        x = torch.cat([x, self.zero_pad.expand(x.shape[0], -1, -1)], dim=1)  # (1, pad_to, D)
        for blk in self.transformer_blocks:
            x = blk(x)
        # Use only real tokens for pooling
        x_real = x[:, :self.n_real, :]           # (1, 196, D)
        x_real = self.ln_post(x_real)
        pooled = x_real.mean(dim=1)              # GAP
        proj = self.image_projection
        if proj is None or isinstance(proj, nn.Identity):
            return pooled
        if isinstance(proj, nn.Module):
            return proj(pooled)
        return pooled @ proj


class TextEncoderONNXPrompt(nn.Module):
    """
    Text encoder with baked-in prompt template.

    Prepends prompt tokens after SOT before the actual text tokens.
    Input:    [SOT, t1, t2, ..., tn, EOT, PAD, ...]  (text_len tokens)
    Internal: [SOT, p1, p2, p3, t1, ..., tn, EOT, PAD, ...]  (text_len tokens)

    The prompt shifts all content tokens right by prompt_len positions.
    """
    def __init__(self, clip_model, prompt_token_ids: torch.Tensor, text_len: int = 77,
                 use_mask_sum: bool = False, use_text_gap: bool = False):
        """
        Args:
            clip_model: OpenCLIP model
            prompt_token_ids: 1D tensor of prompt tokens (no SOT/EOT),
                              e.g. tensor([320, 8853, 539]) for "a photograph of"
            text_len: sequence length (77 default, 80 for NPU-friendly)
        """
        super().__init__()
        self.text_len = text_len
        self.use_mask_sum = use_mask_sum
        self.use_text_gap = use_text_gap
        text_model = getattr(clip_model, 'text', clip_model)
        self.token_embedding = text_model.token_embedding
        self.transformer = text_model.transformer
        self.ln_final = text_model.ln_final
        self.text_projection = text_model.text_projection
        self.token_width = int(self.token_embedding.embedding_dim)

        # Expand positional_embedding and attn_mask if text_len > 77
        new_pos, new_mask = expand_text_pos_and_mask(text_model, text_len)
        self.positional_embedding = new_pos

        # Replace -inf with -1e9 for QNN compatibility
        new_mask = torch.where(
            new_mask == float('-inf'),
            torch.tensor(-1e9, dtype=new_mask.dtype),
            new_mask
        )
        self.register_buffer('attn_mask', new_mask)

        # Prompt tokens (without SOT/EOT), shape (1, prompt_len)
        self.register_buffer('prompt_tokens', prompt_token_ids.unsqueeze(0).long())
        self.prompt_len = prompt_token_ids.shape[0]

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text: (B, text_len) = [SOT, t1, t2, ..., EOT, PAD, PAD, ...]
        B = text.shape[0]

        sot = text[:, :1]
        content = text[:, 1:]
        prompt = self.prompt_tokens.expand(B, -1)
        new_text = torch.cat([sot, prompt, content], dim=1)[:, :self.text_len]

        x = self.token_embedding(new_text)
        x = x + self.positional_embedding
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)

        if self.use_text_gap:
            # Fully static pooling path: average all tokens.
            x = x.mean(dim=1)
        elif self.use_mask_sum:
            # NPU-friendly static pooling: manual repeat avoids dynamic BMM and broadcast expansion.
            text_f = new_text.float()
            max_vals = text_f.max(dim=-1, keepdim=True)[0]            # (B, 1)
            mask = (text_f == max_vals).to(x.dtype).unsqueeze(-1)     # (B, L, 1)
            mask = mask.repeat(1, 1, self.token_width)                # (B, L, D)
            x = (x * mask).sum(dim=1)                                 # (B, D)
        else:
            x = x[torch.arange(B), new_text.argmax(dim=-1)]

        x = x @ self.text_projection
        return x


class TextEncoderONNXBaseline(nn.Module):
    """Standard text encoder (no prompt) — supports both CLIP and CustomTextCLIP."""
    def __init__(self, clip_model, text_len: int = 77, use_mask_sum: bool = False,
                 use_text_gap: bool = False):
        super().__init__()
        self.text_len = text_len
        self.use_mask_sum = use_mask_sum
        self.use_text_gap = use_text_gap
        text_model = getattr(clip_model, 'text', clip_model)
        self.token_embedding = text_model.token_embedding
        self.transformer = text_model.transformer
        self.ln_final = text_model.ln_final
        self.text_projection = text_model.text_projection
        self.token_width = int(self.token_embedding.embedding_dim)

        new_pos, new_mask = expand_text_pos_and_mask(text_model, text_len)
        self.positional_embedding = new_pos

        new_mask = torch.where(
            new_mask == float('-inf'),
            torch.tensor(-1e9, dtype=new_mask.dtype),
            new_mask
        )
        self.register_buffer('attn_mask', new_mask)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)

        if self.use_text_gap:
            # Fully static pooling path: average all tokens.
            x = x.mean(dim=1)
        elif self.use_mask_sum:
            # NPU-friendly static pooling: manual repeat avoids dynamic BMM and broadcast expansion.
            text_f = text.float()
            max_vals = text_f.max(dim=-1, keepdim=True)[0]            # (B, 1)
            mask = (text_f == max_vals).to(x.dtype).unsqueeze(-1)     # (B, L, 1)
            mask = mask.repeat(1, 1, self.token_width)                # (B, L, D)
            x = (x * mask).sum(dim=1)                                 # (B, D)
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        x = x @ self.text_projection
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  NPU-optimized zero-slice text encoder (fixed 64 input)
# ─────────────────────────────────────────────────────────────────────────────
class NPUTextEncoderWrapper(nn.Module):
    """
    Zero-slice NPU-friendly text encoder for MobileCLIP-B.

    Input must already be (B, 64). This wrapper performs all shape trimming in __init__
    so the exported ONNX forward graph has no runtime slicing.
    """
    def __init__(self, clip_model, use_text_gap: bool = False, text_len: int = 64):
        super().__init__()
        self.text_model = getattr(clip_model, 'text', clip_model)
        self.use_text_gap = use_text_gap
        self.expected_text_len = text_len
        self.token_width = int(self.text_model.token_embedding.embedding_dim)

        # Pre-trim positional embedding once so forward has zero slice ops.
        pos = self.text_model.positional_embedding
        if pos.dim() == 2:
            if pos.shape[0] < self.expected_text_len:
                raise RuntimeError(
                    f"NPUTextEncoderWrapper requires positional_embedding length >= {self.expected_text_len}, "
                    f"got {pos.shape[0]}"
                )
            pos_64 = pos[:self.expected_text_len, :].clone().detach()
        else:
            if pos.shape[1] < self.expected_text_len:
                raise RuntimeError(
                    f"NPUTextEncoderWrapper requires positional_embedding length >= {self.expected_text_len}, "
                    f"got {pos.shape[1]}"
                )
            pos_64 = pos[:, :self.expected_text_len, :].clone().detach()
        self.text_model.positional_embedding = nn.Parameter(pos_64)

        # Keep causal mask explicit and QNN-safe, then pre-trim to 64x64 once.
        raw_mask = getattr(self.text_model, 'attn_mask', None)
        if raw_mask is not None:
            safe_mask = torch.where(
                raw_mask == float('-inf'),
                torch.full_like(raw_mask, -1e9),
                raw_mask
            )
            if safe_mask.shape[0] < self.expected_text_len or safe_mask.shape[1] < self.expected_text_len:
                raise RuntimeError(
                    f"NPUTextEncoderWrapper requires attn_mask >= {self.expected_text_len}x{self.expected_text_len}, "
                    f"got {tuple(safe_mask.shape)}"
                )
            static_mask = safe_mask[:self.expected_text_len, :self.expected_text_len].clone().detach()
            static_mask = static_mask.to(dtype=self.text_model.token_embedding.weight.dtype)
            self.register_buffer('static_mask', static_mask)
        else:
            self.static_mask = None

    def forward(self, text_input: torch.Tensor) -> torch.Tensor:
        if text_input.dim() != 2 or text_input.shape[1] != self.expected_text_len:
            raise RuntimeError(
                f"NPUTextEncoderWrapper expects text_input shape (B, {self.expected_text_len}), "
                f"got {tuple(text_input.shape)}"
            )

        x = self.text_model.token_embedding(text_input)
        x = x + self.text_model.positional_embedding

        transformer = self.text_model.transformer
        if getattr(transformer, 'batch_first', True):
            x = transformer(x, attn_mask=self.static_mask) if self.static_mask is not None else transformer(x)
        else:
            x = x.permute(1, 0, 2)
            x = transformer(x, attn_mask=self.static_mask) if self.static_mask is not None else transformer(x)
            x = x.permute(1, 0, 2)

        x = self.text_model.ln_final(x)

        if self.use_text_gap:
            pooled = x.mean(dim=1)
        else:
            # Static EOT mask: manual repeat to avoid implicit broadcasting on DSP.
            text_f = text_input.float()
            max_vals = text_f.max(dim=-1, keepdim=True)[0]         # (B, 1)
            mask = (text_f == max_vals).to(x.dtype).unsqueeze(-1)  # (B, 64, 1)
            mask = mask.repeat(1, 1, self.token_width)             # (B, 64, D)
            pooled = (x * mask).sum(dim=1)                         # (B, D)

        proj = getattr(self.text_model, 'text_projection', None)
        if proj is not None:
            pooled = pooled @ proj

        return pooled


# ─────────────────────────────────────────────────────────────────────────────
#  Eval-Compatible Wrappers (external 77/224 → internal 80/256)
# ─────────────────────────────────────────────────────────────────────────────
class TextEncoderWrapper(nn.Module):
    """Accepts external shape (B, 77) from the official evaluator.
    Adjusts to target_len before feeding the internal encoder:
      - target_len > external_len (e.g. 80): pad zeros on the right
      - target_len < external_len (e.g. 64): slice first target_len tokens
      - target_len == external_len (77): pass through unchanged

    NPU alignment guide:
      80 tokens / 8 heads = 10 per head  → DSP Error 0x27 (crash)
      64 tokens / 8 heads =  8 per head  → NPU-friendly ✓
    """
    def __init__(self, internal_encoder: nn.Module, external_len: int = 77, target_len: int = 64):
        super().__init__()
        self.encoder = internal_encoder
        self.external_len = external_len
        self.target_len = target_len

    def forward(self, text_input: torch.Tensor) -> torch.Tensor:
        # text_input: (B, external_len=77)
        if self.target_len < text_input.shape[1]:
            # Truncate: e.g. 77 → 64 (safe: tokens beyond 64 are zero-padding)
            text_input = text_input[:, :self.target_len]
        elif self.target_len > text_input.shape[1]:
            # Pad: e.g. 77 → 80
            B = text_input.shape[0]
            pad = torch.zeros(B, self.target_len - text_input.shape[1],
                              dtype=text_input.dtype, device=text_input.device)
            text_input = torch.cat([text_input, pad], dim=1)
        return self.encoder(text_input)


class ImageEncoderWrapper(nn.Module):
    """Accepts external shape (B, 3, 224, 224) from the official evaluator.
    Bilinear-upsamples to (B, 3, target_size, target_size) before feeding the internal encoder.

    Use this when the model is trained/exported with img_size=256 but the
    competition eval pipeline always sends (1, 3, 224, 224) images.
    """
    def __init__(self, internal_encoder: nn.Module, target_size: int = 256):
        super().__init__()
        self.encoder = internal_encoder
        self.target_size = target_size

    def forward(self, image_input: torch.Tensor) -> torch.Tensor:
        # image_input: (B, 3, 224, 224)
        if image_input.shape[-1] != self.target_size:
            image_input = F.interpolate(
                image_input,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
            )  # (B, 3, target_size=256, target_size=256)
        return self.encoder(image_input)


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: get prompt tokens
# ─────────────────────────────────────────────────────────────────────────────
def get_prompt_tokens(prompt_text: str, model_name: str = 'ViT-B-16') -> torch.Tensor:
    """
    Tokenize a prompt and extract the content tokens (no SOT/EOT).

    Example: "a photograph of" → tensor([320, 8853, 539])
    """
    tokenizer = open_clip.get_tokenizer(model_name)
    tokens = tokenizer([prompt_text])[0]  # (77,)
    # SOT is at position 0 (49406), find EOT (49407)
    eot_pos = (tokens == 49407).nonzero(as_tuple=True)[0][0].item()
    prompt_tokens = tokens[1:eot_pos]  # Content tokens only
    return prompt_tokens


def tokenize_with_length(tokenizer, texts, text_len: int = 77) -> torch.Tensor:
    """Tokenize texts directly at the target context length when supported."""
    if text_len == 77:
        return tokenizer(texts)
    try:
        return tokenizer(texts, context_length=text_len)
    except TypeError:
        tokens = tokenizer(texts)
        B = tokens.shape[0]
        padded = torch.zeros(B, text_len, dtype=tokens.dtype)
        copy_len = min(tokens.shape[1], text_len)
        padded[:, :copy_len] = tokens[:, :copy_len]
        return padded


# ─────────────────────────────────────────────────────────────────────────────
#  Local evaluation
# ─────────────────────────────────────────────────────────────────────────────
def eval_text_encoder_wrapper(img_wrapper, txt_wrapper, tokenizer, image_size=224, label="", text_len=77):
    """Evaluate an image+text encoder wrapper pair on the sample dataset."""
    image_names, all_texts, gt = load_sample_dataset()

    # Encode images
    image_embs = []
    with torch.no_grad():
        for img_name in image_names:
            img = preprocess_image_raw(os.path.join(IMAGE_DIR, img_name), image_size)
            image_embs.append(img_wrapper(img))
    image_embs = F.normalize(torch.cat(image_embs, 0), dim=-1)

    # Encode texts at the exact target length.
    text_tokens = tokenize_with_length(tokenizer, all_texts, text_len=text_len)
    with torch.no_grad():
        text_embs = txt_wrapper(text_tokens)
    text_embs = F.normalize(text_embs, dim=-1)

    # Compute recall
    sim = image_embs @ text_embs.T
    r1 = compute_recall_at_k(sim, gt, k=1)
    r5 = compute_recall_at_k(sim, gt, k=5)
    r10 = compute_recall_at_k(sim, gt, k=10)
    print(f"  {label:<50} R@1={r1*100:5.1f}%  R@5={r5*100:5.1f}%  R@10={r10*100:5.1f}%")
    return r1, r5, r10


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_only", action="store_true", help="Only export ONNX")
    parser.add_argument("--local_eval", action="store_true", help="Export + local eval only")
    parser.add_argument("--prompt", type=str, default="a photograph of",
                        help="Prompt template prefix (default: 'a photograph of')")
    parser.add_argument("--no_prompt", action="store_true", help="No prompt (use baseline text encoder)")
    parser.add_argument("--standard_gelu", action="store_true", help="Skip sigmoid-GELU replacement (use standard nn.GELU / Erf ops)")
    parser.add_argument("--no_profile", action="store_true", help="Compile but skip profiling")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint (.pt) to load state_dict from")
    parser.add_argument("--arch", type=str, default="ViT-B-16-quickgelu",
                        help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="dfn2b",
                        help="Pretrained weights name")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (auto if not set)")
    parser.add_argument("--bake_norm", action="store_true",
                        help="Bake CLIP mean/std normalization into image encoder (for old checkpoints trained with CLIP norm)")
    parser.add_argument("--text_len", type=int, default=77,
                        help="Text sequence length (default 77, use 80 for NPU-friendly tiling)")
    parser.add_argument("--eval_wrapper", action="store_true",
                        help="Wrap model to accept external shapes (1,77)/(1,3,224,224) "
                             "while running internally at text_len/img_size. "
                             "Use with --img_size 256 --text_len 80 for 256-trained models.")
    parser.add_argument("--npu_mask_sum", action="store_true",
                        help="Replace dynamic Gather with static repeat-mask + sum pooling in text encoder. "
                            "Avoids dynamic BMM and implicit broadcast expansion on DSP.")
    parser.add_argument("--npu_text_gap", action="store_true",
                        help="Use fully static Text GAP pooling (mean over tokens) in text encoder. "
                            "Removes EOS gather/mask ops from text pooling path.")
    parser.add_argument("--no_cls", action="store_true",
                        help="Remove cls_token, use GAP pooling. "
                             "Sequence length 256 (vs 257 with cls), NPU-friendly. "
                             "Use with --img_size 256.")
    parser.add_argument("--npu_img_192", action="store_true",
                        help="Use NPU image wrapper: keep first 192 patch tokens, then GAP pooling. "
                            "Designed for DSP-friendly static sequence length.")
    parser.add_argument("--img_pad_to", type=int, default=0,
                        help="NoCLS + zero-pad image tokens to this count (e.g. 200). "
                             "Must be used with --no_cls. Pads the 196 patch tokens to pad_to "
                             "before transformer, then GAPs over first 196 real tokens only.")
    parser.add_argument("--qairt_version", type=str, default=None,
                        help="QAIRT SDK version for compile (e.g. '2.43.0', '2.45.0', 'latest'). "
                             "Use '2.43.0' to avoid the ErfDummyLayoutInferer bug in 2.45.0.")
    parser.add_argument("--txt_w8a8", action="store_true",
                        help="Compile text encoder with W8A8 INT8 quantization. "
                             "Image encoder stays FP16. May reduce text latency ~20-30%%.")
    parser.add_argument("--drop_text_layers", type=int, default=0,
                        help="Drop last N layers from text transformer before export. "
                             "Use with 9-layer checkpoints (N=3). "
                             "Checkpoint loading uses strict=False to handle both 9 and 12 layer ckpts.")
    parser.add_argument("--drop_image_layers", type=int, default=0,
                        help="Drop last N layers from image encoder before export.")
    parser.add_argument("--smooth_quant", action="store_true",
                        help="Apply SmoothQuant to text encoder before ONNX export. "
                             "Reduces activation outliers so QNN int8 PTQ stays accurate. "
                             "Use with --txt_w8a8 or compile-time --quantize_full_type int8.")
    parser.add_argument("--smooth_quant_alpha", type=float, default=0.5,
                        help="SmoothQuant alpha (0.0-1.0). Higher = more smoothing. Default 0.5; try 0.85 if still unstable.")
    args = parser.parse_args()

    if args.no_prompt:
        args.prompt = ""

    # Auto-detect arch/pretrained from checkpoint if available
    if args.checkpoint and args.arch == "ViT-B-16-quickgelu":
        ckpt_meta = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'arch' in ckpt_meta:
            args.arch = ckpt_meta['arch']
            args.pretrained = ckpt_meta.get('pretrained', args.pretrained)
            args.img_size = ckpt_meta.get('img_size', args.img_size)
            print(f"  Auto-detected from checkpoint: {args.arch}/{args.pretrained} img_size={args.img_size}")
        del ckpt_meta

    MODEL_NAME = args.arch
    PRETRAINED = args.pretrained
    IMAGE_SIZE = args.img_size
    TEXT_LEN = args.text_len
    EMBED_DIM = 512
    if args.out_dir:
        OUT_DIR = args.out_dir
    elif args.checkpoint:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        parent_name = os.path.basename(os.path.dirname(args.checkpoint))
        OUT_DIR = f"onnx_models/{parent_name}_{ckpt_name}"
    else:
        OUT_DIR = "onnx_B16_datacomp_prompt"

    print(f"{'='*70}")
    print(f"  Prompt-Baked Text Encoder Export")
    print(f"  Model: {MODEL_NAME} / {PRETRAINED}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Prompt: \"{args.prompt} {{}}\"")
    print(f"  Text length: {TEXT_LEN} (default 77)")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*70}")

    if args.npu_img_192 and IMAGE_SIZE != 224:
        raise ValueError("--npu_img_192 requires --img_size 224 to keep input image size unchanged")

    # ── Load model ───────────────────────────────────────────────
    print(f"\n  Loading {MODEL_NAME}/{PRETRAINED}...")
    clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    
    if args.no_cls:
        apply_no_cls_gap(clip_model)
        
    if getattr(args, 'drop_image_layers', 0) > 0:
        drop_n = args.drop_image_layers
        print(f"  [DropLayer] Dropping last {drop_n} layers from Image Encoder...")
        if hasattr(clip_model.visual.trunk, 'blocks'):
            clip_model.visual.trunk.blocks = clip_model.visual.trunk.blocks[:-drop_n]
        else:
            print("  [DropLayer] WARNING: Could not find 'blocks' in visual.trunk")

    if args.checkpoint:
        if args.no_cls:
            load_no_cls_checkpoint(clip_model, args.checkpoint)
        else:
            import torch as _torch
            ckpt = _torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            strict = (args.drop_text_layers == 0 and getattr(args, 'drop_image_layers', 0) == 0)
            missing, unexpected = clip_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=strict)
            if not strict and unexpected:
                print(f"  [drop_layers] Ignored {len(unexpected)} extra keys from checkpoint")
        print(f"  Loaded checkpoint: {args.checkpoint}")
        
    # Interpolate image pos_embed AFTER loading checkpoint so standard checkpoints load cleanly
    if IMAGE_SIZE != 224 and hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'trunk'):
        resize_image_pos_embed(clip_model, IMAGE_SIZE)
    if args.no_cls:
        verify_no_cls(clip_model, img_size=IMAGE_SIZE)
        print(f"  [NoCLS] seq_len={clip_model.visual.trunk.pos_embed.shape[1]} (NPU-friendly)")

    # Always print image token layout so export logs can confirm cls on/off state.
    if hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'trunk') and hasattr(clip_model.visual.trunk, 'pos_embed'):
        trunk = clip_model.visual.trunk
        patch_tokens = int(trunk.pos_embed.shape[1])
        has_cls = getattr(trunk, 'cls_token', None) is not None
        seq_len = patch_tokens + (1 if has_cls else 0)
        print(f"  [Image] transformer seq_len={seq_len} (patch={patch_tokens}, cls={'on' if has_cls else 'off'})")
        if args.no_cls:
            expected_patch_tokens = (IMAGE_SIZE // 16) * (IMAGE_SIZE // 16)
            if has_cls or patch_tokens != expected_patch_tokens:
                raise RuntimeError(
                    f"[NoCLS] invalid visual layout: patch_tokens={patch_tokens}, "
                    f"cls_token={'present' if has_cls else 'none'}, expected_patch_tokens={expected_patch_tokens}"
                )

    # ── Sigmoid-GELU: avoids Erf ops that break QAIRT 2.42/2.43/2.45 compile ─
    # All available QAIRT versions (2.42, 2.43, 2.45) have a broken
    # ErfDummyLayoutInferer stub that fails whenever Erf ops appear in the graph.
    # MobileCLIP-B uses exact GELU (14 Erf ops in img encoder, 12 in txt encoder).
    #
    # Tanh-GELU was tested but runs at ~76ms on XR2 Gen 2 (6x slower than sigmoid).
    # Sigmoid-GELU: x * sigmoid(1.702 * x)
    #   - Runs as efficient NPU LUT kernel (~12.8ms img, ~3.8ms txt)
    #   - Max error vs exact GELU: ~0.020
    #   - LB drop: ~-0.0143 (multiplicative factor 0.9764 vs standard GELU)
    #   - Only viable Erf-free option until QAIRT Erf regression is fixed

    class _SigmoidGELU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x * 1.702)

    def _replace_gelu_sigmoid(module):
        for name, child in module.named_children():
            if isinstance(child, nn.GELU):
                setattr(module, name, _SigmoidGELU())
            else:
                _replace_gelu_sigmoid(child)

    if args.standard_gelu:
        print("  [GELU] Keeping standard nn.GELU (Erf ops) — testing QAIRT fix")
    else:
        _replace_gelu_sigmoid(clip_model)
        gelu_count = sum(1 for _ in clip_model.modules() if isinstance(_, _SigmoidGELU))
        print(f"  [GELU] Replaced {gelu_count} nn.GELU -> sigmoid-GELU (Erf-free, ~12.8ms NPU)")

    if args.drop_text_layers > 0:
        text_model = getattr(clip_model, 'text', clip_model)
        n_orig = len(text_model.transformer.resblocks)
        text_model.transformer.resblocks = nn.ModuleList(
            list(text_model.transformer.resblocks)[:n_orig - args.drop_text_layers]
        )
        print(f"  [drop_text_layers] Text transformer: {n_orig} → {n_orig - args.drop_text_layers} layers")

    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    # ── Get prompt tokens ────────────────────────────────────────
    use_prompt = bool(args.prompt)
    if use_prompt:
        prompt_tokens = get_prompt_tokens(args.prompt, MODEL_NAME)
        print(f"  Prompt \"{args.prompt}\" → tokens: {prompt_tokens.tolist()} ({len(prompt_tokens)} tokens)")
        print(f"  Effective max content length: {75 - len(prompt_tokens)} tokens (vs 75 baseline)")
    else:
        print(f"  No prompt (baseline text encoder)")

    # ── Create internal wrappers ─────────────────────────────────
    if args.img_pad_to > 0:
        if not args.no_cls:
            raise ValueError("--img_pad_to requires --no_cls")
        img_wrapper_internal = PaddedImageEncoderWrapper(
            clip_model,
            pad_to=args.img_pad_to,
            bake_norm=args.bake_norm,
        ).eval()
        print(f"  [IMG-PAD] NoCLS + zero-pad: {img_wrapper_internal.n_real} real tokens → padded to {args.img_pad_to}")
    elif args.npu_img_192:
        img_wrapper_internal = NPUImageEncoderWrapper(
            clip_model,
            keep_tokens=192,
            bake_norm=args.bake_norm,
        ).eval()
        print("  [NPU-IMG] 192-token cut enabled (patch -> first 192, GAP pooling)")
    else:
        img_wrapper_internal = ImageEncoderONNX(clip_model, bake_norm=args.bake_norm).eval()
    if args.bake_norm:
        print(f"  Baking CLIP normalization into image encoder (for old checkpoints)")
    use_mask_sum = args.npu_mask_sum
    use_text_gap = args.npu_text_gap
    if use_mask_sum and use_text_gap:
        print(f"  [NPU] Both --npu_mask_sum and --npu_text_gap set; Text GAP will take priority")

    # ── NPUTextEncoderWrapper: zero-slice path (any multiple of 8)
    use_npu_wrapper = ((use_mask_sum or use_text_gap) and TEXT_LEN % 8 == 0 and not use_prompt)
    if use_npu_wrapper:
        txt_wrapper_baseline = NPUTextEncoderWrapper(clip_model, use_text_gap=use_text_gap, text_len=TEXT_LEN).eval()
        if args.eval_wrapper:
            print(f"  [NPU] Zero-slice text path: external input shape (1,{TEXT_LEN})")
            print(f"        eval_wrapper is bypassed to keep ONNX forward slice-free")
        if use_text_gap:
            print(f"  [NPU] NPUTextEncoderWrapper: zero-slice fixed-{TEXT_LEN}, Text GAP pooling")
            print(f"        Fully static path: no forward slicing, no EOS gather")
        else:
            print(f"  [NPU] NPUTextEncoderWrapper: zero-slice fixed-{TEXT_LEN}, static repeat-mask pooling")
            print(f"        Avoids GatherND/dynamic BMM and keeps forward slice-free")
        EXPORT_TXT_LEN = TEXT_LEN
    else:
        if use_prompt:
            txt_wrapper_prompt = TextEncoderONNXPrompt(clip_model, prompt_tokens, text_len=TEXT_LEN,
                                                        use_mask_sum=use_mask_sum,
                                                        use_text_gap=use_text_gap).eval()
        txt_wrapper_baseline = TextEncoderONNXBaseline(clip_model, text_len=TEXT_LEN,
                                                        use_mask_sum=use_mask_sum,
                                                        use_text_gap=use_text_gap).eval()
        if use_text_gap:
            print(f"  [NPU] Text GAP pooling enabled (mean over token dimension)")
        elif use_mask_sum:
            print(f"  [NPU] Static repeat-mask pooling enabled (replaces dynamic Gather)")
        if TEXT_LEN != 77:
            print(f"  Text pos_embed: (77, {EMBED_DIM}) → ({TEXT_LEN}, {EMBED_DIM})")

    # ── Apply eval wrappers if requested (external 77/224 → internal TEXT_LEN/IMAGE_SIZE) ──
    use_eval_wrapper = args.eval_wrapper and (TEXT_LEN != 77 or IMAGE_SIZE != 224)
    if use_eval_wrapper and not use_npu_wrapper:
        img_wrapper = ImageEncoderWrapper(img_wrapper_internal, target_size=IMAGE_SIZE).eval()
        if use_prompt:
            txt_wrapper_prompt = TextEncoderWrapper(txt_wrapper_prompt,
                                                     external_len=77, target_len=TEXT_LEN).eval()
        txt_wrapper_baseline = TextEncoderWrapper(txt_wrapper_baseline,
                                                   external_len=77, target_len=TEXT_LEN).eval()
        EXPORT_IMG_SIZE = 224
        EXPORT_TXT_LEN  = 77
        print(f"  [eval_wrapper] External: image=(1,3,224,224), text=(1,77)")
        print(f"  [eval_wrapper] Internal: image=(1,3,{IMAGE_SIZE},{IMAGE_SIZE}), text=(1,{TEXT_LEN})")
    elif use_npu_wrapper:
        if args.eval_wrapper and IMAGE_SIZE != 224:
            img_wrapper = ImageEncoderWrapper(img_wrapper_internal, target_size=IMAGE_SIZE).eval()
            EXPORT_IMG_SIZE = 224
            print(f"  [eval_wrapper:image] External image=(1,3,224,224) -> internal ({IMAGE_SIZE},{IMAGE_SIZE})")
        else:
            img_wrapper = img_wrapper_internal
            EXPORT_IMG_SIZE = IMAGE_SIZE
        if args.eval_wrapper and TEXT_LEN != 77:
            txt_wrapper_baseline = TextEncoderWrapper(txt_wrapper_baseline,
                                                       external_len=77, target_len=TEXT_LEN).eval()
            EXPORT_TXT_LEN = 77
            print(f"  [eval_wrapper:text] External text=(1,77) -> internal (1,{TEXT_LEN})")
    else:
        img_wrapper = img_wrapper_internal
        EXPORT_IMG_SIZE = IMAGE_SIZE
        EXPORT_TXT_LEN  = TEXT_LEN

    # ── Verify forward pass ──────────────────────────────────────
    dummy_img = torch.rand(1, 3, EXPORT_IMG_SIZE, EXPORT_IMG_SIZE, dtype=torch.float32)
    dummy_txt = torch.zeros(1, EXPORT_TXT_LEN, dtype=torch.int64)
    txt_wrapper_export = txt_wrapper_prompt if use_prompt else txt_wrapper_baseline
    with torch.no_grad():
        img_out = img_wrapper(dummy_img)
        txt_out = txt_wrapper_export(dummy_txt)
        print(f"\n  Image output: {img_out.shape}  Text output: {txt_out.shape}")
        assert img_out.shape == (1, EMBED_DIM)
        assert txt_out.shape == (1, EMBED_DIM)
        print("  Forward pass OK [PASS]")

    # ── Local evaluation: compare baseline vs prompt ─────────────
    print(f"\n{'='*70}")
    print(f"  Local Evaluation (sample dataset)")
    print(f"{'='*70}")
    eval_text_encoder_wrapper(img_wrapper, txt_wrapper_baseline, tokenizer, EXPORT_IMG_SIZE,
                              "Baseline (no prompt)", text_len=EXPORT_TXT_LEN)
    if use_prompt:
        eval_text_encoder_wrapper(img_wrapper, txt_wrapper_prompt, tokenizer, EXPORT_IMG_SIZE,
                                  f"Prompt-baked: \"{args.prompt} {{}}\"", text_len=EXPORT_TXT_LEN)

    if args.local_eval:
        print("\n  Local eval complete. Use without --local_eval to also export/compile.")
        return

    # ── SmoothQuant (optional, before ONNX export) ───────────────
    if args.smooth_quant:
        from smooth_quant_utils import (collect_text_act_scales,
                                        apply_smooth_quant_text,
                                        load_competition_texts)
        import open_clip as _oc
        print(f"\n{'='*70}")
        print(f"  SmoothQuant Text Encoder (alpha={args.smooth_quant_alpha})")
        print(f"{'='*70}")
        _sq_texts = load_competition_texts()
        _sq_tok   = _oc.get_tokenizer(MODEL_NAME)
        _ref_out  = clip_model.encode_text(_sq_tok(["a photo of a cat"]))
        print(f"  Collecting activation scales ({len(_sq_texts)} texts)...")
        _sq_scales = collect_text_act_scales(clip_model, _sq_texts, _sq_tok, device='cpu')
        apply_smooth_quant_text(clip_model, _sq_scales, alpha=args.smooth_quant_alpha)
        _post_out = clip_model.encode_text(_sq_tok(["a photo of a cat"]))
        _diff = (_ref_out - _post_out).abs().max().item()
        print(f"  [SmoothQuant] Max output diff after smoothing: {_diff:.2e} (should be <1e-4)")
        if _diff > 1e-2:
            print(f"  [SmoothQuant] WARNING: large diff — check model structure")

    # ── Export ONNX ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Exporting ONNX")
    print(f"{'='*70}")

    os.makedirs(OUT_DIR, exist_ok=True)
    img_onnx = os.path.join(OUT_DIR, "image_encoder.onnx")
    txt_onnx = os.path.join(OUT_DIR, "text_encoder.onnx")

    print(f"\n  Exporting Image Encoder...")
    with torch.no_grad():
        torch.onnx.export(
            img_wrapper, (dummy_img,), img_onnx,
            opset_version=18,
            input_names=["image"],
            output_names=["embedding"],
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )
    img_mb = os.path.getsize(img_onnx) / 1024 / 1024
    print(f"  → {img_onnx} ({img_mb:.1f} MB)")

    txt_label = "prompt-baked" if use_prompt else "baseline"
    print(f"  Exporting Text Encoder ({txt_label})...")
    with torch.no_grad():
        torch.onnx.export(
            txt_wrapper_export, (dummy_txt,), txt_onnx,
            opset_version=18,
            input_names=["text"],
            output_names=["text_embedding"],
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )
    txt_mb = os.path.getsize(txt_onnx) / 1024 / 1024
    print(f"  → {txt_onnx} ({txt_mb:.1f} MB)")

    # ── Validate ONNX ────────────────────────────────────────────
    import onnxruntime as ort

    print(f"\n  Validating ONNX...")
    sess_img = ort.InferenceSession(img_onnx, providers=["CPUExecutionProvider"])
    sess_txt = ort.InferenceSession(txt_onnx, providers=["CPUExecutionProvider"])

    with torch.no_grad():
        pt_img = img_wrapper(dummy_img).numpy()
        pt_txt = txt_wrapper_export(dummy_txt).numpy()
    ort_img = sess_img.run(None, {"image": dummy_img.numpy()})[0]
    ort_txt = sess_txt.run(None, {"text": dummy_txt.numpy()})[0]

    img_diff = float(np.abs(pt_img - ort_img).max())
    txt_diff = float(np.abs(pt_txt - ort_txt).max())
    print(f"  Image max diff: {img_diff:.2e} {'PASS' if img_diff < 1e-4 else 'WARNING'}")
    print(f"  Text  max diff: {txt_diff:.2e} {'PASS' if txt_diff < 1e-4 else 'WARNING'}")

    # ── Evaluate ONNX model on sample dataset ────────────────────
    print(f"\n{'='*70}")
    print(f"  ONNX Evaluation (sample dataset)")
    print(f"{'='*70}")

    image_names, all_texts, gt = load_sample_dataset()
    text_tokens = tokenize_with_length(tokenizer, all_texts, text_len=EXPORT_TXT_LEN)

    # Image embeddings via ONNX (feed EXPORT_IMG_SIZE; wrapper handles upscaling if needed)
    img_embs = []
    for img_name in image_names:
        img = preprocess_image_raw(os.path.join(IMAGE_DIR, img_name), EXPORT_IMG_SIZE)
        out = sess_img.run(None, {"image": img.numpy()})[0]
        img_embs.append(out)
    img_embs = np.concatenate(img_embs, axis=0)
    img_embs = img_embs / np.linalg.norm(img_embs, axis=-1, keepdims=True)

    # Text embeddings via ONNX (one at a time since batch=1)
    txt_embs = []
    for i in range(len(all_texts)):
        tok = text_tokens[i:i+1].numpy().astype(np.int64)
        out = sess_txt.run(None, {"text": tok})[0]
        txt_embs.append(out)
    txt_embs = np.concatenate(txt_embs, axis=0)
    txt_embs = txt_embs / np.linalg.norm(txt_embs, axis=-1, keepdims=True)

    sim = img_embs @ txt_embs.T
    sim_torch = torch.from_numpy(sim)
    r1 = compute_recall_at_k(sim_torch, gt, k=1)
    r5 = compute_recall_at_k(sim_torch, gt, k=5)
    r10 = compute_recall_at_k(sim_torch, gt, k=10)
    print(f"  ONNX {txt_label}:  R@1={r1*100:5.1f}%  R@5={r5*100:5.1f}%  R@10={r10*100:5.1f}%")

    # ── Organize for AI Hub ──────────────────────────────────────
    img_dir = os.path.join(OUT_DIR, "img_dir")
    txt_dir = os.path.join(OUT_DIR, "txt_dir")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    shutil.copy2(img_onnx, os.path.join(img_dir, "image_encoder.onnx"))
    shutil.copy2(txt_onnx, os.path.join(txt_dir, "text_encoder.onnx"))

    # Copy .data files if they exist
    for src, dst_dir in [(img_onnx, img_dir), (txt_onnx, txt_dir)]:
        data_file = src + ".data"
        if os.path.exists(data_file):
            shutil.copy2(data_file, os.path.join(dst_dir, os.path.basename(data_file)))

    print(f"\n  AI Hub directories ready:")
    print(f"    {img_dir}/")
    print(f"    {txt_dir}/")

    if args.export_only:
        print("\n  Export complete.")
        return

    # ── Compile on AI Hub ────────────────────────────────────────
    import qai_hub as hub
    device = hub.Device("XR2 Gen 2 (Proxy)")
    share_email = "lowpowervision@gmail.com"

    print(f"\n{'='*70}")
    print(f"  Compiling on AI Hub")
    print(f"{'='*70}")

    _base_options = "--target_runtime qnn_dlc --truncate_64bit_io"
    if args.qairt_version:
        _base_options += f" --qairt_version {args.qairt_version}"
        print(f"  QAIRT version override: {args.qairt_version}")

    _txt_options = _base_options
    if args.txt_w8a8:
        _txt_options += " --quantize_full_type int8"
        print(f"  Text encoder: int8 (W8A8) quantization enabled")

    print(f"\n  Compiling Image Encoder...")
    img_compile = hub.submit_compile_job(
        model=img_dir + "/",
        device=device,
        input_specs={"image": (1, 3, EXPORT_IMG_SIZE, EXPORT_IMG_SIZE)},
        name=f"LPCVC_ImgEnc_{MODEL_NAME}",
        options=_base_options,
    )
    print(f"  Job ID: {img_compile.job_id}")
    img_compile.modify_sharing(add_emails=[share_email])

    print(f"\n  Compiling Text Encoder (prompt-baked)...")
    txt_compile = hub.submit_compile_job(
        model=txt_dir + "/",
        device=device,
        input_specs={"text": ((1, EXPORT_TXT_LEN), "int64")},
        name=f"LPCVC_TxtEnc_{MODEL_NAME}",
        options=_txt_options,
    )
    print(f"  Job ID: {txt_compile.job_id}")
    txt_compile.modify_sharing(add_emails=[share_email])

    print(f"\n  Waiting for compilation...")
    img_compile.wait()
    txt_compile.wait()

    img_ok = img_compile.get_target_model() is not None
    txt_ok = txt_compile.get_target_model() is not None

    print(f"  Image compile: {'OK' if img_ok else 'FAILED'}")
    print(f"  Text  compile: {'OK' if txt_ok else 'FAILED'}")

    if not (img_ok and txt_ok):
        print("\n  Compilation failed!")
        print(f"  Image: https://app.aihub.qualcomm.com/jobs/{img_compile.job_id}/")
        print(f"  Text:  https://app.aihub.qualcomm.com/jobs/{txt_compile.job_id}/")
        return

    if args.no_profile:
        print(f"\n  Compilation success. Job IDs for submission:")
        print(f"    Image: {img_compile.job_id}")
        print(f"    Text:  {txt_compile.job_id}")
        return

    # ── Profile ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Profiling on AI Hub")
    print(f"{'='*70}")

    img_target = img_compile.get_target_model()  # already verified non-None above
    txt_target = txt_compile.get_target_model()

    img_profile = hub.submit_profile_job(
        model=img_target, device=device,
        name=f"LPCVC_ImgEnc_{MODEL_NAME}_profile",
        options="--max_profiler_iterations 100",
    )
    img_profile.modify_sharing(add_emails=[share_email])

    txt_profile = hub.submit_profile_job(
        model=txt_target, device=device,
        name=f"LPCVC_TxtEnc_{MODEL_NAME}_profile",
        options="--max_profiler_iterations 100",
    )
    txt_profile.modify_sharing(add_emails=[share_email])

    print(f"  Image profile: {img_profile.job_id}")
    print(f"  Text  profile: {txt_profile.job_id}")

    print(f"\n  Waiting for profiling...")
    img_profile.wait()
    txt_profile.wait()

    img_result = img_profile.download_profile()
    txt_result = txt_profile.download_profile()
    img_lat = img_result["execution_summary"]["estimated_inference_time"]
    txt_lat = txt_result["execution_summary"]["estimated_inference_time"]
    total = img_lat + txt_lat

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Image Latency: {img_lat:,} us")
    print(f"  Text  Latency: {txt_lat:,} us")
    print(f"  Total Latency: {total:,} us")
    print(f"  Baseline:      30,712 us")
    print(f"  Previous best: 27,360 us (B-16 datacomp v3, acc=0.52)")
    print(f"")
    print(f"  Compile Job IDs for submission:")
    print(f"    Image: {img_compile.job_id}")
    print(f"    Text:  {txt_compile.job_id}")
    print(f"")
    if total < 30712:
        print(f"  [OK] WITHIN BUDGET - ready to submit!")
        print(f"    https://lpcv.ai/2026LPCVC/submission/track1")
    else:
        print(f"  [FAIL] OVER BUDGET - latency exceeds baseline")


if __name__ == "__main__":
    main()
