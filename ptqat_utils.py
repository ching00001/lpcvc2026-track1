"""ptqat_utils.py

Post-Training Quantization-Aware Training for MobileCLIP-B text encoder.

Injects INT8 fake quantization (STE) during fine-tuning so the model's
activation and weight distributions become INT8-friendly before QNN export.

Coverage:
  - Each ResidualAttentionBlock: pre-hook fakes-quantizes the block's input
    activation (the hidden state flowing through the residual stream).
  - attn.out_proj / mlp.c_fc / mlp.c_proj: replaced with FakeQuantLinear
    so weights are also fake-quantized per-channel during training.

Usage in training:
    ptqat_hooks = apply_ptqat_text(model)
    # ... training loop ...
    # Before torch.save():
    remove_ptqat_text(model, ptqat_hooks)
    torch.save({'state_dict': model.state_dict(), ...}, path)
    ptqat_hooks = apply_ptqat_text(model, verbose=False)

Usage in export: load checkpoint normally (no PTQAT in saved file).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── STE fake quantization ─────────────────────────────────────────────────────

def fake_quant_ste(x: torch.Tensor, n_bits: int = 8,
                   per_channel: bool = False) -> torch.Tensor:
    """Symmetric INT-n fake quantization with straight-through estimator.

    per_channel=True: scale computed per output-channel (dim 0). Use for weights.
    per_channel=False: single scale over entire tensor. Use for activations.
    """
    qmax = 2 ** (n_bits - 1) - 1  # 127 for INT8

    if per_channel:
        shape = x.shape
        x_flat = x.view(shape[0], -1)
        scale = x_flat.abs().max(dim=1)[0].clamp(min=1e-8) / qmax
        scale = scale.view(shape[0], *([1] * (len(shape) - 1)))
    else:
        scale = x.abs().max().clamp(min=1e-8) / qmax

    x_q = (x / scale).clamp(-qmax - 1, qmax).round() * scale
    return x + (x_q - x).detach()  # STE: gradient = 1


# ── FakeQuantLinear ───────────────────────────────────────────────────────────

class FakeQuantLinear(nn.Module):
    """nn.Linear with per-channel weight fake-quant only (NO activation quant).

    Weight-only fake-quant is sufficient: the model learns weights that tolerate
    INT8 quantization noise without destroying the activation signal.
    Activation fake-quant stacks across 9 layers and causes catastrophic degradation.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = fake_quant_ste(self.linear.weight, per_channel=True)
        else:
            w = self.linear.weight
        return F.linear(x, w, self.linear.bias)

    # Proxy properties so downstream code (SmoothQuant, etc.) still works
    @property
    def weight(self):        return self.linear.weight
    @property
    def bias(self):          return self.linear.bias
    @property
    def in_features(self):   return self.linear.in_features
    @property
    def out_features(self):  return self.linear.out_features


# ── Apply / remove ────────────────────────────────────────────────────────────

def apply_ptqat_text(clip_model: nn.Module,
                     verbose: bool = True) -> list:
    """Patch text transformer for PTQAT (weight-only fake-quant).

    Only the LINEAR WEIGHTS are fake-quantized (per-channel INT8 STE).
    Activations are left untouched — stacking activation fake-quant across
    9+ layers destroys the signal and causes catastrophic accuracy collapse.

    Returns empty list (no hooks needed for weight-only approach).
    """
    text = clip_model.text
    n_wrapped = 0

    for rb in text.transformer.resblocks:
        if isinstance(rb.attn.out_proj, nn.Linear):
            rb.attn.out_proj = FakeQuantLinear(rb.attn.out_proj)
            n_wrapped += 1
        if hasattr(rb.mlp, 'c_fc') and isinstance(rb.mlp.c_fc, nn.Linear):
            rb.mlp.c_fc = FakeQuantLinear(rb.mlp.c_fc)
            n_wrapped += 1
        if hasattr(rb.mlp, 'c_proj') and isinstance(rb.mlp.c_proj, nn.Linear):
            rb.mlp.c_proj = FakeQuantLinear(rb.mlp.c_proj)
            n_wrapped += 1

    if verbose:
        print(f"  [PTQAT] {n_wrapped} FakeQuantLinear (weight-only INT8 fake-quant, no activation hooks)")

    return []   # no hooks


def remove_ptqat_text(clip_model: nn.Module,
                      hooks: list,
                      verbose: bool = True):
    """Unwrap FakeQuantLinear and remove hooks (call before torch.save)."""
    text = clip_model.text
    n_unwrapped = 0

    for rb in text.transformer.resblocks:
        if isinstance(rb.attn.out_proj, FakeQuantLinear):
            rb.attn.out_proj = rb.attn.out_proj.linear
            n_unwrapped += 1
        if hasattr(rb.mlp, 'c_fc') and isinstance(rb.mlp.c_fc, FakeQuantLinear):
            rb.mlp.c_fc = rb.mlp.c_fc.linear
            n_unwrapped += 1
        if hasattr(rb.mlp, 'c_proj') and isinstance(rb.mlp.c_proj, FakeQuantLinear):
            rb.mlp.c_proj = rb.mlp.c_proj.linear
            n_unwrapped += 1

    for h in hooks:
        h.remove()

    if verbose:
        print(f"  [PTQAT] Removed {n_unwrapped} wrappers + {len(hooks)} hooks (checkpoint clean)")
