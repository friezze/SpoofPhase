"""Grad-CAM for the detector backbones.

Reference
---------
Selvaraju R. R., Cogswell M., Das A., Vedantam R., Parikh D., Batra D.
*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization*. ICCV 2017, pp. 618–626.

This module implements the canonical formulation (formulas (2.8) and (2.9)
of the course report) with two practical extensions:

* The target convolutional block is exposed by every detector via a
  ``gradcam_target`` property — this avoids monkey-patching and keeps the
  XAI module decoupled from model internals.
* The returned heatmap is **already upsampled** to the feature-tensor
  resolution ``(n_mels, T)`` so visualisation code does not have to know
  about the model's stride pattern.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Light-weight Grad-CAM wrapper. One instance per model.

    Usage::

        cam = GradCAM(model)
        heat = cam.explain(input_tensor, target_class=1)   # numpy (n_mels, T)
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None) -> None:
        self.model = model
        if target_layer is None:
            if not hasattr(model, "gradcam_target"):
                raise AttributeError(
                    "Model does not expose .gradcam_target; pass target_layer explicitly."
                )
            target_layer = model.gradcam_target
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_hook = target_layer.register_forward_hook(self._on_forward)
        self._bwd_hook = target_layer.register_full_backward_hook(self._on_backward)

    # --- hooks ------------------------------------------------------------
    def _on_forward(self, _module, _inp, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _on_backward(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    # --- public -----------------------------------------------------------
    def explain(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """Return a (n_mels, T) heatmap for ``target_class`` (default = spoof)."""
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError("GradCAM.explain expects a (1, C, F, T) batch.")
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        x = x.detach().requires_grad_(True)
        logits = self.model(x)
        score = logits[0, target_class]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not fire — wrong target layer?")

        # alpha_k^c = (1/Z) sum_ij grad_{ij}^k
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)        # (1, K, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)    # (1, 1, h, w)
        cam = F.relu(cam)
        # Upsample to input feature-tensor resolution (H, W) of x.
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min) < 1e-12:
            return np.zeros_like(cam.cpu().numpy())
        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.cpu().numpy().astype(np.float32)

    def close(self) -> None:
        """Detach hooks. Call this when the model is no longer needed."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass


def gradcam_overlay(heatmap: np.ndarray, base: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a normalised heatmap ``[0,1]`` over a base feature image ``[0,1]``.

    Returned shape is ``(H, W, 3)`` RGB float in ``[0,1]``.
    """
    if heatmap.shape != base.shape:
        raise ValueError(f"shape mismatch: heatmap {heatmap.shape} vs base {base.shape}")
    # Simple jet-like colormap without importing matplotlib here.
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - 4 * np.abs(h - 0.75), 0, 1)
    g = np.clip(1.5 - 4 * np.abs(h - 0.5),  0, 1)
    b = np.clip(1.5 - 4 * np.abs(h - 0.25), 0, 1)
    color = np.stack([r, g, b], axis=-1)
    base_rgb = np.stack([base, base, base], axis=-1)
    return alpha * color + (1.0 - alpha) * base_rgb
