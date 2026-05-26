"""Programmatic inference helpers used by the web UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from features import get_audio_features, stack_features
from models import build_model
from preprocess import load_canonical
from gradcam import GradCAM


@dataclass
class ModelHandle:
    name: str
    description: str
    module: torch.nn.Module
    threshold: float = 0.5
    weights_path: Optional[str] = None


@dataclass
class InferenceResult:
    file: str
    duration_s: float
    sample_rate: int
    features_shape: tuple
    per_model: List[Dict[str, Any]] = field(default_factory=list)
    intermediates: Dict[str, np.ndarray] = field(default_factory=dict)

    def as_summary(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "features_shape": list(self.features_shape),
            "per_model": self.per_model,
        }


class InferenceEngine:
    def __init__(self, cfg: Dict[str, Any], device: str | torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device or cfg.get("device", "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.frame_target = int(cfg.get("frame_target", 400))
        self.handles: List[ModelHandle] = []
        for entry in cfg["models"]:
            self._load_model(entry)

    def _load_model(self, entry: Dict[str, Any]) -> None:
        name = entry["name"]
        weights = entry.get("weights")
        description = entry.get("description", name)
        module = build_model(name).to(self.device).eval()
        threshold = 0.5
        if weights and Path(weights).exists():
            ckpt = torch.load(weights, map_location="cpu")
            state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            module.load_state_dict(state_dict)
            threshold = float(ckpt.get("threshold", 0.5))
        self.handles.append(ModelHandle(name=name, description=description,
                                        module=module, threshold=threshold,
                                        weights_path=weights))

    def predict_path(self, audio_path: str | Path, run_gradcam: bool = True, active_filters: list[str] = None) -> InferenceResult:
        if active_filters is None: active_filters = []
        wav = load_canonical(audio_path)
        
        try:
            mel, gd, _, _ = get_audio_features(str(audio_path))
            feat_full = torch.from_numpy(stack_features(mel, gd))
            
            # --- ЗАСТОСУВАННЯ ФІЛЬТРІВ ПЕРЕД НАРІЗКОЮ ---
            if active_filters:
                feat_full = feat_full.unsqueeze(0) # (1, 2, 128, T)
                
                if "bandpass" in active_filters or "combined_survival" in active_filters:
                    mask = torch.ones_like(feat_full)
                    mask[:, :, :2, :] = 0
                    mask[:, :, 120:, :] = 0
                    feat_full = feat_full * mask
                    
                if "noise_gate" in active_filters or "combined_survival" in active_filters:
                    noise_thresh = feat_full.mean() - 0.5 * feat_full.std()
                    gate_mask = (feat_full > noise_thresh).float()
                    feat_full = feat_full * gate_mask
                    
                if "delta_boost" in active_filters:
                    delta = torch.zeros_like(feat_full)
                    delta[:, :, :, 1:-1] = feat_full[:, :, :, 2:] - feat_full[:, :, :, :-2]
                    feat_full = feat_full + (delta * 0.5)
                    
                feat_full = feat_full.squeeze(0) # Повертаємо до (2, 128, T)

        except Exception:
            feat_full = torch.zeros((2, 128, self.frame_target))

        T_full = feat_full.shape[-1]
        w_size = self.frame_target
        stride = w_size // 2

        chunks = []
        if T_full <= w_size:
            pad_len = w_size - T_full
            feat_padded = F.pad(feat_full, (0, pad_len))
            chunks.append((feat_padded, 0, T_full))
            T_alloc = w_size
        else:
            for start in range(0, T_full - w_size + 1, stride):
                chunks.append((feat_full[:, :, start:start+w_size], start, start+w_size))
            if (T_full - w_size) % stride != 0:
                start = T_full - w_size
                chunks.append((feat_full[:, :, start:T_full], start, T_full))
            T_alloc = T_full

        per_model_scores = {h.name: [] for h in self.handles}
        gradcam_accum = {h.name: torch.zeros((128, T_alloc), device=self.device) for h in self.handles}
        gradcam_weights = {h.name: torch.zeros((128, T_alloc), device=self.device) for h in self.handles}

        for chunk_feat, start_f, end_f in chunks:
            tensor = chunk_feat.unsqueeze(0).float().to(self.device)
            for h in self.handles:
                with torch.no_grad():
                    logits = h.module(tensor)
                    probs = F.softmax(logits, dim=-1)
                    spoof_p = float(probs[0, 1])
                per_model_scores[h.name].append(spoof_p)

                if run_gradcam and h.weights_path and Path(h.weights_path).exists():
                    try:
                        cam = GradCAM(h.module).explain(tensor, target_class=1)
                        if isinstance(cam, np.ndarray):
                            cam = torch.from_numpy(cam).to(self.device)
                        cam = cam.squeeze(0)
                        gradcam_accum[h.name][:, start_f:start_f+w_size] += cam
                        gradcam_weights[h.name][:, start_f:start_f+w_size] += 1.0
                    except Exception:
                        pass

        per_model_final = []
        final_cams = {}
        for h in self.handles:
            max_spoof = float(np.max(per_model_scores[h.name]))
            per_model_final.append({
                "model": h.name,
                "description": h.description,
                "spoof_probability": max_spoof,
                "bonafide_probability": 1.0 - max_spoof,
                "decision": "spoof" if max_spoof >= h.threshold else "bona-fide",
                "threshold": h.threshold,
                "weights_present": bool(h.weights_path and Path(h.weights_path).exists()),
            })

            if run_gradcam and h.name in gradcam_accum:
                w = gradcam_weights[h.name]
                w[w == 0] = 1.0
                avg_cam = gradcam_accum[h.name] / w
                
                c_min, c_max = avg_cam.min(), avg_cam.max()
                if c_max > c_min:
                    avg_cam = (avg_cam - c_min) / (c_max - c_min)
                
                if T_full < w_size:
                    avg_cam = avg_cam[:, :T_full]
                    
                final_cams[h.name] = avg_cam.cpu().numpy()

        mel_out = feat_full[0].numpy()
        gd_out = feat_full[1].numpy()

        return InferenceResult(
            file=str(audio_path),
            duration_s=len(wav) / 16000.0,
            sample_rate=16000,
            features_shape=tuple(feat_full.shape),
            per_model=per_model_final,
            intermediates={
                "waveform": wav,
                "mel_db": mel_out,
                "group_delay": gd_out,
                "mel_db_cropped": mel_out,
                "group_delay_cropped": gd_out,
                **{f"gradcam_{k}": v for k, v in final_cams.items()},
            },
        )