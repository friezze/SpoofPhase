"""Web UI для детекції deepfake аудіо та Programmatic inference helpers."""
from __future__ import annotations

import base64
import io
import sys
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SpoofPhase")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_infer_config
from features import get_audio_features, stack_features
from models import build_model
from preprocess import load_canonical
from gradcam import GradCAM
from plots import plot_features_panel

st.set_page_config(page_title="SpoofPhase — Deepfake Audio Detector", layout="centered")

def filter_bandpass(features):
    mask = torch.ones_like(features)
    mask[:, :, :2, :] = 0
    mask[:, :, 120:, :] = 0
    return features * mask

def filter_mean_subtraction(features):
    mean_val = features.mean(dim=-1, keepdim=True)
    return features - mean_val

def filter_delta_boost(features):
    delta = torch.zeros_like(features)
    delta[:, :, :, 1:-1] = features[:, :, :, 2:] - features[:, :, :, :-2]
    return features + (delta * 0.5)

def filter_bp_delta(features):
    return filter_delta_boost(filter_bandpass(features))

def filter_spec_augment(features):
    b, c, f, t = features.shape
    f_idx = torch.randint(0, max(1, f - 15), (1,))
    features_clone = features.clone()
    features_clone[:, :, f_idx:f_idx+15, :] = 0
    return features_clone

def adaptive_fix_group_delay(gd: np.ndarray) -> np.ndarray:
    """
    Нормалізація карти групової затримки (n_mels, T) → [0, 1].

    Стара реалізація (глобальний MAD) давала монохромну картинку:
    при std~1000 діапазон 5*MAD виходив ~[-750, 750], і нормальні
    значення в [-100, 100] всі стискались у вузьку смугу [0.43, 0.57].

    Виправлення: per-row percentile normalization — кожен mel-рядок
    незалежно розтягується від свого p2 до p98, тому жоден рядок
    не може «заразити» інші своїми викидами.
    """
    out = np.empty_like(gd, dtype=np.float32)
    for i in range(gd.shape[0]):
        row = gd[i]
        p2, p98 = np.percentile(row, [2, 98])
        span = p98 - p2
        if span > 1e-6:
            out[i] = np.clip((row - p2) / span, 0.0, 1.0)
        else:
            out[i] = 0.5   # рядок без інформації → нейтральне значення
    return out

ui_filters = {
    "bandpass": filter_bandpass,
    "mean_sub": filter_mean_subtraction,
    "delta_boost": filter_delta_boost,
    "comb_bp_delta": filter_bp_delta,
    "spec_aug": filter_spec_augment
}

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

    def predict_path(self, audio_path: str | Path, run_gradcam: bool = True, active_filter: str = "") -> InferenceResult:
        wav = load_canonical(audio_path)

        try:
            mel, gd, _, _ = get_audio_features(str(audio_path))

            gd = adaptive_fix_group_delay(gd)

            feat_full = torch.from_numpy(stack_features(mel, gd))

            if active_filter and active_filter in ui_filters:
                feat_full = feat_full.unsqueeze(0)
                feat_full = ui_filters[active_filter](feat_full)
                feat_full = feat_full.squeeze(0)

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

INFER_CONFIG = Path("configs/infer.yaml")

@st.cache_resource
def _load_engine():
    logger.info(f"Завантаження конфігу: {INFER_CONFIG}")
    cfg = load_infer_config(INFER_CONFIG)
    engine_obj = InferenceEngine(cfg)

    for handle in engine_obj.handles:
        try:
            param = next(handle.module.parameters())
            mean_val = param.abs().mean().item()
            if mean_val < 1e-4:
                logger.warning(f"⚠️ УВАГА: Модель {handle.name} має нульові/випадкові ваги (mean: {mean_val:.6f})")
            else:
                logger.info(f"✅ Модель {handle.name} завантажена (mean weight: {mean_val:.6f})")
        except Exception as e:
            logger.error(f"❌ Модель {handle.name} не має доступних параметрів: {e}")
    return engine_obj

st.markdown("""
<div style="background: #1a1d27; padding: 20px; border-bottom: 1px solid #2d3148; border-radius: 12px; margin-bottom: 24px;">
    <h1 style="font-size: 1.5rem; font-weight: 700; color: #fff; margin: 0;">🎙 SpoofPhase</h1>
    <span style="font-size: 0.85rem; color: #888;">Deepfake Audio Detector — Phase-Aware Dual-Stream + Ensemble</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Перетягни WAV, MP3 або FLAC файл сюди (до 50MB)", type=["wav", "mp3", "flac"])
mic_file = st.audio_input("Або запиши з мікрофона")

active_file = uploaded_file or mic_file

if active_file:
    file_name = active_file.name if hasattr(active_file, 'name') else "мікрофон_запис.wav"
    st.markdown(f"""<div style="text-align: center; margin-top: 16px; font-size: 1.1rem; color: #2ecc71; font-weight: bold;">
    Файл завантажено: <span style="color:#fff;">{file_name}</span></div>""", unsafe_allow_html=True)

st.markdown("### 🎛 Акустичні фільтри (Режим оцінки)")
filter_options = {
    "Без фільтру (Оригінал)": "",
    "Mean Subtraction (Видаляє статичний шум кімнати)": "mean_sub",
    "Delta Boost (Підсилює різкі переходи фази/інтонації)": "delta_boost",
    "Bandpass + Delta (Обрізає гул і підсилює динаміку)": "comb_bp_delta",
    "Bandpass (Обрізає низький бас і високочастотний писк)": "bandpass",
    "Spec Augment (Симулює втрату пакетів/даних)": "spec_aug"
}
selected_filter_label = st.radio("Фільтри", list(filter_options.keys()), label_visibility="collapsed")
active_filter = filter_options[selected_filter_label]

analyze_btn = st.button("▶ Аналізувати", type="primary", disabled=(active_file is None))

if analyze_btn and active_file:
    try:
        eng = _load_engine()
    except Exception as e:
        st.error(f"Не вдалося завантажити модель: {e}")
        st.stop()

    suffix = Path(file_name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(active_file.getvalue())
        tmp_path = Path(tmp.name)

    with st.spinner("Аналіз аудіо..."):
        try:
            logger.info(f"Аналіз файлу: {tmp_path}, Фільтр: {active_filter or 'None'}")
            result = eng.predict_path(tmp_path, run_gradcam=True, active_filter=active_filter)
            for m in result.per_model:
                logger.info(f"Модель {m['model']} -> Prob: {m['spoof_probability']:.4f}, Decision: {m['decision']}")
        except Exception as e:
            st.error(f"Помилка аналізу: {e}")
            tmp_path.unlink(missing_ok=True)
            st.stop()
        finally:
            tmp_path.unlink(missing_ok=True)

    spoof_count = sum(1 for m in result.per_model if m['decision'] == 'spoof')
    is_spoof = spoof_count > len(result.per_model) / 2
    avg_prob = sum(m['spoof_probability'] for m in result.per_model) / len(result.per_model)

    verdict_style = "background: #2d1a1a; border: 1px solid #c0392b;" if is_spoof else "background: #1a2d1a; border: 1px solid #27ae60;"
    verdict_icon = "🚨" if is_spoof else "✅"
    verdict_label = "DEEPFAKE" if is_spoof else "СПРАВЖНІЙ ГОЛОС"
    label_color = "#e74c3c" if is_spoof else "#2ecc71"

    st.markdown(f"""
    <div style="{verdict_style} padding: 24px; border-radius: 12px; margin-bottom: 24px; display: flex; align-items: center; gap: 20px;">
        <div style="font-size: 3rem;">{verdict_icon}</div>
        <div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {label_color};">{verdict_label}</div>
            <div style="color: #aaa; font-size: 0.9rem; margin-top: 4px;">
                Середня ймовірність спуфу: {avg_prob * 100:.1f}%  &bull;  Тривалість: {result.duration_s:.2f}s
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.audio(active_file)

    cols = st.columns(3)
    for idx, m in enumerate(result.per_model):
        is_s = m['decision'] == 'spoof'
        pct = m['spoof_probability'] * 100
        bar_color = "#e74c3c" if is_s else "#2ecc71"
        card_html = f"""
        <div style="background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
            <h4 style="font-size: 0.85rem; color: #888; margin: 0 0 8px 0;">{m['model']}</h4>
            <div style="font-size: 1.4rem; font-weight: 700; color: {bar_color};">{pct:.1f}%</div>
            <div style="font-size: 0.75rem; color: #888; margin-top: 2px;">{m['decision']} (thr {m['threshold']:.2f})</div>
            <div style="background: #2d3148; border-radius: 4px; height: 6px; margin-top: 8px; overflow: hidden;">
                <div style="height: 6px; background: {bar_color}; width: {pct}%;"></div>
            </div>
        </div>
        """
        cols[idx % 3].markdown(card_html, unsafe_allow_html=True)

    interm = result.intermediates
    wav = interm["waveform"]
    mel = interm["mel_db"]
    gd  = interm["group_delay"]

    cams = {k.replace("gradcam_", ""): v for k, v in interm.items() if k.startswith("gradcam_")}
    if not cams:
        cams = None

    fig = plot_features_panel(wav, mel, gd, cams=cams, title=f"{Path(result.file).name}")

    ablation = None
    for m_result in result.per_model:
        if m_result.get("weights_present"):
            handle = next((h for h in eng.handles if h.name == m_result["model"]), None)
            if handle:
                feat = interm["mel_db_cropped"]
                if feat.shape[1] > 400:
                    feat = feat[:, :400]
                gd_c = interm["group_delay_cropped"]
                if gd_c.shape[1] > 400:
                    gd_c = gd_c[:, :400]
                if feat.shape[1] < 400:
                    pad = 400 - feat.shape[1]
                    feat = np.pad(feat, ((0, 0), (0, pad)), mode="reflect")
                    gd_c = np.pad(gd_c, ((0, 0), (0, pad)), mode="reflect")
                feat_t = torch.from_numpy(
                    np.stack([feat, gd_c], axis=0)
                ).unsqueeze(0).float().to(eng.device)
                try:
                    from attribution import channel_ablation
                    ablation = channel_ablation(handle.module, feat_t)
                    ablation["model_name"] = handle.name
                except Exception:
                    pass
            break

    if ablation:
        st.markdown(f"### Channel Ablation ({ablation['model_name']}) — що важливіше: Mel чи Group Delay?")
        items = [
            {"label": "Full", "value": ablation["full"], "color": "#6c7bff"},
            {"label": "Без Mel", "value": ablation["ablate_mel"], "color": "#e74c3c"},
            {"label": "Без GD", "value": ablation["ablate_gd"], "color": "#f39c12"},
            {"label": "Mel важл.", "value": max(0, ablation["mel_importance"]), "color": "#2ecc71"},
            {"label": "GD важл.", "value": max(0, ablation["gd_importance"]), "color": "#1abc9c"},
        ]

        abl_cols = st.columns(len(items))
        for idx, item in enumerate(items):
            h = round(item["value"] * 80)
            col_html = f"""
            <div style="background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px; padding: 12px; display: flex; flex-direction: column; align-items: center; margin-bottom: 16px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 8px;">{item['value'] * 100:.1f}%</div>
                <div style="width: 100%; max-width: 60px; background: #2d3148; border-radius: 6px; height: 80px; display: flex; align-items: flex-end; overflow: hidden; margin-bottom: 8px;">
                    <div style="width: 100%; border-radius: 6px; height: {h}px; background: {item['color']};"></div>
                </div>
                <div style="font-size: 0.75rem; color: #888; text-align: center;">{item['label']}</div>
            </div>
            """
            abl_cols[idx].markdown(col_html, unsafe_allow_html=True)

    st.markdown("### Детальний аналіз (Фічі та Grad-CAM)")
    st.pyplot(fig)