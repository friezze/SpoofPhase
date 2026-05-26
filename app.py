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
from flask import Flask, jsonify, render_template_string, request
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SpoofPhase")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_infer_config
from features import get_audio_features, stack_features
from models import build_model
from preprocess import load_canonical
from gradcam import GradCAM
from plots import plot_features_panel
import librosa
def compute_local_gd(y, sr=16000):
    n_fft = 1024
    hop_length = 256
    n_mels = 128
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    phases = np.angle(D)
    unwrapped_phase = np.unwrap(phases, axis=0)
    gd_matrix = -np.diff(unwrapped_phase, axis=0)
    gd_matrix = np.vstack((gd_matrix, np.zeros(gd_matrix.shape[1])))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    gd_mel = np.dot(mel_basis, gd_matrix)
    p_low = np.percentile(gd_mel, 5)
    p_high = np.percentile(gd_mel, 95)
    gd_clipped = np.clip(gd_mel, p_low, p_high)
    return gd_clipped
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
        self.handles.append(ModelHandle(name=name, description=description, module=module, threshold=threshold, weights_path=weights))
    def predict_path(self, audio_path: str | Path, run_gradcam: bool = True, active_filter: str = "") -> InferenceResult:
        wav = load_canonical(audio_path)
        try:
            mel, _, _, _ = get_audio_features(str(audio_path))
            gd = compute_local_gd(wav)
            lower_bound = np.percentile(gd, 1)
            upper_bound = np.percentile(gd, 99)
            gd_model = np.clip(gd, lower_bound, upper_bound)
            feat_full = torch.from_numpy(stack_features(mel, gd_model))
            if active_filter and active_filter in ui_filters:
                feat_full = feat_full.unsqueeze(0)
                feat_full = ui_filters[active_filter](feat_full)
                feat_full = feat_full.squeeze(0)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            feat_full = torch.zeros((2, 128, self.frame_target))
            gd = np.zeros((128, self.frame_target))
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
        return InferenceResult(
            file=str(audio_path),
            duration_s=len(wav) / 16000.0,
            sample_rate=16000,
            features_shape=tuple(feat_full.shape),
            per_model=per_model_final,
            intermediates={
                "waveform": wav,
                "mel_db": mel_out,
                "group_delay": gd,
                "mel_db_cropped": mel_out,
                "group_delay_cropped": gd,
                **{f"gradcam_{k}": v for k, v in final_cams.items()},
            },
        )
INFER_CONFIG = Path("configs/infer.yaml")
app = Flask(__name__)
engine: InferenceEngine | None = None
def _load_engine():
    global engine
    if engine is None:
        logger.info(f"Завантаження конфігу: {INFER_CONFIG}")
        cfg = load_infer_config(INFER_CONFIG)
        engine = InferenceEngine(cfg)
        for handle in engine.handles:
            try:
                param = next(handle.module.parameters())
                mean_val = param.abs().mean().item()
                if mean_val < 1e-4:
                    logger.warning(f"⚠️ УВАГА: Модель {handle.name} має нульові/випадкові ваги (mean: {mean_val:.6f})")
                else:
                    logger.info(f"✅ Модель {handle.name} завантажена (mean weight: {mean_val:.6f})")
            except Exception as e:
                logger.error(f"❌ Модель {handle.name} не має доступних параметрів: {e}")
    return engine
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
HTML = """
<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpoofPhase — Deepfake Audio Detector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; min-height: 100vh; }
  header { background: #1a1d27; padding: 20px 40px; border-bottom: 1px solid #2d3148; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.5rem; font-weight: 700; color: #fff; }
  header span { font-size: 0.85rem; color: #888; }
  .container { max-width: 960px; margin: 40px auto; padding: 0 20px; }
  .upload-zone { border: 2px dashed #3d4268; border-radius: 16px; padding: 48px; text-align: center; cursor: pointer; transition: all 0.2s; background: #1a1d27; }
  .upload-zone:hover, .upload-zone.drag { border-color: #6c7bff; background: #1e2236; }
  .upload-zone input { display: none; }
  .upload-zone .icon { font-size: 3rem; margin-bottom: 12px; }
  .upload-zone p { color: #888; margin-top: 8px; font-size: 0.9rem; }
  .file-status { text-align: center; margin-top: 16px; font-size: 1.1rem; color: #2ecc71; font-weight: bold; display: none; }
  .filters-section { background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px; padding: 20px; margin-top: 24px; }
  .filters-section h3 { font-size: 1rem; color: #aaa; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.05em; }
  .filter-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .filter-label { display: flex; align-items: center; cursor: pointer; color: #ccc; font-size: 0.95rem; background: #222634; padding: 10px; border-radius: 8px; border: 1px solid #2d3148; transition: border-color 0.2s;}
  .filter-label:hover { border-color: #6c7bff; }
  .filter-label input { margin-right: 12px; transform: scale(1.3); accent-color: #6c7bff; cursor: pointer; }
  .actions { display: flex; gap: 16px; justify-content: center; margin-top: 24px; }
  .btn { display: inline-block; padding: 12px 32px; background: #6c7bff; color: #fff; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; transition: background 0.2s; font-weight: 600; }
  .btn:hover { background: #5a6aee; }
  .btn:disabled { background: #3d4268; color: #888; cursor: not-allowed; }
  .record-btn { background: #1a1d27; border: 1px solid #e74c3c; border-radius: 8px; color: #e74c3c; }
  .record-btn:hover { background: #e74c3c; color: #fff; }
  .record-btn.recording { background: #e74c3c; color: #fff; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
  .result { margin-top: 32px; display: none; }
  .verdict { padding: 24px; border-radius: 12px; margin-bottom: 24px; display: flex; align-items: center; gap: 20px; }
  .verdict.spoof { background: #2d1a1a; border: 1px solid #c0392b; }
  .verdict.bona { background: #1a2d1a; border: 1px solid #27ae60; }
  .verdict-icon { font-size: 3rem; }
  .verdict-label { font-size: 1.8rem; font-weight: 700; }
  .verdict.spoof .verdict-label { color: #e74c3c; }
  .verdict.bona .verdict-label { color: #2ecc71; }
  .verdict-sub { color: #aaa; font-size: 0.9rem; margin-top: 4px; }
  .models-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .model-card { background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px; padding: 16px; }
  .model-card h4 { font-size: 0.85rem; color: #888; margin-bottom: 8px; }
  .model-card .prob { font-size: 1.4rem; font-weight: 700; }
  .model-card .prob.spoof { color: #e74c3c; }
  .model-card .prob.bona { color: #2ecc71; }
  .model-card .bar-wrap { background: #2d3148; border-radius: 4px; height: 6px; margin-top: 8px; }
  .model-card .bar { height: 6px; border-radius: 4px; transition: width 0.5s; }
  .model-card .bar.spoof { background: #e74c3c; }
  .model-card .bar.bona { background: #2ecc71; }
  .section { background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .section h3 { font-size: 1rem; color: #aaa; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.05em; }
  .section img { width: 100%; border-radius: 8px; }
  .ablation-bars { display: flex; gap: 16px; align-items: flex-end; height: 120px; }
  .abl-bar-wrap { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; }
  .abl-bar-bg { width: 100%; background: #2d3148; border-radius: 6px; height: 80px; display: flex; align-items: flex-end; overflow: hidden; }
  .abl-bar-fill { width: 100%; border-radius: 6px; transition: height 0.5s; }
  .abl-label { font-size: 0.75rem; color: #888; }
  .abl-value { font-size: 0.9rem; font-weight: 600; }
  .spinner { display: none; text-align: center; padding: 40px; }
  .spinner.active { display: block; }
  .spin { width: 48px; height: 48px; border: 4px solid #2d3148; border-top-color: #6c7bff; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 16px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { background: #2d1a1a; border: 1px solid #c0392b; border-radius: 8px; padding: 16px; color: #e74c3c; margin-top: 16px; display: none; }
  audio { width: 100%; margin-bottom: 16px; filter: invert(0.9); }
</style>
</head>
<body>
<header>
  <div>
    <h1>🎙 SpoofPhase</h1>
    <span>Deepfake Audio Detector — Phase-Aware Dual-Stream + Ensemble</span>
  </div>
</header>
<div class="container">
  <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <div class="icon">🎵</div>
    <p style="font-size:1.1rem;color:#ccc;">Перетягни WAV файл або клікни сюди</p>
    <p>Підтримуються: WAV, MP3, FLAC (до 50MB)</p>
    <input type="file" id="fileInput" accept=".wav,.mp3,.flac">
  </div>
  <div class="file-status" id="fileStatus">
    Файл завантажено: <span id="fileName" style="color:#fff;"></span>
  </div>
  <div class="filters-section" id="filtersSection">
    <h3>🎛 Акустичні фільтри (Режим оцінки)</h3>
    <div class="filter-grid">
      <label class="filter-label"><input type="radio" name="filter_mode" value="" checked> <b>Без фільтру</b> (Оригінал)</label>
      <label class="filter-label"><input type="radio" name="filter_mode" value="mean_sub"> <b>Mean Subtraction</b> (Видаляє статичний шум кімнати)</label>
      <label class="filter-label"><input type="radio" name="filter_mode" value="delta_boost"> <b>Delta Boost</b> (Підсилює різкі переходи фази/інтонації)</label>
      <label class="filter-label"><input type="radio" name="filter_mode" value="comb_bp_delta"> <b>Bandpass + Delta</b> (Обрізає гул і підсилює динаміку)</label>
      <label class="filter-label"><input type="radio" name="filter_mode" value="bandpass"> <b>Bandpass</b> (Обрізає низький бас і високочастотний писк)</label>
      <label class="filter-label"><input type="radio" name="filter_mode" value="spec_aug"> <b>Spec Augment</b> (Симулює втрату пакетів/даних)</label>
    </div>
  </div>
  <div class="actions">
    <button class="btn" id="analyzeBtn" onclick="runAnalysis()" disabled>▶ Аналізувати</button>
    <button class="btn record-btn" id="recordBtn" onclick="toggleRecord()">🎤 Записати з мікрофона</button>
  </div>
  <div class="spinner" id="spinner">
    <div class="spin"></div>
    <p>Аналіз аудіо...</p>
  </div>
  <div class="error" id="error"></div>
  <div class="result" id="result">
    <div class="verdict" id="verdict">
      <div class="verdict-icon" id="verdictIcon"></div>
      <div>
        <div class="verdict-label" id="verdictLabel"></div>
        <div class="verdict-sub" id="verdictSub"></div>
      </div>
    </div>
    <audio id="audioPlayer" controls></audio>
    <div class="models-grid" id="modelsGrid"></div>
    <div class="section" id="ablationSection">
      <h3 id="ablationTitle">Channel Ablation — що важливіше: Mel чи Group Delay?</h3>
      <div class="ablation-bars" id="ablationBars"></div>
    </div>
    <div class="section">
      <h3>Детальний аналіз (Фічі та Grad-CAM)</h3>
      <img id="featuresImg" src="" alt="Features">
    </div>
  </div>
</div>
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const recordBtn = document.getElementById('recordBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
let mediaRecorder;
let audioChunks = [];
let currentFile = null;
fileInput.addEventListener('click', e => e.stopPropagation());
function setReadyFile(file) {
  currentFile = file;
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileStatus').style.display = 'block';
  analyzeBtn.disabled = false;
  document.getElementById('result').style.display = 'none';
  document.getElementById('error').style.display = 'none';
  document.getElementById('featuresImg').src = '';
}
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag');
  if (e.dataTransfer.files[0]) setReadyFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { 
  if (fileInput.files[0]) setReadyFile(fileInput.files[0]); 
});
async function toggleRecord() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    recordBtn.classList.remove('recording');
    recordBtn.textContent = '🎤 Записати з мікрофона';
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      const file = new File([blob], "мікрофон_запис.webm", { type: 'audio/webm' });
      setReadyFile(file);
    };
    audioChunks = [];
    mediaRecorder.start();
    recordBtn.classList.add('recording');
    recordBtn.textContent = '⏹ Зупинити запис';
  } catch (err) {
    alert('Помилка мікрофона: ' + err.message);
  }
}
function runAnalysis() {
  if (!currentFile) return;
  const formData = new FormData();
  formData.append('audio', currentFile);
  const selectedFilter = document.querySelector('input[name="filter_mode"]:checked').value;
  formData.append('filter', selectedFilter);
  document.getElementById('spinner').classList.add('active');
  document.getElementById('result').style.display = 'none';
  document.getElementById('error').style.display = 'none';
  document.getElementById('modelsGrid').innerHTML = '';
  document.getElementById('ablationBars').innerHTML = '';
  const url = URL.createObjectURL(currentFile);
  document.getElementById('audioPlayer').src = url;
  fetch('/predict', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      document.getElementById('spinner').classList.remove('active');
      if (data.error) {
        document.getElementById('error').textContent = data.error;
        document.getElementById('error').style.display = 'block';
        return;
      }
      renderResult(data);
    })
    .catch(err => {
      document.getElementById('spinner').classList.remove('active');
      document.getElementById('error').textContent = 'Помилка: ' + err;
      document.getElementById('error').style.display = 'block';
    });
}
function renderResult(data) {
  const spoofCount = data.per_model.filter(m => m.decision === 'spoof').length;
  const isSpoof = spoofCount > data.per_model.length / 2;
  const avgProb = data.per_model.reduce((s, m) => s + m.spoof_probability, 0) / data.per_model.length;
  const verdict = document.getElementById('verdict');
  verdict.className = 'verdict ' + (isSpoof ? 'spoof' : 'bona');
  document.getElementById('verdictIcon').textContent = isSpoof ? '🚨' : '✅';
  document.getElementById('verdictLabel').textContent = isSpoof ? 'DEEPFAKE' : 'СПРАВЖНІЙ ГОЛОС';
  document.getElementById('verdictSub').textContent =
    `Середня ймовірність спуфу: ${(avgProb * 100).toFixed(1)}%  •  ` +
    `Тривалість: ${data.duration_s.toFixed(2)}s`;
  const grid = document.getElementById('modelsGrid');
  grid.innerHTML = '';
  data.per_model.forEach(m => {
    const isS = m.decision === 'spoof';
    const pct = (m.spoof_probability * 100).toFixed(1);
    grid.innerHTML += `
      <div class="model-card">
        <h4>${m.model}</h4>
        <div class="prob ${isS ? 'spoof' : 'bona'}">${pct}%</div>
        <div style="font-size:0.75rem;color:#888;margin-top:2px">${m.decision} (thr ${m.threshold.toFixed(2)})</div>
        <div class="bar-wrap"><div class="bar ${isS ? 'spoof' : 'bona'}" style="width:${pct}%"></div></div>
      </div>`;
  });
  if (data.ablation) {
    document.getElementById('ablationTitle').textContent = `Channel Ablation (${data.ablation.model_name}) — що важливіше?`;
    const bars = document.getElementById('ablationBars');
    bars.innerHTML = '';
    const items = [
      { label: 'Full', value: data.ablation.full, color: '#6c7bff' },
      { label: 'Без Mel', value: data.ablation.ablate_mel, color: '#e74c3c' },
      { label: 'Без GD', value: data.ablation.ablate_gd, color: '#f39c12' },
      { label: 'Mel важл.', value: Math.max(0, data.ablation.mel_importance), color: '#2ecc71' },
      { label: 'GD важл.', value: Math.max(0, data.ablation.gd_importance), color: '#1abc9c' },
    ];
    items.forEach(item => {
      const h = Math.round(item.value * 80);
      bars.innerHTML += `
        <div class="abl-bar-wrap">
          <div class="abl-value">${(item.value * 100).toFixed(1)}%</div>
          <div class="abl-bar-bg">
            <div class="abl-bar-fill" style="height:${h}px;background:${item.color}"></div>
          </div>
          <div class="abl-label">${item.label}</div>
        </div>`;
    });
    document.getElementById('ablationSection').style.display = 'block';
  } else {
    document.getElementById('ablationSection').style.display = 'none';
  }
  document.getElementById('featuresImg').src = 'data:image/png;base64,' + data.features_img;
  document.getElementById('result').style.display = 'block';
}
</script>
</body>
</html>
"""
@app.route("/")
def index():
    return render_template_string(HTML)
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "немає файлу"}), 400
    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "порожній файл"}), 400
    active_filter = request.form.get("filter", "")
    try:
        eng = _load_engine()
    except Exception as e:
        return jsonify({"error": f"не вдалося завантажити модель: {e}"}), 500
    suffix = Path(f.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = Path(tmp.name)
    try:
        logger.info(f"Аналіз файлу: {tmp_path}, Фільтр: {active_filter or 'None'}")
        result = eng.predict_path(tmp_path, run_gradcam=True, active_filter=active_filter)
        for m in result.per_model:
            logger.info(f"Модель {m['model']} -> Prob: {m['spoof_probability']:.4f}, Decision: {m['decision']}")
    except Exception as e:
        logger.error(f"помилка аналізу: {e}")
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": f"помилка аналізу: {e}"}), 500
    finally:
        tmp_path.unlink(missing_ok=True)
    interm = result.intermediates
    wav = interm["waveform"]
    mel = interm["mel_db"]
    gd  = interm["group_delay"]
    cams = {k.replace("gradcam_", ""): v for k, v in interm.items() if k.startswith("gradcam_")}
    if not cams:
        cams = None
    fig = plot_features_panel(wav, mel, gd, cams=cams, title=f"{Path(result.file).name}")
    features_b64 = _fig_to_b64(fig)
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
    return jsonify({
        "file": result.file,
        "duration_s": result.duration_s,
        "per_model": result.per_model,
        "features_img": features_b64,
        "ablation": ablation,
    })
if __name__ == "__main__":
    if not INFER_CONFIG.exists():
        print(f"[warn] конфіг не знайдено: {INFER_CONFIG}")
        print("Створи configs/infer.yaml з описом моделей")
    print("Запуск на http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)