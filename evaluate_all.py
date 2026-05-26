import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
from tqdm import tqdm
import numpy as np
import warnings
import random

warnings.filterwarnings("ignore")

from models.aasist_lite import AASISTLite
from models.lcnn import LCNN
from models.phase_aware import PhaseAwareDualStream
from models.se_resnet import SEResNet18
from features import get_audio_features, stack_features

model_weights = {
    "AASIST-lite": "checkpoints/aasist_lite/last.pt",
    "LCNN": "checkpoints/lcnn/last.pt",
    "PhaseAware": "checkpoints/phase_aware/last.pt",
    "SEResNet18": "checkpoints/se_resnet18/last.pt"
}

class ReplayDFDataset(Dataset):
    def __init__(self, meta_path: str, data_dir: str, max_frames: int = 400):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.samples = []
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 3 or parts[0] == "original_file": continue
                audio_path = self.data_dir / parts[1].strip()
                label = 1 if parts[2].strip() == "bona-fide" else 0
                if audio_path.exists(): self.samples.append((audio_path, label))

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int):
        audio_path, label = self.samples[idx]
        try:
            mel, gd, _, _ = get_audio_features(audio_path)
            feat_tensor = torch.from_numpy(stack_features(mel, gd))
        except:
            return torch.zeros((2, 128, self.max_frames)), torch.tensor(label, dtype=torch.long)
        
        t_len = feat_tensor.shape[-1]
        if t_len < self.max_frames: feat_tensor = F.pad(feat_tensor, (0, self.max_frames - t_len))
        else: feat_tensor = feat_tensor[:, :, :self.max_frames]
        label = 1 - label
        return feat_tensor, torch.tensor(label, dtype=torch.long)

def load_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state)

# ── БАЗОВІ ФІЛЬТРИ ───────────────────────────────────────────────────────────

def filter_bandpass(features):
    mask = torch.ones_like(features)
    mask[:, :, :2, :] = 0 
    mask[:, :, 120:, :] = 0
    return features * mask

def filter_mean_subtraction(features):
    mean_val = features.mean(dim=-1, keepdim=True)
    return features - mean_val

def filter_telegram_lowpass(features):
    mask = torch.ones_like(features)
    mask[:, :, 80:, :] = 0
    return features * mask

def filter_spec_augment(features):
    b, c, f, t = features.shape
    f_idx = torch.randint(0, max(1, f - 15), (1,))
    features_clone = features.clone()
    features_clone[:, :, f_idx:f_idx+15, :] = 0
    return features_clone

def filter_tg_soft_lowpass(features):
    b, c, f, t = features.shape
    weight = torch.ones(f, device=features.device)
    weight[55:85] = torch.linspace(1.0, 0.05, 30, device=features.device)
    weight[85:] = 0.05
    return features * weight.view(1, 1, f, 1)

def filter_temporal_blur(features):
    b, c, f, t = features.shape
    kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=features.device).view(1, 1, 1, -1)
    feat_pad = F.pad(features, (2, 2, 0, 0), mode='replicate')
    feat_reshaped = feat_pad.view(b * c * f, 1, 1, t + 4)
    blurred = F.conv2d(feat_reshaped, kernel)
    return blurred.view(b, c, f, t)

def filter_noise_gate(features):
    noise_thresh = features.mean() - 0.5 * features.std()
    mask = (features > noise_thresh).float()
    return features * mask

def filter_delta_boost(features):
    delta = torch.zeros_like(features)
    delta[:, :, :, 1:-1] = features[:, :, :, 2:] - features[:, :, :, :-2]
    return features + (delta * 0.5)

# ── КОМБІНАЦІЇ ФІЛЬТРІВ ──────────────────────────────────────────────────────

def filter_bp_gate(features):
    return filter_noise_gate(filter_bandpass(features))

def filter_bp_delta(features):
    return filter_delta_boost(filter_bandpass(features))

def filter_gate_delta(features):
    return filter_delta_boost(filter_noise_gate(features))

def filter_bp_gate_delta(features):
    return filter_delta_boost(filter_noise_gate(filter_bandpass(features)))

def filter_bp_softlow_gate(features):
    return filter_noise_gate(filter_tg_soft_lowpass(filter_bandpass(features)))

filters = {
    "bandpass": filter_bandpass,
    "mean_sub": filter_mean_subtraction,
    "tg_lowpass": filter_telegram_lowpass,
    "spec_aug": filter_spec_augment,
    "tg_soft_lowpass": filter_tg_soft_lowpass,
    "temporal_blur": filter_temporal_blur,
    "noise_gate": filter_noise_gate,
    "delta_boost": filter_delta_boost,
    "comb_bp_gate": filter_bp_gate,
    "comb_bp_delta": filter_bp_delta,
    "comb_gate_delta": filter_gate_delta,
    "comb_bp_gate_delta": filter_bp_gate_delta,
    "comb_bp_soft_gate": filter_bp_softlow_gate
}

# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, dataloader, device, filter_func):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc=f"Оцінка {model.__class__.__name__}"):
            features = filter_func(features.to(device))
            logits = model(features)
            all_preds.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ReplayDFDataset("data/replaydf/replaydf_meta_all.csv", "data/replaydf")
    
    limit = 1200
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    dataset = Subset(dataset, indices[:limit])
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    models = {
        "AASIST-lite": AASISTLite().to(device),
        "LCNN": LCNN().to(device),
        "PhaseAware": PhaseAwareDualStream().to(device),
        "SEResNet18": SEResNet18().to(device)
    }
    
    for name, model in models.items():
        if Path(model_weights[name]).exists():
            load_weights(model, model_weights[name], device)
            print(f"✅ {name}: ваги завантажено.")
        else:
            print(f"⚠️ {name}: випадкові ваги.")

    for filter_name, filter_func in filters.items():
        print(f"\n--- Запуск з фільтром: {filter_name} ---")
        results = {}
        for name, model in models.items():
            y_true, y_scores = evaluate_model(model, dataloader, device, filter_func)
            results[name] = {"y_true": y_true, "y_scores": y_scores}

        plt.figure(figsize=(10, 8))
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
            plt.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate (Справжні голоси, які помилково назвали фейком)")
        plt.ylabel("True Positive Rate (Фейки, які правильно знайшли)")
        plt.title(f"ROC Curves - Фільтр: {filter_name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"roc_curves_{filter_name}.png")
        plt.close()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, (name, res) in enumerate(results.items()):
            y_pred = [1 if p >= 0.5 else 0 for p in res["y_scores"]]
            cm = confusion_matrix(res["y_true"], y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes.flatten()[idx],
                        xticklabels=["Bona-fide (Справжній)", "Spoof (Фейк)"], 
                        yticklabels=["Bona-fide (Справжній)", "Spoof (Фейк)"])
            axes.flatten()[idx].set_title(f"{name}")
            axes.flatten()[idx].set_xlabel("Прогноз моделі")
            axes.flatten()[idx].set_ylabel("Справжній клас")
        
        fig.suptitle(f"Матриці помилок - Фільтр: {filter_name}")
        plt.tight_layout()
        plt.savefig(f"confusion_matrices_{filter_name}.png")
        plt.close()
        print(f"Збережено: roc_curves_{filter_name}.png та confusion_matrices_{filter_name}.png")

if __name__ == "__main__":
    main()