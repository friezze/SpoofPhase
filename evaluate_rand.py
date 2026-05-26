import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from features import get_audio_features, stack_features
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

# Імпорт моделей
from models.aasist_lite import AASISTLite
from models.lcnn import LCNN
from models.phase_aware import PhaseAwareDualStream
from models.se_resnet import SEResNet18

# ВПИШИ СВОЇ ШЛЯХИ
model_weights = {
    "AASIST": "checkpoints/aasist_lite/last.pt",
    "LCNN": "checkpoints/lcnn/last.pt",
    "PhaseAware": "checkpoints/phase_aware/last.pt",
    "SEResNet": "checkpoints/se_resnet18/last.pt"
}

class ExternalDataset(Dataset):
    def __init__(self, bona_dir: str, spoof_dir: str, limit_per_class: int = 5000):
        self.bona_files = list(Path(bona_dir).rglob("*.wav"))
        self.spoof_files = list(Path(spoof_dir).rglob("*.wav"))
        
        count = min(len(self.bona_files), len(self.spoof_files), limit_per_class)
        self.samples = []
        for f in random.sample(self.bona_files, count): self.samples.append((f, 0))
        for f in random.sample(self.spoof_files, count): self.samples.append((f, 1))
        random.shuffle(self.samples)
        print(f"Завантажено: {len(self.samples)} файлів")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            mel, gd, _, _ = get_audio_features(path)
            feat = torch.from_numpy(stack_features(mel, gd))
            if feat.shape[-1] < 400: feat = F.pad(feat, (0, 400 - feat.shape[-1]))
            else: feat = feat[:, :, :400]
            return feat, torch.tensor(label, dtype=torch.long)
        except:
            return torch.zeros((2, 128, 400)), torch.tensor(0, dtype=torch.long)

def load_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ExternalDataset("data/mailabs", "data/mlaad", limit_per_class=1000)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    models = {
        "AASIST": AASISTLite().to(device),
        "LCNN": LCNN().to(device),
        "PhaseAware": PhaseAwareDualStream().to(device),
        "SEResNet": SEResNet18().to(device)
    }
    
    results = {}
    for name, model in models.items():
        path = model_weights.get(name)
        if path and Path(path).exists():
            try:
                load_weights(model, path, device)
                print(f"✅ {name}: ваги успішно завантажено.")
            except Exception as e:
                print(f"❌ {name}: помилка завантаження: {e}")
        else:
            print(f"⚠️ {name}: ваги не знайдені, працюємо на випадкових.")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feat, lbl in tqdm(loader, desc=f"Тест {name}"):
                logits = model(feat.to(device))
                probs = F.softmax(logits, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(lbl.numpy())
        
        y_pred = [1 if p >= 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_labels, y_pred)
        print(f"📈 {name} Accuracy: {acc:.4f}")
        results[name] = {"y_true": all_labels, "y_scores": all_preds}

    # Малювання ROC
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
        plt.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    plt.legend(); plt.savefig("roc_external.png")
    
    # Малювання Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (name, res) in enumerate(results.items()):
        y_pred = [1 if p >= 0.5 else 0 for p in res["y_scores"]]
        cm = confusion_matrix(res["y_true"], y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Spoof", "Bona-fide"], yticklabels=["Spoof", "Bona-fide"])
        axes[idx].set_title(f"{name} Matrix")
    
    plt.tight_layout(); plt.savefig("confusion_matrices.png")
    print("Графіки збережено: roc_external.png, confusion_matrices.png")

if __name__ == "__main__":
    main()