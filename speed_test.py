import torch
import time
import numpy as np
from pathlib import Path

from models.aasist_lite import AASISTLite
from models.lcnn import LCNN
from models.phase_aware import PhaseAwareDualStream
from models.se_resnet import SEResNet18

model_weights = {
    "AASIST-lite": "checkpoints/aasist_lite/last.pt",
    "LCNN": "checkpoints/lcnn/last.pt",
    "PhaseAware": "checkpoints/phase_aware/last.pt",
    "SEResNet18": "checkpoints/se_resnet18/last.pt"
}

def load_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state)

def measure_inference_stats(model, device, dummy_input, warmup_runs=20, test_runs=2000):
    model.eval()
    
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    times = []
    with torch.no_grad():
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000) # мс
            
    return np.mean(times), np.median(times), np.min(times), np.max(times), np.std(times)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")
    
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
            
    dummy_input = torch.randn(1, 2, 128, 400).to(device)
    
    print(f"{'Модель':<15} | {'Сер':<6} | {'Мед':<6} | {'Мін':<6} | {'Макс':<6} | {'Std':<6}")
    print("-"*75)
    
    for name, model in models.items():
        mean, median, min_t, max_t, std = measure_inference_stats(model, device, dummy_input)
        print(f"{name:<15} | {mean:>6.2f} | {median:>6.2f} | {min_t:>6.2f} | {max_t:>6.2f} | {std:>6.2f}")

if __name__ == "__main__":
    main()