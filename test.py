import os
from pathlib import Path

meta_path = "data/replaydf/replaydf_meta_all.csv"
data_dir = Path("data/replaydf")

with open(meta_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        # Пропускаємо порожні рядки і заголовок
        if len(parts) < 3 or parts[0] == "original_file":
            continue
        
        audio_rel_path = parts[1].strip()
        p = data_dir / audio_rel_path
        
        if not p.exists():
            print(f"❌ Python шукає і не бачить файл: {p}")
            
            if p.parent.exists():
                print(f"📁 Але папка існує! Її реальний вміст на диску (перші 5 файлів):")
                print(os.listdir(p.parent)[:5])
            else:
                print(f"☠️ Папки взагалі не існує на диску: {p.parent}")
            break