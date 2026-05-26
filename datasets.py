"""PyTorch ``Dataset`` over pre-computed NPZ feature tensors.

Each NPZ contains a single key ``features`` of shape ``(2, n_mels, T)`` in
``float16`` (see ``scripts/precompute_features.py``). The dataset reads a
manifest CSV (``path,label,dataset``) where ``path`` is relative to
``features_root`` (with a ``.npz`` extension swapped in).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_TO_INDEX = {"bona-fide": 0, "spoof": 1}


def _swap_to_npz(rel_path: str) -> str:
    return str(Path(rel_path).with_suffix(".npz"))


class NpzFeatureDataset(Dataset):
    """A simple, fast dataset over NPZ feature tensors.

    Parameters
    ----------
    manifest_csv : str | Path
        CSV with columns ``path,label,dataset`` (see ``data/manifests.py``).
    features_root : str | Path
        Directory the manifest paths are relative to (after extension swap).
    frame_target : int
        Output time dimension. Longer tensors are randomly cropped (training)
        or centre-cropped (eval); shorter tensors are zero-padded.
    train : bool
        If ``True`` use random crop and optional augmentation.
    transform : callable
        Applied to the ``(2, n_mels, frame_target)`` tensor before return.
    """

    def __init__(self,
                 manifest_csv: str | Path,
                 features_root: str | Path,
                 frame_target: int = 400,
                 train: bool = True,
                 transform: Callable | None = None) -> None:
        super().__init__()
        self.features_root = Path(features_root)
        self.frame_target = int(frame_target)
        self.train = bool(train)
        self.transform = transform
        self.items: List[Tuple[str, int, str]] = self._read_manifest(manifest_csv)
        valid = []
        for rel, label, dataset in self.items:
            if (self.features_root / rel).exists():
                valid.append((rel, label, dataset))
        skipped = len(self.items) - len(valid)
        if skipped:
            print(f"[dataset] пропущено {skipped} відсутніх NPZ з {len(self.items)}")
        self.items = valid
        if not self.items:
            raise ValueError(f"Після фільтрації маніфест порожній: {manifest_csv}")
        if not self.items:
            raise ValueError(f"Empty / unreadable manifest: {manifest_csv}")

    @staticmethod
    def _read_manifest(path: str | Path) -> List[Tuple[str, int, str]]:
        rows: List[Tuple[str, int, str]] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                lbl = LABEL_TO_INDEX.get(row["label"])
                if lbl is None:
                    continue
                rows.append((_swap_to_npz(row["path"]), lbl, row.get("dataset", "")))
        return rows

    def __len__(self) -> int:
        return len(self.items)

    def _crop_or_pad(self, x: np.ndarray) -> np.ndarray:
        _c, _f, T = x.shape
        if T == self.frame_target:
            return x
        if T > self.frame_target:
            if self.train:
                start = random.randint(0, T - self.frame_target)
            else:
                start = (T - self.frame_target) // 2
            return x[:, :, start:start + self.frame_target]
        pad = self.frame_target - T
        return np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")

    def __getitem__(self, index: int):
        for attempt in range(len(self.items)):
            rel, label, dataset = self.items[(index + attempt) % len(self.items)]
            path = self.features_root / rel
            try:
                with np.load(path) as data:
                    feat = np.asarray(data["features"], dtype=np.float32)
                feat = self._crop_or_pad(feat)
                tensor = torch.from_numpy(feat)
                if self.transform is not None:
                    tensor = self.transform(tensor)
                return tensor, torch.tensor(label, dtype=torch.long), dataset
            except Exception:
                continue
        raise RuntimeError(f"No valid sample found starting at index {index}")


def collate_pad(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]):
    """Stack tensors and labels; keep ``dataset`` tag as a Python list."""
    feats = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    tags = [b[2] for b in batch]
    return feats, labels, tags
