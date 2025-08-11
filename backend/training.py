from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .model import TinyCNN, CLASSES

@dataclass
class TrainConfig:
    session_dir: Path
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 16
    val_split: float = 0.2


class MosaicDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], train: bool):
        self.items = items
        self.train = train
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), y


def load_labeled_items(session_dir: Path) -> List[Tuple[Path,int]]:
    labels_path = session_dir / "labels.json"
    if not labels_path.exists():
        return []
    labels = json.loads(labels_path.read_text())
    out = []
    for rel, lab in labels.items():
        p = session_dir / rel
        if p.exists():
            out.append((p, CLASSES.index(lab)))
    return out


def split_train_val(items: List[Tuple[Path,int]], val_split: float):
    random.shuffle(items)
    n_val = max(1, int(len(items)*val_split)) if items else 0
    return items[n_val:], items[:n_val]


def train_model(cfg: TrainConfig, progress: Dict):
    items = load_labeled_items(cfg.session_dir)
    if len(items) < 6:
        raise RuntimeError("Not enough labeled mosaics to train (need >=6)")

    train_items, val_items = split_train_val(items, cfg.val_split)
    train_ds = MosaicDataset(train_items, train=True)
    val_ds   = MosaicDataset(val_items,   train=False)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, cfg.epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*xb.size(0)
            preds = logits.argmax(1)
            correct += (preds==yb).sum().item()
            total += xb.size(0)
        train_acc = correct/total if total else 0
        train_loss = loss_sum/total if total else 0

        # Validate
        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(1)
                vcorrect += (preds==yb).sum().item()
                vtotal += xb.size(0)
        val_acc = vcorrect/vtotal if vtotal else 0

        progress.update({
            "epoch": epoch,
            "epochs": cfg.epochs,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        if val_acc > best_val:
            best_val = val_acc
            (cfg.session_dir/"model"/"weights.pth").parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.session_dir/"model"/"weights.pth")

    progress.update({"status": "done", "best_val_acc": best_val})