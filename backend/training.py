from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import random
import math
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Try LightGBM; fall back to prototypes if not installed.
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # fallback path below

from .model import TinyCNN, CLASSES
from .utils import read_json, write_json_atomic

# -------------------------
# Config
# -------------------------

@dataclass
class TrainConfig:
    session_dir: Path
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 16
    val_split: float = 0.2


# -------------------------
# Baseline (legacy) dataset
# -------------------------

class MosaicDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], train: bool):
        self.items = items
        self.train = train
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            # NOTE: For charts, horizontal flips break time axis semantics; avoid flipping.
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


def _train_legacy_cnn(cfg: TrainConfig, progress: Dict):
    """
    Legacy trainer (TinyCNN on mosaics + labels.json).
    Kept for backward compatibility if group annotations are not present.
    """
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
            "mode": "legacy_cnn",
            "epoch": epoch,
            "epochs": cfg.epochs,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        if val_acc > best_val:
            best_val = val_acc
            (cfg.session_dir/"model").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.session_dir/"model"/"weights.pth")

    progress.update({"status": "done", "best_val_acc": best_val})


# -------------------------
# New pipeline: embeddings + heads
# -------------------------

def _load_backbone():
    """
    EfficientNet-B0 feature extractor (1280-dim), pretrained if available.
    Returns (model, transform).
    """
    try:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        pretrained = True
    except Exception:
        # No weights (air-gapped) – still fine for a first pass; you can fine-tune later.
        m = models.efficientnet_b0(weights=None)
        pretrained = False

    feat_dim = m.classifier[1].in_features
    m.classifier = nn.Identity()
    m.eval()
    if torch.cuda.is_available():
        m = m.cuda()

    tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        # Use ImageNet stats to match the backbone
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return m, tf, feat_dim, pretrained


def _embed_image(backbone, tf, path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        z = backbone(x).squeeze().detach().cpu().numpy()
    return z.astype(np.float32, copy=False)


def _time_features(ts_ms: int, tz_offset_minutes: int = 480) -> np.ndarray:
    # Simple MYT shift (+8h) by default; precise tz per-group can be added later if needed.
    ts = (ts_ms // 1000) + tz_offset_minutes * 60
    # Derive hour and weekday
    hour = (ts // 3600) % 24
    dow  = (ts // (3600*24) + 4) % 7  # 1970-01-01 was Thursday (=4)
    def sc(v: int, T: int):
        a = 2*math.pi*v/T
        return [math.sin(a), math.cos(a)]
    return np.array([*sc(hour,24), *sc(int(dow),7)], dtype=np.float32)


def _collect_roles(groups: List[Dict]) -> List[str]:
    roles = set()
    for g in groups:
        for z in g.get("zones", []):
            r = z.get("role")
            if r:
                roles.add(r)
    return sorted(roles)  # deterministic order


def _group_vector(session_dir: Path, g: Dict, roles: List[str], feat_dim: int, backbone, tf) -> np.ndarray:
    """
    Concatenate zone embeddings by fixed role order; zeros for missing roles.
    Append time features at the end.
    """
    by_role = {}
    for z in g.get("zones", []):
        role = z.get("role")
        path = session_dir / z.get("path")
        if role and path.exists():
            by_role.setdefault(role, path)

    vecs = []
    for r in roles:
        p = by_role.get(r)
        if p is not None:
            vecs.append(_embed_image(backbone, tf, p))
        else:
            vecs.append(np.zeros((feat_dim,), dtype=np.float32))

    tfeat = _time_features(int(g["timestamp"]))
    return np.concatenate([*vecs, tfeat], axis=0)


def _train_heads(session_dir: Path, roles: List[str], X: np.ndarray, Y: np.ndarray, label_ids: List[str]) -> Dict:
    """
    Train one-vs-rest LightGBM heads if available, else fall back to prototype
    (class centroid) scoring. Returns a dict ready to pickle.
    """
    out = {"roles": roles, "label_ids": label_ids, "type": None, "models": {}}

    if lgb is not None:
        models = {}
        for i, lid in enumerate(label_ids):
            y = Y[:, i]
            if y.sum() == 0:
                # No positives → fallback to a dummy prior
                models[lid] = ("prior", float(y.mean()))
                continue
            params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31, verbose=-1)
            dtrain = lgb.Dataset(X, label=y)
            booster = lgb.train(params, dtrain, num_boost_round=300)
            models[lid] = ("lgb", booster.dump_model())  # dump to JSON-ish so it's pickle-safe
        out["type"] = "lightgbm_dump"
        out["models"] = models
    else:
        # Prototype/centroid fallback
        centroids = {}
        for i, lid in enumerate(label_ids):
            mask = Y[:, i] == 1
            if mask.any():
                centroids[lid] = X[mask].mean(axis=0).astype(np.float32)
            else:
                centroids[lid] = None
        out["type"] = "prototypes"
        out["models"] = centroids

    (session_dir/"model").mkdir(parents=True, exist_ok=True)
    with open(session_dir/"model"/"heads.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


def _can_use_group_pipeline(session_dir: Path) -> bool:
    gi = read_json(session_dir/"group_index.json", {"groups": []})
    ann = read_json(session_dir/"annotations.json", {})
    # Require at least a handful of annotated groups
    annotated = [g for g in gi.get("groups", []) if str(g.get("timestamp")) in ann]
    return len(annotated) >= 6  # heuristic minimum


def _train_group_embeddings_and_heads(cfg: TrainConfig, progress: Dict):
    """
    New pipeline:
    1) Load group_index + annotations
    2) Build per-group vectors (concat zone embeddings by role + time feats)
    3) Train LightGBM one-vs-rest heads (or prototypes)
    4) Save heads.pkl
    """
    session_dir = cfg.session_dir
    gi = read_json(session_dir/"group_index.json", {"groups": []})
    ann = read_json(session_dir/"annotations.json", {})

    groups = [g for g in gi.get("groups", []) if str(g.get("timestamp")) in ann]
    if len(groups) < 6:
        raise RuntimeError("Not enough annotated groups to train (need >=6)")

    # Build label space from annotations' global labels
    # (You can expand this later to include vocab filtering, parents, etc.)
    label_ids = sorted({lid for t, rec in ann.items() for lid in rec.get("global", [])})
    if not label_ids:
        raise RuntimeError("No global labels found in annotations")

    roles = _collect_roles(groups)
    backbone, tf, feat_dim, pretrained = _load_backbone()
    vec_dim = len(roles)*feat_dim + 4  # +4 for time sin/cos features

    X_rows, Y_rows = [], []
    total = len(groups)
    for i, g in enumerate(groups, start=1):
        x = _group_vector(session_dir, g, roles, feat_dim, backbone, tf)
        # map labels
        y = np.zeros((len(label_ids),), dtype=np.int8)
        for lid in ann[str(g["timestamp"])].get("global", []):
            if lid in label_ids:
                y[label_ids.index(lid)] = 1
        X_rows.append(x)
        Y_rows.append(y)
        progress.update({
            "mode": "embeddings+heads",
            "stage": "extract",
            "count": i,
            "total": total
        })

    X = np.stack(X_rows, axis=0)
    Y = np.stack(Y_rows, axis=0)

    # Simple train/val split for reporting
    idx = np.arange(len(X))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx)*0.2))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    # Train heads
    heads = _train_heads(session_dir, roles, Xtr, Ytr, label_ids)

    # Quick val score (macro AUC or cosine to prototypes)
    metrics = {}
    if heads["type"] == "lightgbm_dump" and lgb is not None:
        # Rehydrate boosters from dump
        def load_booster(dump):
            b = lgb.Booster(model_str=json.dumps(dump))
            return b
        aucs = []
        for i, lid in enumerate(label_ids):
            kind, payload = heads["models"][lid]
            if kind != "lgb":  # prior only
                continue
            booster = load_booster(payload)
            pred = booster.predict(Xva)
            y = Yva[:, i]
            # simple AUC approximation if sklearn not present: use rank correlation proxy
            try:
                from sklearn.metrics import roc_auc_score  # type: ignore
                auc = float(roc_auc_score(y, pred))
            except Exception:
                # fallback: tie-aware ranking correlation proxy
                order = np.argsort(pred)
                auc = float((y[order].cumsum().sum())/(y.sum()*len(y)) if y.sum() else 0.5)
            aucs.append(auc)
        metrics["val_macro_auc"] = float(np.mean(aucs)) if aucs else None
    elif heads["type"] == "prototypes":
        # cosine to centroids
        def cos(a,b): 
            na = np.linalg.norm(a)+1e-9; nb=np.linalg.norm(b)+1e-9
            return float((a@b)/(na*nb))
        scores = []
        for i, lid in enumerate(label_ids):
            c = heads["models"][lid]
            if c is None:
                continue
            sims = np.array([cos(x, c) for x in Xva], dtype=np.float32)
            y = Yva[:, i]
            # naive threshold @ 0.0 sim
            pred = (sims > 0.0).astype(np.int32)
            acc = float((pred==y).mean())
            scores.append(acc)
        metrics["val_proto_acc"] = float(np.mean(scores)) if scores else None

    # Save a small train report
    report = {
        "mode": "embeddings+heads",
        "roles": roles,
        "n_groups": int(len(groups)),
        "vec_dim": int(vec_dim),
        "labels": label_ids,
        "metrics": metrics
    }
    write_json_atomic(session_dir/"model"/"train_report.json", report)

    # Mark as done
    progress.update({"status": "done", "report": report})


def train_model(cfg: TrainConfig, progress: Dict):
    """
    Unified entry point. If group annotations exist, run the new
    embeddings+heads pipeline. Otherwise, fall back to legacy TinyCNN.
    Progress dict is updated for /api/train/status.
    """
    progress.update({"status": "running", "stage": "detect_pipeline"})
    try:
        if _can_use_group_pipeline(cfg.session_dir):
            _train_group_embeddings_and_heads(cfg, progress)
        else:
            _train_legacy_cnn(cfg, progress)
    except Exception as e:
        progress.update({"status": "error", "error": str(e)})
        raise
