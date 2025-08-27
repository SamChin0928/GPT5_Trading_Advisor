# backend/training.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import random
import math
import pickle
import os
import re
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
from .utils import read_json, write_json_atomic, load_annotations as load_ann

# -------------------------
# Config
# -------------------------

@dataclass
class TrainConfig:
    session_dir: Path                 # directory that contains group_index.json + images (can be a merged workspace)
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 16
    val_split: float = 0.2            # legacy CNN only


# -------------------------
# Helpers (auto params, metrics)
# -------------------------

def _auto_cnn_hparams(n: int) -> Dict[str, int | float]:
    # Simple heuristics that behave well across small/medium datasets
    epochs = int(min(60, max(6, round(8 * math.log10(max(10, n))))))
    batch = 16 if n < 500 else 32
    patience = max(3, epochs // 4)
    return {"epochs": epochs, "batch": batch, "patience": patience}

def _auto_lgb_hparams(n: int) -> Dict[str, int | float]:
    num_leaves = 31 if n < 2000 else 63
    lr = 0.05 if n < 10000 else 0.03
    rounds = int(min(1000, max(100, 3 * math.sqrt(max(16, n)))))  # grows ~sqrt(n)
    early = max(20, rounds // 10)
    return {"num_leaves": num_leaves, "lr": lr, "rounds": rounds, "early": early}

def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    # pick threshold that maximizes F1 on validation
    if y_prob.size == 0:
        return 0.5, 0.0
    # Use ~201 quantiles for speed
    qs = np.unique(np.quantile(y_prob, np.linspace(0, 1, 201)))
    best_f1, best_t = 0.0, 0.5
    for t in qs:
        pred = (y_prob >= t).astype(np.int32)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# -------------------------
# Utility: robust timestamp parsing
# -------------------------

_DIGITS = re.compile(r"(\d{10,17})")

def _ts_to_int_maybe(ts_val) -> int:
    """
    Accept plain millisecond ints/strings, or composite strings like 'SID__1691234567890'.
    If nothing parseable is found, return 0.
    """
    s = str(ts_val)
    # common case: '1234567890123'
    try:
        return int(s)
    except Exception:
        pass
    # composite 'SID__1234567890123'
    if "__" in s:
        try:
            return int(s.split("__")[-1])
        except Exception:
            pass
    # last resort: find a long digit run
    m = _DIGITS.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 0


# -------------------------
# Baseline (legacy) dataset
# -------------------------

class MosaicDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]]):
        self.items = items
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self): return len(self.items)

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
        if p.exists() and lab in CLASSES:
            out.append((p, CLASSES.index(lab)))
    return out


def split_train_val(items: List[Tuple[Path,int]], val_split: float):
    random.shuffle(items)
    n_val = max(1, int(len(items)*val_split)) if items else 0
    return items[n_val:], items[:n_val]


def _train_legacy_cnn(cfg: TrainConfig, progress: Dict):
    """
    Legacy trainer (TinyCNN on mosaics + labels.json).
    Auto-params + early stopping added.
    """
    print(f"[train] legacy CNN starting in: {cfg.session_dir}")
    items = load_labeled_items(cfg.session_dir)
    if len(items) < 6:
        raise RuntimeError("Not enough labeled mosaics to train (need >=6)")

    # Auto params
    hp = _auto_cnn_hparams(len(items))
    epochs = hp["epochs"]
    batch  = hp["batch"]
    patience = hp["patience"]

    train_items, val_items = split_train_val(items, cfg.val_split)
    train_ds = MosaicDataset(train_items)
    val_ds   = MosaicDataset(val_items)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Class weights (handle imbalance)
    counts = np.bincount([y for _, y in items], minlength=len(CLASSES))
    inv = (counts.max() + 1e-6) / (counts + 1e-6)
    w = torch.tensor(inv / inv.sum() * len(CLASSES), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best_val, best_state = 0.0, None
    stalls = 0
    for epoch in range(1, int(epochs)+1):
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

        improved = val_acc > best_val + 1e-4
        if improved:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stalls = 0
        else:
            stalls += 1

        progress.update({
            "mode": "legacy_cnn",
            "epoch": epoch,
            "epochs": int(epochs),
            "auto_params": {"batch": batch, "patience": patience},
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        if stalls >= patience:
            break

    # Save best
    (cfg.session_dir/"model").mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(),
               cfg.session_dir/"model"/"weights.pth")
    print(f"[train] saved weights.pth to {cfg.session_dir/'model'} (best_val={best_val:.3f})")
    progress.update({"status": "done", "best_val_acc": float(best_val)})


# -------------------------
# New pipeline: embeddings + heads
# -------------------------

# Cache the backbone once per process
_BACKBONE_CACHE = None  # (model, transform, feat_dim, pretrained)

def _load_backbone():
    """
    Offline-safe EfficientNet-B0 feature extractor.
    We DO NOT attempt to download weights unless CP_USE_PRETRAINED=1 is set.
    """
    global _BACKBONE_CACHE
    if _BACKBONE_CACHE is not None:
        return _BACKBONE_CACHE

    use_pretrained = os.environ.get("CP_USE_PRETRAINED", "0") in ("1", "true", "True")
    pretrained = False
    if use_pretrained:
        try:
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            pretrained = True
        except Exception:
            m = models.efficientnet_b0(weights=None)
            pretrained = False
    else:
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
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    _BACKBONE_CACHE = (m, tf, feat_dim, pretrained)
    print(f"[train] Backbone loaded (EfficientNet-B0, pretrained={pretrained})")
    return _BACKBONE_CACHE


def _embed_image(backbone, tf, path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        z = backbone(x).squeeze().detach().cpu().numpy()
    return z.astype(np.float32, copy=False)


def _time_features(ts_ms: int, tz_offset_minutes: int = 480) -> np.ndarray:
    ts = (ts_ms // 1000) + tz_offset_minutes * 60
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
            if r: roles.add(r)
    return sorted(roles)  # deterministic


def _group_vector(session_dir: Path, g: Dict, roles: List[str], feat_dim: int, backbone, tf) -> np.ndarray:
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
    ts_ms = _ts_to_int_maybe(g.get("timestamp"))
    tfeat = _time_features(int(ts_ms))
    return np.concatenate([*vecs, tfeat], axis=0)


def _can_use_group_pipeline(session_dir: Path) -> bool:
    gi = read_json(session_dir/"group_index.json", {"groups": []})
    ann = load_ann(session_dir)  # CENTRAL annotations
    annotated = [g for g in gi.get("groups", []) if str(g.get("timestamp")) in ann]
    return len(annotated) >= 6  # heuristic minimum


def _train_heads(Xtr: np.ndarray, Ytr: np.ndarray, Xva: np.ndarray, Yva: np.ndarray,
                 roles: List[str], label_ids: List[str], session_dir: Path) -> Dict:
    """
    Train one-vs-rest LightGBM heads with early stopping if available,
    else fall back to prototype centroids. Also computes per-label thresholds.
    """
    out = {"roles": roles, "label_ids": label_ids, "type": None, "models": {}, "thresholds": {}}

    if lgb is not None:
        # Auto params based on training size
        hp = _auto_lgb_hparams(len(Xtr))
        params = dict(objective="binary", metric="auc",
                      learning_rate=hp["lr"], num_leaves=hp["num_leaves"], verbose=-1)
        models = {}
        thresholds = {}
        for i, lid in enumerate(label_ids):
            ytr = Ytr[:, i]
            yva = Yva[:, i]
            if ytr.sum() == 0:
                # No positives → prior
                prior = float(ytr.mean())
                models[lid] = ("prior", prior)
                thresholds[lid] = 0.5
                continue
            dtr = lgb.Dataset(Xtr, label=ytr)
            dva = lgb.Dataset(Xva, label=yva, reference=dtr)
            booster = lgb.train(params, dtr, num_boost_round=hp["rounds"],
                                valid_sets=[dva], valid_names=["val"],
                                callbacks=[lgb.early_stopping(hp["early"], verbose=False)])
            # Save dump for portability
            models[lid] = ("lgb", booster.dump_model())
            # Threshold using validation predictions
            pva = booster.predict(Xva)
            thr, _ = _best_threshold(yva, pva)
            thresholds[lid] = float(thr)
        out["type"] = "lightgbm_dump"
        out["models"] = models
        out["thresholds"] = thresholds
    else:
        # Prototype fallback
        centroids = {}
        thresholds = {}
        for i, lid in enumerate(label_ids):
            mask = Ytr[:, i] == 1
            if mask.any():
                c = Xtr[mask].mean(axis=0).astype(np.float32)
                centroids[lid] = c
                # Cosine sim → [0,1] score → fit threshold on val
                def _cos(a,b):
                    na = np.linalg.norm(a)+1e-9; nb=np.linalg.norm(b)+1e-9
                    return float((a@b)/(na*nb))
                sims = np.array([_cos(x, c) for x in Xva], dtype=np.float32)
                prob = (sims + 1.0) / 2.0
                thr, _ = _best_threshold(Yva[:, i], prob)
                thresholds[lid] = float(thr)
            else:
                centroids[lid] = None
                thresholds[lid] = 0.5
        out["type"] = "prototypes"
        out["models"] = centroids
        out["thresholds"] = thresholds

    (session_dir/"model").mkdir(parents=True, exist_ok=True)
    with open(session_dir/"model"/"heads.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"[train] saved heads.pkl to {session_dir/'model'}")
    return out


def _train_group_embeddings_and_heads(cfg: TrainConfig, progress: Dict):
    """
    New pipeline:
    1) Load group_index + CENTRAL annotations
    2) Build per-group vectors (concat zone embeddings by role + time feats)
    3) Train LightGBM one-vs-rest heads (or prototypes) with thresholds
    4) Save heads.pkl + train_report.json
    """
    print(f"[train] embeddings+heads starting in: {cfg.session_dir}")
    session_dir = cfg.session_dir
    gi = read_json(session_dir/"group_index.json", {"groups": []})
    ann = load_ann(session_dir)  # CENTRAL
    groups = [g for g in gi.get("groups", []) if str(g.get("timestamp")) in ann]
    if len(groups) < 6:
        raise RuntimeError("Not enough annotated groups to train (need >=6)")

    # Label space from global labels
    label_ids = sorted({lid for _, rec in ann.items() for lid in (rec.get("global") or [])})
    if not label_ids:
        raise RuntimeError("No global labels found in annotations")

    roles = _collect_roles(groups)

    # Let UI show progress while backbone loads
    progress.update({"mode": "embeddings+heads", "stage": "load_backbone"})
    backbone, tf, feat_dim, pretrained = _load_backbone()
    vec_dim = len(roles)*feat_dim + 4  # +4 for time sin/cos features

    X_rows, Y_rows = [], []
    total = len(groups)
    for i, g in enumerate(groups, start=1):
        x = _group_vector(session_dir, g, roles, feat_dim, backbone, tf)
        y = np.zeros((len(label_ids),), dtype=np.int8)
        for lid in (ann[str(g["timestamp"])].get("global") or []):
            if lid in label_ids:
                y[label_ids.index(lid)] = 1
        X_rows.append(x); Y_rows.append(y)
        progress.update({"mode": "embeddings+heads", "stage": "extract", "count": i, "total": total})

    X = np.stack(X_rows, axis=0); Y = np.stack(Y_rows, axis=0)

    # Simple train/val split for reporting
    idx = np.arange(len(X))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx)*0.2))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    # Train heads (auto params inside)
    heads = _train_heads(Xtr, Ytr, Xva, Yva, roles, label_ids, session_dir)

    # Validation metrics
    metrics = {}
    if heads["type"] == "lightgbm_dump" and lgb is not None:
        def load_booster(dump): return lgb.Booster(model_str=json.dumps(dump))
        per_label = {}
        for i, lid in enumerate(label_ids):
            kind, payload = heads["models"][lid]
            if kind == "prior":
                pva = np.full((len(Xva),), float(payload), dtype=np.float32)
            else:
                booster = load_booster(payload)
                pva = booster.predict(Xva)
            thr = heads["thresholds"].get(lid, 0.5)
            pred = (pva >= thr).astype(np.int32)
            y = Yva[:, i]
            tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
            fn = int(((pred==0)&(y==1)).sum())
            prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
            f1 = 2*prec*rec/(prec+rec+1e-9)
            per_label[lid] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(y.sum())}
        metrics["per_label"] = per_label
    elif heads["type"] == "prototypes":
        per_label = {}
        def cos(a,b):
            na = np.linalg.norm(a)+1e-9; nb=np.linalg.norm(b)+1e-9
            return float((a@b)/(na*nb))
        for i, lid in enumerate(label_ids):
            c = heads["models"][lid]
            if c is None: 
                per_label[lid] = {"precision": None, "recall": None, "f1": None, "support": int(Yva[:,i].sum())}
                continue
            sims = np.array([cos(x, c) for x in Xva], dtype=np.float32)
            pva = (sims + 1.0)/2.0
            thr = heads["thresholds"].get(lid, 0.5)
            pred = (pva >= thr).astype(np.int32)
            y = Yva[:, i]
            tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
            fn = int(((pred==0)&(y==1)).sum())
            prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
            f1 = 2*prec*rec/(prec+rec+1e-9)
            per_label[lid] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(y.sum())}
        metrics["per_label"] = per_label

    report = {
        "mode": "embeddings+heads",
        "roles": roles,
        "n_groups": int(len(groups)),
        "vec_dim": int(vec_dim),
        "labels": label_ids,
        "metrics": metrics,
        "auto_params": _auto_lgb_hparams(len(Xtr))
    }
    write_json_atomic(session_dir/"model"/"train_report.json", report)
    print(f"[train] saved train_report.json to {session_dir/'model'}")
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
        print(f"[train] error: {e}")
        raise


# -------------------------
# Inference helpers for heads (used by API)
# -------------------------

def _get_group_by_ts(session_dir: Path, ts: str) -> Optional[Dict]:
    gi = read_json(session_dir/"group_index.json", {"groups": []})
    for g in gi.get("groups", []):
        if str(g.get("timestamp")) == str(ts):
            return g
    return None

def _rehydrate_booster(dump) -> "lgb.Booster":
    if lgb is None:
        raise RuntimeError("LightGBM heads present but LightGBM is not installed.")
    return lgb.Booster(model_str=json.dumps(dump))  # type: ignore

def predict_group_probs(session_dir: Path, timestamp: str) -> Dict[str, float]:
    """Return per-label probability-like scores for a specific group timestamp using heads.pkl."""
    heads_path = session_dir/"model"/"heads.pkl"
    if not heads_path.exists():
        raise FileNotFoundError("heads.pkl not found. Train the embeddings+heads pipeline first.")
    with open(heads_path, "rb") as f:
        heads = pickle.load(f)

    g = _get_group_by_ts(session_dir, timestamp)
    if not g:
        raise FileNotFoundError(f"group {timestamp} not found")

    backbone, tf, feat_dim, _ = _load_backbone()
    x = _group_vector(session_dir, g, heads["roles"], feat_dim, backbone, tf)
    x = x.reshape(1, -1)

    scores: Dict[str, float] = {}
    if heads["type"] == "lightgbm_dump" and lgb is not None:
        for lid in heads["label_ids"]:
            kind, payload = heads["models"][lid]
            if kind == "prior":
                p = float(payload)
            else:
                booster = _rehydrate_booster(payload)
                p = float(booster.predict(x)[0])
            scores[lid] = p
    else:
        # prototypes
        def cos(a,b):
            na = np.linalg.norm(a)+1e-9; nb=np.linalg.norm(b)+1e-9
            return float((a@b)/(na*nb))
        for lid in heads["label_ids"]:
            c = heads["models"][lid]
            if c is None:
                scores[lid] = 0.0
            else:
                sim = cos(x[0], np.asarray(c, dtype=np.float32))
                scores[lid] = float((sim + 1.0)/2.0)
    return scores


def evaluate_session(session_dir: Path) -> Dict:
    """
    Evaluate heads on all annotated groups in this session using CENTRAL annotations.
    Returns overall & per-label metrics, plus per-ts predictions.
    """
    heads_path = session_dir/"model"/"heads.pkl"
    if not heads_path.exists():
        raise FileNotFoundError("heads.pkl not found. Train the embeddings+heads pipeline first.")
    with open(heads_path, "rb") as f:
        heads = pickle.load(f)

    ann = load_ann(session_dir)
    gi = read_json(session_dir/"group_index.json", {"groups": []})
    ts_list = [str(g["timestamp"]) for g in gi.get("groups", []) if str(g.get("timestamp")) in ann]
    if not ts_list:
        return {"n_eval": 0, "per_label": {}, "overall": {}}

    thresholds = heads.get("thresholds", {})
    per_label = {lid: {"tp":0,"fp":0,"fn":0,"support":0} for lid in heads["label_ids"]}
    preds_by_ts: Dict[str, Dict[str, float]] = {}

    for ts in ts_list:
        probs = predict_group_probs(session_dir, ts)
        preds_by_ts[ts] = probs
        true = set(ann[ts].get("global") or [])
        for lid in heads["label_ids"]:
            p = probs.get(lid, 0.0)
            th = thresholds.get(lid, 0.5)
            pred = (p >= th)
            is_true = (lid in true)
            if is_true: per_label[lid]["support"] += 1
            if pred and is_true: per_label[lid]["tp"] += 1
            elif pred and not is_true: per_label[lid]["fp"] += 1
            elif (not pred) and is_true: per_label[lid]["fn"] += 1

    # compute metrics
    per_label_metrics = {}
    micro_tp = micro_fp = micro_fn = 0
    for lid, d in per_label.items():
        tp, fp, fn, sup = d["tp"], d["fp"], d["fn"], d["support"]
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        per_label_metrics[lid] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(sup)}
        micro_tp += tp; micro_fp += fp; micro_fn += fn
    micro_prec = micro_tp/(micro_tp+micro_fp+1e-9)
    micro_rec  = micro_tp/(micro_tp+micro_fn+1e-9)
    micro_f1   = 2*micro_prec*micro_rec/(micro_prec+micro_rec+1e-9)

    return {
        "n_eval": len(ts_list),
        "per_label": per_label_metrics,
        "overall": {"micro_precision": float(micro_prec), "micro_recall": float(micro_rec), "micro_f1": float(micro_f1)},
        "preds": preds_by_ts,
        "thresholds": thresholds,
        "labels": heads["label_ids"]
    }

# --------- Incremental updates (optional, non-breaking) ---------

_INC_STATE = "inc_state.json"  # sidecar to keep counts etc.

def _heads_file(heads_dir: Path) -> Path:
    return heads_dir / "heads.pkl"

def _state_file(heads_dir: Path) -> Path:
    return heads_dir / _INC_STATE

def _load_heads_dict(heads_dir: Path) -> Optional[Dict[str, Any]]:
    p = _heads_file(heads_dir)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

def _save_heads_dict(heads_dir: Path, heads: Dict[str, Any]) -> None:
    with open(_heads_file(heads_dir), "wb") as f:
        pickle.dump(heads, f)

def _load_inc_state(heads_dir: Path) -> Dict[str, Any]:
    p = _state_file(heads_dir)
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {"counts": {}}  # label_id -> int

def _save_inc_state(heads_dir: Path, st: Dict[str, Any]) -> None:
    _state_file(heads_dir).write_text(json.dumps(st, indent=2))

def _cos01(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-9)
    nb = float(np.linalg.norm(b) + 1e-9)
    return float((np.dot(a, b) / (na * nb) + 1.0) / 2.0)

def _derive_pred_sets_from_feedback(feedback: Optional[Dict[str, Any]],
                                    thresholds: Dict[str, float]) -> Tuple[set, Dict[str, float]]:
    """
    Returns (pred_pos, score_by_label) using feedback['global']['pred'] if present.
    Fallback: empty sets.
    """
    pred_pos = set()
    score_map = {}
    try:
        g = (feedback or {}).get("global") or {}
        preds = g.get("pred") or []  # [["lid", score], ...]
        for item in preds:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lid = str(item[0])
                s = float(item[1])
                score_map[lid] = s
                if s >= float(thresholds.get(lid, 0.5)):
                    pred_pos.add(lid)
    except Exception:
        pass
    return pred_pos, score_map

def incremental_update_from_annotation(
    session_dir: Path,
    timestamp: str | int,
    positive_labels: List[str],
    feedback: Optional[Dict[str, Any]] = None,
    heads_dir: Optional[Path] = None,
    alpha_threshold: float = 0.06,
) -> Optional[Dict[str, Any]]:
    """
    Incrementally update heads with a single labeled group:
      - If heads type is 'prototypes': update label centroids with a running mean.
      - For ALL types: gently adapt per-label thresholds using EMA and feedback.

    Args:
        session_dir: the session that contains images/group_index for this timestamp
        timestamp: group timestamp
        positive_labels: authoritative labels saved by the user (global labels)
        feedback: optional model_feedback blob (from annotate payload)
        heads_dir: directory that contains heads.pkl (e.g., GLOBAL heads) — REQUIRED by caller
        alpha_threshold: EMA rate for threshold adaptation

    Returns:
        Updated heads dict, or None if heads not found.
    """
    # Resolve heads dir/file
    if heads_dir is None:
        heads_dir = session_dir / "model"
    heads = _load_heads_dict(heads_dir)
    if heads is None:
        return None

    label_ids: List[str] = list(heads.get("label_ids") or [])
    thresholds: Dict[str, float] = dict(heads.get("thresholds") or {})
    roles: List[str] = list(heads.get("roles") or [])

    # Find group & compute vector
    g = _get_group_by_ts(session_dir, str(timestamp))
    if not g:
        return None
    backbone, tf, feat_dim, _ = _load_backbone()
    x = _group_vector(session_dir, g, roles, feat_dim, backbone, tf)
    x = x.astype(np.float32, copy=False)
    x_norm = x / (np.linalg.norm(x) + 1e-9)

    # Prepare feedback-derived sets
    pred_pos_from_fb, score_from_fb = _derive_pred_sets_from_feedback(feedback, thresholds)
    truth = set(map(str, positive_labels))

    # --- Prototype centroid update (running mean on normalized vectors) ---
    st = _load_inc_state(heads_dir)
    if heads.get("type") in ("prototypes", "proto"):
        models = heads.get("models") or {}
        # make numpy views
        np_models: Dict[str, Optional[np.ndarray]] = {}
        for lid in set(label_ids) | truth:
            v = models.get(lid, None)
            np_models[lid] = (None if v is None else np.asarray(v, dtype=np.float32))

        # register new labels if needed
        for lid in truth:
            if lid not in label_ids:
                label_ids.append(lid)
                np_models[lid] = None
                thresholds[lid] = thresholds.get(lid, 0.5)

        # running mean for positives
        for lid in truth:
            n = int(st["counts"].get(lid, 0))
            c = np_models[lid]
            if c is None or n <= 0:
                np_models[lid] = x_norm.copy()
                st["counts"][lid] = 1
            else:
                new_c = (c * n + x_norm) / (n + 1.0)
                new_c = new_c / (np.linalg.norm(new_c) + 1e-9)
                np_models[lid] = new_c
                st["counts"][lid] = n + 1

        # write back models as lists
        heads["models"] = {lid: (None if np_models[lid] is None else np_models[lid].astype(np.float32).tolist())
                           for lid in label_ids}
        heads["label_ids"] = label_ids

        # If feedback lacks scores, compute them from updated prototypes
        for lid in label_ids:
            if lid not in score_from_fb and np_models[lid] is not None:
                score_from_fb[lid] = _cos01(x_norm, np_models[lid])

    # --- Threshold adaptation for all heads ---
    # Determine pred_pos set (fall back to recomputed scores if needed)
    if not pred_pos_from_fb:
        pred_pos_from_fb = {lid for lid, s in score_from_fb.items() if s >= float(thresholds.get(lid, 0.5))}

    # Sets
    agree = truth & pred_pos_from_fb
    fn = truth - pred_pos_from_fb      # missed positives
    fp = pred_pos_from_fb - truth      # predicted but not true

    # EMA updates
    for lid in agree:
        s = float(score_from_fb.get(lid, thresholds.get(lid, 0.5)))
        thr = float(thresholds.get(lid, 0.5))
        thresholds[lid] = float(max(0.05, min(0.95, (1 - alpha_threshold) * thr + alpha_threshold * (s * 0.9))))

    for lid in fn:
        # lower threshold towards observed score (encourage recall on this label)
        s = float(score_from_fb.get(lid, thresholds.get(lid, 0.5)))
        thr = float(thresholds.get(lid, 0.5))
        thresholds[lid] = float(max(0.05, min(0.95, (1 - alpha_threshold) * thr + alpha_threshold * (s * 0.85))))

    for lid in fp:
        # raise threshold towards observed score (reduce false positives)
        s = float(score_from_fb.get(lid, thresholds.get(lid, 0.5)))
        thr = float(thresholds.get(lid, 0.5))
        thresholds[lid] = float(max(0.05, min(0.95, (1 - alpha_threshold) * thr + alpha_threshold * max(thr, s))))

    heads["thresholds"] = thresholds

    # Persist heads + incremental state
    _save_heads_dict(heads_dir, heads)
    _save_inc_state(heads_dir, st)
    return heads

def apply_feedback_log(
    session_dir: Path,
    heads_dir: Path,
    limit: Optional[int] = None,
    alpha_threshold: float = 0.06,
) -> int:
    """
    Replays feedback.jsonl into heads incrementally.
    Useful for periodic, batched refinement when live updates are off.
    Returns number of feedback entries applied.
    """
    fb_path = session_dir / "feedback.jsonl"
    if not fb_path.exists():
        return 0

    # Read lines (optionally tail 'limit')
    lines = fb_path.read_text().splitlines()
    if limit is not None:
        lines = lines[-int(limit):]

    n_applied = 0
    for line in lines:
        try:
            obj = json.loads(line)
            ts = obj.get("timestamp")
            labels = (obj.get("labels") or {}).get("global") or []
            feedback = obj.get("feedback")
            updated = incremental_update_from_annotation(
                session_dir=session_dir,
                timestamp=ts,
                positive_labels=list(map(str, labels)),
                feedback=feedback,
                heads_dir=heads_dir,
                alpha_threshold=alpha_threshold,
            )
            if updated is not None:
                n_applied += 1
        except Exception:
            continue
    return n_applied
