from __future__ import annotations
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
import json
import threading

from PIL import Image
from .utils import (
    DATA_DIR, ensure_session_dirs, decode_data_url_to_pil,
    save_zone_crops, make_mosaic_horiz
)
from .model import load_model, predict_pil, CLASSES
from .training import train_model, TrainConfig

app = FastAPI(title="Chart Pattern Detector API")

app.mount("/data", StaticFiles(directory=(DATA_DIR.parent)), name="data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In‑memory training job progress by session_id
TRAIN_PROGRESS: Dict[str, Dict] = {}


class IngestPayload(BaseModel):
    session_id: str
    timestamp: str                 # client supplied (e.g., ms since epoch)
    zone_ids: List[int]
    images: List[str]              # data URLs matching zone_ids order


class ZonesPayload(BaseModel):
    session_id: str
    zones: List[dict]              # stored as JSON (normalized [0..1] coords)


class LabelPayload(BaseModel):
    session_id: str
    mosaic_rel_path: str           # e.g., "mosaics/1723365281.jpg"
    label: str                     # bullish | bearish | neutral


class TrainPayload(BaseModel):
    session_id: str
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 16


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/zones/save")
def save_zones(payload: ZonesPayload):
    sdir = ensure_session_dirs(payload.session_id)
    (sdir/"zones.json").write_text(json.dumps(payload.zones, indent=2))
    return {"ok": True}


@app.get("/api/zones/load")
def load_zones(session_id: str):
    sdir = ensure_session_dirs(session_id)
    zf = sdir/"zones.json"
    return json.loads(zf.read_text()) if zf.exists() else []


@app.post("/api/ingest")
def ingest(payload: IngestPayload):
    sdir = ensure_session_dirs(payload.session_id)
    imgs = [decode_data_url_to_pil(d) for d in payload.images]
    save_zone_crops(sdir, payload.timestamp, payload.zone_ids, imgs)
    return {"ok": True}


@app.post("/api/consolidate")
def consolidate(session_id: str):
    sdir = ensure_session_dirs(session_id)
    captures = sorted((sdir/"captures").glob("*/"))
    created = []
    for tdir in captures:
        parts = sorted(tdir.glob("zone_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
        if not parts:
            continue
        images = [Image.open(p).convert("RGB") for p in parts]
        mosaic = make_mosaic_horiz(images)
        out = (sdir/"mosaics"/f"{tdir.name}.jpg")
        mosaic.save(out, quality=92)
        created.append(str(out.relative_to(sdir)))
    return {"ok": True, "mosaics": created}


@app.get("/api/mosaics")
def list_mosaics(session_id: str):
    sdir = ensure_session_dirs(session_id)
    mos = sorted((sdir/"mosaics").glob("*.jpg"))
    return [str(p.relative_to(sdir)) for p in mos]


@app.post("/api/label")
def set_label(payload: LabelPayload):
    assert payload.label in CLASSES
    sdir = ensure_session_dirs(payload.session_id)
    labels_path = sdir/"labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    labels[payload.mosaic_rel_path] = payload.label
    labels_path.write_text(json.dumps(labels, indent=2))
    return {"ok": True}


@app.post("/api/train")
def start_train(payload: TrainPayload):
    sdir = ensure_session_dirs(payload.session_id)
    job_key = payload.session_id
    TRAIN_PROGRESS[job_key] = {"status": "running", "epoch": 0, "epochs": payload.epochs}

    def _runner():
        try:
            cfg = TrainConfig(
                session_dir=sdir,
                epochs=payload.epochs,
                lr=payload.lr,
                batch_size=payload.batch_size
            )
            train_model(cfg, TRAIN_PROGRESS[job_key])
        except Exception as e:
            TRAIN_PROGRESS[job_key].update({"status": "error", "error": str(e)})

    threading.Thread(target=_runner, daemon=True).start()
    return {"ok": True, "job": job_key}


@app.get("/api/train/status")
def train_status(session_id: str):
    return TRAIN_PROGRESS.get(session_id, {"status": "idle"})


@app.post("/api/predict")
def predict(session_id: str, file: UploadFile = File(...)):
    sdir = ensure_session_dirs(session_id)
    weights = sdir/"model"/"weights.pth"
    model, preprocess = load_model(weights if weights.exists() else None)
    img = Image.open(file.file).convert("RGB")

    # If no trained weights, we still return a deterministic pseudo‑score
    if not weights.exists():
        # Simple mean brightness heuristic as a placeholder
        gray = img.convert("L")
        mean = sum(gray.getdata())/(gray.width*gray.height)
        # Map brightness to a soft distribution
        import math
        b = min(max((mean-100)/80, -1), 1)
        probs = {
            "bullish": float(max(0.0, b)),
            "bearish": float(max(0.0, -b)),
            "neutral": float(max(0.0, 1-abs(b)))
        }
        # normalize
        s = sum(probs.values()) or 1.0
        probs = {k: v/s for k,v in probs.items()}
        label = max(probs, key=probs.get)
        return {"label": label, "probs": probs, "note": "heuristic (train a model for real predictions)"}

    # Real model prediction
    out = predict_pil(model, preprocess, img)
    return out