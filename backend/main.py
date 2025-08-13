from __future__ import annotations
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import json
import threading

from PIL import Image
from .utils import (
    DATA_DIR, ensure_session_dirs, decode_data_url_to_pil,
    save_zone_crops, make_mosaic_horiz,
    # NEW helpers
    upsert_group_entry, read_json, add_label_to_vocab, load_vocab,
    upsert_annotation
)
from .model import load_model, predict_pil, CLASSES
from .training import train_model, TrainConfig

app = FastAPI(title="Chart Pattern Detector API")

app.mount("/data", StaticFiles(directory=(DATA_DIR.parent)), name="data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # keep for dev; narrow in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .utils import ensure_session_dirs, ensure_simple_dir  # ensure_simple_dir is new

def _dir_for_zones(session_id: str) -> Path:
    # Treat any zones-only namespace specially; adjust the prefix if you like
    if session_id.startswith("zones-"):
        return ensure_simple_dir(session_id)
    return ensure_session_dirs(session_id)

# In-memory training job progress by session_id
TRAIN_PROGRESS: Dict[str, Dict] = {}


# =========
# Schemas
# =========

class IngestPayload(BaseModel):
    session_id: str
    timestamp: str                 # client supplied (e.g., ms since epoch)
    zone_ids: List[int]
    images: List[str]              # data URLs matching zone_ids order
    # NEW (optional) – helps build group_index.json
    roles: Optional[List[Optional[str]]] = None
    primary_id: Optional[int] = None
    tz: Optional[str] = None       # e.g. "Asia/Kuala_Lumpur"


class ZonesPayload(BaseModel):
    session_id: str
    zones: List[dict]              # stored as JSON (normalized [0..1] coords)


class LabelPayload(BaseModel):
    session_id: str
    mosaic_rel_path: str           # e.g., "mosaics/1723365281.jpg"
    label: str                     # bullish | bearish | neutral  (legacy)


class TrainPayload(BaseModel):
    session_id: str
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 16


# NEW: vocab + annotations + groups

class VocabCreate(BaseModel):
    name: str
    parent: Optional[str] = None   # optional parent label id


class AnnotationPayload(BaseModel):
    session_id: str
    timestamp: str                 # group key
    global_labels: Optional[List[str]] = None
    by_role: Optional[Dict[str, List[str]]] = None
    notes: Optional[str] = None


# =========
# Health
# =========

@app.get("/api/health")
def health():
    return {"ok": True}


# =========
# Zones (unchanged)
# =========

@app.post("/api/zones/save")
def save_zones(payload: ZonesPayload):
    sdir = _dir_for_zones(payload.session_id)   # was ensure_session_dirs
    (sdir/"zones.json").write_text(json.dumps(payload.zones, indent=2))
    return {"ok": True}

@app.get("/api/zones/load")
def load_zones(session_id: str):
    sdir = _dir_for_zones(session_id)           # was ensure_session_dirs
    zf = sdir/"zones.json"
    return json.loads(zf.read_text()) if zf.exists() else []


# =========
# Ingest (extended to upsert group_index)
# =========

@app.post("/api/ingest")
def ingest(payload: IngestPayload):
    sdir = ensure_session_dirs(payload.session_id)
    # decode
    imgs = [decode_data_url_to_pil(d) for d in payload.images]
    # save crops, get relative paths
    rel_paths = save_zone_crops(sdir, payload.timestamp, payload.zone_ids, imgs)
    # build/update group_index.json (roles default to "zone_{id}" if not provided)
    upsert_group_entry(
        session_dir=sdir,
        timestamp=payload.timestamp,
        zone_ids=payload.zone_ids,
        rel_paths=rel_paths,
        roles=payload.roles,
        primary_id=payload.primary_id,
        tz=payload.tz
    )
    return {"ok": True}


# =========
# Consolidate (also ensures group_index entry exists)
# =========

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

        # also ensure group_index has an entry for this timestamp
        zone_ids = [int(p.stem.split("_")[1]) for p in parts]
        rel_paths = [str(Path("captures")/tdir.name/p.name) for p in parts]
        upsert_group_entry(
            session_dir=sdir,
            timestamp=tdir.name,
            zone_ids=zone_ids,
            rel_paths=rel_paths,
            roles=None,           # unknown; default to "zone_{id}"
            primary_id=None,
            tz=None
        )
    return {"ok": True, "mosaics": created}


# =========
# Mosaics (unchanged)
# =========

@app.get("/api/mosaics")
def list_mosaics(session_id: str):
    sdir = ensure_session_dirs(session_id)
    mos = sorted((sdir/"mosaics").glob("*.jpg"))
    return [str(p.relative_to(sdir)) for p in mos]


# =========
# Legacy single-label endpoint (kept for backward compatibility)
# =========

@app.post("/api/label")
def set_label(payload: LabelPayload):
    assert payload.label in CLASSES
    sdir = ensure_session_dirs(payload.session_id)
    labels_path = sdir/"labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    labels[payload.mosaic_rel_path] = payload.label
    labels_path.write_text(json.dumps(labels, indent=2))
    return {"ok": True}


# =========
# Training (unchanged pipeline for now)
# =========

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
            # train_model now auto-detects and runs either:
            #  - new embeddings+heads pipeline (if annotations exist), or
            #  - legacy TinyCNN (labels.json on mosaics)
            train_model(cfg, TRAIN_PROGRESS[job_key])
        except Exception as e:
            TRAIN_PROGRESS[job_key].update({"status": "error", "error": str(e)})

    threading.Thread(target=_runner, daemon=True).start()
    return {"ok": True, "job": job_key}


@app.get("/api/train/status")
def train_status(session_id: str):
    return TRAIN_PROGRESS.get(session_id, {"status": "idle"})


# =========
# Predict (unchanged)
# =========

@app.post("/api/predict")
def predict(session_id: str, file: UploadFile = File(...)):
    sdir = ensure_session_dirs(session_id)
    weights = sdir/"model"/"weights.pth"
    model, preprocess = load_model(weights if weights.exists() else None)
    img = Image.open(file.file).convert("RGB")

    # If no trained weights, return heuristic
    if not weights.exists():
        gray = img.convert("L")
        mean = sum(gray.getdata())/(gray.width*gray.height)
        import math
        b = min(max((mean-100)/80, -1), 1)
        probs = {
            "bullish": float(max(0.0, b)),
            "bearish": float(max(0.0, -b)),
            "neutral": float(max(0.0, 1-abs(b)))
        }
        s = sum(probs.values()) or 1.0
        probs = {k: v/s for k,v in probs.items()}
        label = max(probs, key=probs.get)
        return {"label": label, "probs": probs, "note": "heuristic (train a model for real predictions)"}

    out = predict_pil(model, preprocess, img)
    return out


# =========
# NEW: Vocab & Annotations & Groups APIs
# =========

@app.get("/api/labels/vocab")
def get_vocab():
    return load_vocab()

@app.post("/api/labels/vocab")
def create_vocab(item: VocabCreate):
    label = add_label_to_vocab(item.name, item.parent)
    return {"ok": True, "label": label}

@app.get("/api/groups")
def get_groups(session_id: str):
    sdir = ensure_session_dirs(session_id)
    data = read_json(sdir/"group_index.json", {"session_id": session_id, "groups": []})
    return data

@app.post("/api/annotate")
def annotate(payload: AnnotationPayload):
    sdir = ensure_session_dirs(payload.session_id)
    rec = upsert_annotation(
        session_dir=sdir,
        timestamp=payload.timestamp,
        global_labels=payload.global_labels,
        by_role=payload.by_role,
        notes=payload.notes
    )
    return {"ok": True, "annotation": rec}
