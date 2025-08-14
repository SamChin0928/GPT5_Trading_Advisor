# backend/main.py
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import json
import threading
import shutil
from uuid import uuid4

from PIL import Image
from .utils import (
    DATA_DIR, ensure_session_dirs, decode_data_url_to_pil,
    save_zone_crops, make_mosaic_horiz,
    upsert_group_entry, read_json, add_label_to_vocab, load_vocab,
    upsert_annotation, ensure_simple_dir
)
from .model import load_model, predict_pil, CLASSES
from .training import train_model, TrainConfig

app = FastAPI(title="Chart Pattern Detector API")

# ---------- Static mount (/data) ----------
# Works whether DATA_DIR == ".../data" (contains "sessions") or ".../data/sessions"
DATA_PUBLIC_ROOT = DATA_DIR.parent if DATA_DIR.name == "sessions" else DATA_DIR
app.mount("/data", StaticFiles(directory=DATA_PUBLIC_ROOT, html=False), name="data")

# Also serve via /api/data so Vite proxying /api/* in dev fetches images, too.
app.mount("/api/data", StaticFiles(directory=DATA_PUBLIC_ROOT, html=False), name="data-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _dir_for_zones(session_id: str) -> Path:
    if session_id.startswith("zones-"):
        return ensure_simple_dir(session_id)
    return ensure_session_dirs(session_id)

# In-memory training job progress by session_id (keyed by anchor session_id)
TRAIN_PROGRESS: Dict[str, Dict] = {}

def _norm_ts(val) -> str:
    """Normalize timestamps to a canonical string of the integer value."""
    try:
        return str(int(str(val).strip()))
    except Exception:
        return str(val).strip()

# ========= Schemas =========

class IngestPayload(BaseModel):
    session_id: str
    timestamp: str                 # e.g., ms since epoch (stringified)
    zone_ids: List[int]
    images: List[str]              # data URLs matching zone_ids order
    roles: Optional[List[Optional[str]]] = None
    primary_id: Optional[int] = None
    tz: Optional[str] = None       # e.g. "Asia/Kuala_Lumpur"

class ZonesPayload(BaseModel):
    session_id: str
    zones: List[dict]              # normalized [0..1] coords

class LabelPayload(BaseModel):
    session_id: str
    mosaic_rel_path: str           # e.g., "mosaics/1723365281.jpg"
    label: str                     # bullish | bearish | neutral (legacy)

class TrainPayload(BaseModel):
    session_id: str
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 16
    # NEW: multi-session training controls
    include_all: bool = False            # train on all sessions under data/sessions (excluding internal folders)
    sessions: Optional[List[str]] = None # or provide an explicit list of session ids

class VocabCreate(BaseModel):
    name: str
    parent: Optional[str] = None

class AnnotationPayload(BaseModel):
    session_id: str
    timestamp: str
    global_labels: Optional[List[str]] = None
    by_role: Optional[Dict[str, List[str]]] = None
    notes: Optional[str] = None

# --- NEW: delete schemas ---
class DeleteGroupPayload(BaseModel):
    session_id: str
    timestamp: str  # folder key


class DeleteImagePayload(BaseModel):
    session_id: str
    timestamp: str
    # Provide either the full relative path (preferred) or just the filename
    path: Optional[str] = None
    filename: Optional[str] = None

# ========= Health =========

@app.get("/api/health")
def health():
    return {"ok": True}

# ========= Zones =========

@app.post("/api/zones/save")
def save_zones(payload: ZonesPayload):
    sdir = _dir_for_zones(payload.session_id)
    (sdir / "zones.json").write_text(json.dumps(payload.zones, indent=2))
    return {"ok": True}

@app.get("/api/zones/load")
def load_zones(session_id: str):
    sdir = _dir_for_zones(session_id)
    zf = sdir / "zones.json"
    return json.loads(zf.read_text()) if zf.exists() else []

# ========= Ingest =========

@app.post("/api/ingest")
def ingest(payload: IngestPayload):
    sdir = ensure_session_dirs(payload.session_id)
    imgs = [decode_data_url_to_pil(d) for d in payload.images]
    rel_paths = save_zone_crops(sdir, payload.timestamp, payload.zone_ids, imgs)
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

# ========= Consolidate =========

@app.post("/api/consolidate")
def consolidate(session_id: str):
    sdir = ensure_session_dirs(session_id)
    captures = sorted((sdir / "captures").glob("*/"))
    created = []
    for tdir in captures:
        parts = sorted(tdir.glob("zone_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
        if not parts:
            continue
        images = [Image.open(p).convert("RGB") for p in parts]
        mosaic = make_mosaic_horiz(images)
        out = (sdir / "mosaics" / f"{tdir.name}.jpg")
        mosaic.save(out, quality=92)
        created.append(str(out.relative_to(sdir)))

        # Ensure group_index has an entry for this timestamp
        zone_ids = [int(p.stem.split("_")[1]) for p in parts]
        rel_paths = [str(Path("captures") / tdir.name / p.name) for p in parts]
        upsert_group_entry(
            session_dir=sdir,
            timestamp=tdir.name,
            zone_ids=zone_ids,
            rel_paths=rel_paths,
            roles=None,
            primary_id=None,
            tz=None
        )
    return {"ok": True, "mosaics": created}

# ========= Mosaics =========

@app.get("/api/mosaics")
def list_mosaics(session_id: str):
    sdir = ensure_session_dirs(session_id)
    mos = sorted((sdir / "mosaics").glob("*.jpg"))
    return [str(p.relative_to(sdir)) for p in mos]

# ========= Legacy single-label =========

@app.post("/api/label")
def set_label(payload: LabelPayload):
    assert payload.label in CLASSES
    sdir = ensure_session_dirs(payload.session_id)
    labels_path = sdir / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    labels[payload.mosaic_rel_path] = payload.label
    labels_path.write_text(json.dumps(labels, indent=2))
    return {"ok": True}

# ========= Groups & Annotations helpers =========

def _sessions_root() -> Path:
    """Resolve folder that directly contains all session folders."""
    if (DATA_DIR / "sessions").exists():
        return DATA_DIR / "sessions"
    if DATA_DIR.name == "sessions":
        return DATA_DIR
    return DATA_DIR / "sessions"

# NEW: resolve a session path WITHOUT creating any subfolders
def _session_path(session_id: str) -> Path:
    return _sessions_root() / session_id

def _ts_sort_key(val):
    s = str(val)
    try:
        return (0, int(s))  # numeric first
    except Exception:
        return (1, s)

def _collect_from_index_or_captures_or_flat(sdir: Path) -> List[Dict]:
    """
    Return [{'timestamp','tz','zones'}] from group_index.json if present & non-empty,
    else scan:
      A) captures/<timestamp>/*.{png,jpg,jpeg,webp}
      B) <timestamp>/*.{png,jpg,jpeg,webp}
    """
    data = read_json(sdir / "group_index.json", None)
    if data and isinstance(data, dict):
        glist = data.get("groups")
        if isinstance(glist, list) and len(glist) > 0:
            return glist

    def scan(base: Path, sub: Optional[Path]):
        groups: List[Dict] = []
        root = base / sub if sub else base
        if not root.exists():
            return groups
        for tdir in sorted(root.iterdir()):
            if not tdir.is_dir():
                continue
            ts = tdir.name
            zones, idx = [], 1
            for img in sorted(tdir.iterdir()):
                if img.is_file() and img.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    rel = (sub / ts / img.name) if sub else (Path(ts) / img.name)
                    zones.append({
                        "id": idx,
                        "role": f"zone_{idx}",
                        "path": str(rel).replace("\\", "/"),
                    })
                    idx += 1
            if zones:
                groups.append({"timestamp": ts, "tz": "UTC", "zones": zones})
        return groups

    groups = scan(sdir, Path("captures"))
    if groups:
        return groups
    return scan(sdir, None)

# UPDATED: allow read-only mode and never scaffold for zones-*
def collect_groups_for_session(session_id: str, create_if_needed: bool = True) -> List[Dict]:
    """
    Per-session groups with annotation info:
      { session_id, timestamp, tz, zones, ann, labeled }
    If create_if_needed=False, do not create any subfolders.
    """
    if session_id.startswith("zones-"):
        sdir = _session_path(session_id)  # never scaffold zones-*
    else:
        sdir = ensure_session_dirs(session_id) if create_if_needed else _session_path(session_id)

    ann_map = read_json(sdir / "annotations.json", {})
    base_groups = _collect_from_index_or_captures_or_flat(sdir)

    out: List[Dict] = []
    for g in base_groups:
        ts = str(g.get("timestamp", ""))
        ann = ann_map.get(ts, {}) if isinstance(ann_map, dict) else {}
        labeled = bool(ann.get("global")) if isinstance(ann.get("global"), list) else False
        out.append({
            "session_id": session_id,
            "timestamp": ts,
            "tz": g.get("tz") or "UTC",
            "zones": g.get("zones") or [],
            "ann": ann,
            "labeled": labeled,
        })
    return sorted(out, key=lambda x: _ts_sort_key(x["timestamp"]))

# ========= Groups & Annotations =========

@app.get("/api/groups")
def get_groups(session_id: str):
    groups = collect_groups_for_session(session_id)  # default create_if_needed=True for real sessions
    return {"session_id": session_id, "groups": groups}

@app.get("/api/groups_all")
def get_groups_all(only_unlabeled: int = Query(0, ge=0, le=1)):
    sessions_root = _sessions_root()
    if not sessions_root.exists():
        return {"groups": []}

    all_groups: List[Dict] = []
    for sid_dir in sorted(sessions_root.iterdir()):
        if not sid_dir.is_dir():
            continue
        sid = sid_dir.name

        # skip internal/merged folders and any zones-* namespace
        if sid.startswith("_") or sid.startswith("zones-"):
            continue

        try:
            # read-only: don't scaffold while scanning
            all_groups.extend(collect_groups_for_session(sid, create_if_needed=False))
        except Exception:
            continue

    if only_unlabeled:
        all_groups = [g for g in all_groups if not g.get("labeled")]

    all_groups = sorted(all_groups, key=lambda g: _ts_sort_key(g.get("timestamp", "")))
    return {"groups": all_groups}

@app.get("/api/annotations")
def get_annotations(session_id: str):
    # avoid creating structure for zones-* namespaces
    if session_id.startswith("zones-"):
        sdir = _session_path(session_id)
    else:
        sdir = ensure_session_dirs(session_id)
    return read_json(sdir / "annotations.json", {})

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

# ========= Vocab =========

@app.get("/api/labels/vocab")
def get_vocab():
    return load_vocab()

@app.post("/api/labels/vocab")
def create_vocab(item: VocabCreate):
    label = add_label_to_vocab(item.name, item.parent)
    return {"ok": True, "label": label}

# ========= Deletion (folders & images) =========

def _safe_under(base: Path, p: Path) -> bool:
    """Ensure p is within base (avoid path traversal)."""
    try:
        return p.resolve().is_relative_to(base.resolve())
    except Exception:
        br = str(base.resolve())
        pr = str(p.resolve())
        return pr.startswith(br.rstrip("/") + "/")

@app.post("/api/group/delete")
def delete_group(payload: DeleteGroupPayload):
    sdir = ensure_session_dirs(payload.session_id)
    ts = _norm_ts(payload.timestamp)  # <-- normalize once

    cap_dir = sdir / "captures" / ts
    mos_jpg = sdir / "mosaics" / f"{ts}.jpg"

    removed = {"captures": False, "mosaic": False, "index_pruned": False, "annotation_pruned": False}

    if cap_dir.exists() and _safe_under(sdir, cap_dir):
        shutil.rmtree(cap_dir, ignore_errors=True)
        removed["captures"] = True

    if mos_jpg.exists() and _safe_under(sdir, mos_jpg):
        try:
            mos_jpg.unlink(missing_ok=True)
        except Exception:
            pass
        removed["mosaic"] = True

    gi_path = sdir / "group_index.json"
    gi = read_json(gi_path, None)
    if isinstance(gi, dict) and isinstance(gi.get("groups"), list):
        # normalize group timestamps on comparison
        new_groups = [g for g in gi["groups"] if _norm_ts(g.get("timestamp")) != ts]
        if len(new_groups) != len(gi["groups"]):
            gi["groups"] = new_groups
            gi_path.write_text(json.dumps(gi, indent=2))
            removed["index_pruned"] = True

    ann_path = sdir / "annotations.json"
    ann = read_json(ann_path, {})
    if isinstance(ann, dict) and ts in ann:
        ann.pop(ts, None)
        ann_path.write_text(json.dumps(ann, indent=2))
        removed["annotation_pruned"] = True

    return {"ok": True, "removed": removed}

@app.post("/api/group/delete_image")
def delete_group_image(payload: DeleteImagePayload):
    if not payload.path and not payload.filename:
        raise HTTPException(status_code=400, detail="Provide either 'path' or 'filename'.")

    sdir = ensure_session_dirs(payload.session_id)
    ts = _norm_ts(payload.timestamp)  # <-- normalize once

    # Resolve the on-disk file
    if payload.path:
        rel = Path(payload.path)
        if rel.parts[0] != "captures":
            rel = Path("captures") / rel
        if ts not in rel.parts:
            rel = Path("captures") / ts / rel.name
        target = sdir / rel
    else:
        target = sdir / "captures" / ts / payload.filename
    
    gi_path = sdir / "group_index.json"
    gi = read_json(gi_path, None)
    pruned_group = False
    group_empty = False

    if isinstance(gi, dict) and isinstance(gi.get("groups"), list):
        changed = False
        for g in gi["groups"]:
            if _norm_ts(g.get("timestamp")) == ts:
                zones = g.get("zones") or []
                before = len(zones)
                def norm(p): return str(p).replace("\\", "/")
                zones = [z for z in zones if norm(z.get("path", "")) != str(target.relative_to(sdir)).replace("\\", "/")]
                if len(zones) != before:
                    changed = True
                    if zones:
                        g["zones"] = zones
                    else:
                        pruned_group = True
                break
        if pruned_group:
            gi["groups"] = [g for g in gi["groups"] if _norm_ts(g.get("timestamp")) != ts]
            group_empty = True
            changed = True
        if changed:
            gi_path.write_text(json.dumps(gi, indent=2))

    tdir = sdir / "captures" / ts
    if tdir.exists() and not any(tdir.iterdir()):
        try:
            shutil.rmtree(tdir, ignore_errors=True)
        except Exception:
            pass
        group_empty = True

    if group_empty:
        ann_path = sdir / "annotations.json"
        ann = read_json(ann_path, {})
        if isinstance(ann, dict) and ts in ann:
            ann.pop(ts, None)
            ann_path.write_text(json.dumps(ann, indent=2))
        mos_jpg = sdir / "mosaics" / f"{ts}.jpg"
        if mos_jpg.exists():
            try:
                mos_jpg.unlink(missing_ok=True)
            except Exception:
                pass


# ========= Multi-session training support =========

def _build_merged_training_session(session_ids: List[str]) -> Path:
    """
    Create a synthetic session containing all labeled groups/images from the given sessions.
    Layout:
      data/sessions/_merged_train/<job-id>/
        captures/<SID__TS>/<files...>
        annotations.json
        group_index.json
        model/
    Returns the merged directory path.
    """
    sessions_root = _sessions_root()
    merged_root = sessions_root / "_merged_train"
    merged_root.mkdir(parents=True, exist_ok=True)

    job_id = f"job_{uuid4().hex[:8]}"
    merged_dir = merged_root / job_id
    (merged_dir / "captures").mkdir(parents=True, exist_ok=True)
    (merged_dir / "model").mkdir(parents=True, exist_ok=True)

    merged_ann: Dict[str, Dict] = {}
    merged_groups: List[Dict] = []

    for sid in session_ids:
        # only real sessions should be merged
        if sid.startswith("zones-"):
            continue

        sdir = ensure_session_dirs(sid)
        base_groups = _collect_from_index_or_captures_or_flat(sdir)
        ann_map = read_json(sdir / "annotations.json", {})

        for g in base_groups:
            ts = str(g.get("timestamp", ""))
            ann = ann_map.get(ts) or {}
            gl = ann.get("global") or []
            # Only include groups that have at least one folder/global label
            if not (isinstance(gl, list) and len(gl) > 0):
                continue

            new_ts = f"{sid}__{ts}"
            dest_tdir = merged_dir / "captures" / new_ts
            dest_tdir.mkdir(parents=True, exist_ok=True)

            zones_out = []
            for z in (g.get("zones") or []):
                src = (sdir / z["path"]).resolve()
                if not src.exists():
                    continue
                dst = dest_tdir / Path(z["path"]).name
                shutil.copy2(src, dst)
                zone_id = z.get("id")
                if zone_id is None:
                    zone_id = len(zones_out) + 1
                zones_out.append({
                    "id": zone_id,
                    "role": z.get("role") or f"zone_{zone_id}",
                    "path": str(Path("captures") / new_ts / dst.name).replace("\\", "/"),
                })

            if not zones_out:
                continue

            merged_groups.append({
                "timestamp": new_ts,
                "tz": g.get("tz") or "UTC",
                "zones": zones_out
            })

            merged_ann[new_ts] = {
                "global": list(gl),
                "by_role": ann.get("by_role") or {},
                "notes": ann.get("notes") or None
            }

    (merged_dir / "group_index.json").write_text(json.dumps({
        "session_id": f"_merged:{job_id}",
        "groups": merged_groups
    }, indent=2))

    (merged_dir / "annotations.json").write_text(json.dumps(merged_ann, indent=2))
    return merged_dir

# ========= Training =========

@app.post("/api/train")
def start_train(payload: TrainPayload):
    anchor_sid = payload.session_id  # where we’ll copy final weights for predict()

    # Resolve which sessions to include
    if payload.include_all:
        sroot = _sessions_root()
        if not sroot.exists():
            raise HTTPException(status_code=400, detail="No sessions directory found.")
        session_ids = [
            d.name for d in sorted(sroot.iterdir())
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith("zones-")
        ]
    elif payload.sessions:
        # de-dupe while keeping order; also drop any zones-* namespaces
        session_ids = [sid for sid in dict.fromkeys(payload.sessions) if not str(sid).startswith("zones-")]
    else:
        session_ids = [anchor_sid]

    if not session_ids:
        raise HTTPException(status_code=400, detail="No sessions found to train on.")

    # Choose directory we actually train on
    if len(session_ids) == 1 and session_ids[0] == anchor_sid:
        train_dir = ensure_session_dirs(anchor_sid)
    else:
        train_dir = _build_merged_training_session(session_ids)

    job_key = anchor_sid  # keep status keyed by anchor session_id for UI compatibility
    TRAIN_PROGRESS[job_key] = {
        "status": "running",
        "epoch": 0,
        "epochs": payload.epochs,
        "sessions": session_ids
    }

    def _runner():
        try:
            cfg = TrainConfig(
                session_dir=train_dir,
                epochs=payload.epochs,
                lr=payload.lr,
                batch_size=payload.batch_size
            )
            # Trainer auto-detects annotations.json vs legacy labels.json
            train_model(cfg, TRAIN_PROGRESS[job_key])

            # Copy learned weights back to the anchor session
            src_weights = train_dir / "model" / "weights.pth"
            if src_weights.exists():
                dst_weights = ensure_session_dirs(anchor_sid) / "model" / "weights.pth"
                dst_weights.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_weights, dst_weights)

        except Exception as e:
            TRAIN_PROGRESS[job_key].update({"status": "error", "error": str(e)})

    threading.Thread(target=_runner, daemon=True).start()
    return {"ok": True, "job": job_key, "sessions": session_ids}

@app.get("/api/train/status")
def train_status(session_id: str):
    return TRAIN_PROGRESS.get(session_id, {"status": "idle"})

# ========= Predict =========

@app.post("/api/predict")
def predict(session_id: str, file: UploadFile = File(...)):
    sdir = ensure_session_dirs(session_id)
    weights = sdir / "model" / "weights.pth"
    model, preprocess = load_model(weights if weights.exists() else None)
    img = Image.open(file.file).convert("RGB")

    # If no trained weights, return heuristic
    if not weights.exists():
        gray = img.convert("L")
        mean = sum(gray.getdata()) / (gray.width * gray.height)
        b = min(max((mean - 100) / 80, -1), 1)
        probs = {
            "bullish": float(max(0.0, b)),
            "bearish": float(max(0.0, -b)),
            "neutral": float(max(0.0, 1 - abs(b)))
        }
        s = sum(probs.values()) or 1.0
        probs = {k: v / s for k, v in probs.items()}
        label = max(probs, key=probs.get)
        return {"label": label, "probs": probs, "note": "heuristic (train a model for real predictions)"}

    out = predict_pil(model, preprocess, img)
    return out
