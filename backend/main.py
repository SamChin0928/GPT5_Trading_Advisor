# backend/main.py
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import threading
import shutil
from uuid import uuid4
import math
import numpy as np
import time
import os
import asyncio

from PIL import Image

from .training import (
    predict_group_probs,
    evaluate_session,
    _load_backbone,
    _group_vector,
    _rehydrate_booster,
    incremental_update_from_annotation,  # NEW
    apply_feedback_log,                  # NEW (optional endpoint below)
)
from .utils import (
    DATA_DIR, ensure_session_dirs, decode_data_url_to_pil,
    save_zone_crops, make_mosaic_horiz,
    upsert_group_entry, read_json, add_label_to_vocab, load_vocab,
    upsert_annotation, ensure_simple_dir,
    load_annotations, save_session_annotations
)
from .model import load_model, predict_pil, CLASSES

app = FastAPI(title="Chart Pattern Detector API")

# ---------- Static mount (/data) ----------
DATA_PUBLIC_ROOT = DATA_DIR.parent if DATA_DIR.name == "sessions" else DATA_DIR
app.mount("/data", StaticFiles(directory=DATA_PUBLIC_ROOT, html=False), name="data")
app.mount("/api/data", StaticFiles(directory=DATA_PUBLIC_ROOT, html=False), name="data-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _dir_for_zones(session_id: str) -> Path:
    if session_id.startswith("zones-"):
        return ensure_simple_dir(session_id)
    return ensure_session_dirs(session_id)

TRAIN_PROGRESS: Dict[str, Dict] = {}

def _norm_ts(val) -> str:
    try:
        return str(int(str(val).strip()))
    except Exception:
        return str(val).strip()

# --- incremental toggle (hard-coded ON) ---
def _incremental_enabled() -> bool:
    # Always-on incremental updates. (Change to `return False` to disable,
    # or make it read an env var if you want a kill switch later.)
    return True

# ========= SSE (Server-Sent Events) =========
# Minimal in-process pub/sub to notify frontends about changes.

_SUBSCRIBERS: List[asyncio.Queue] = []

def _broadcast(evt: dict):
    """Non-blocking fan-out to all connected SSE clients."""
    dead = []
    for q in list(_SUBSCRIBERS):
        try:
            q.put_nowait(evt)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            _SUBSCRIBERS.remove(q)
        except Exception:
            pass

@app.get("/api/events")
async def sse_events():
    """
    Event stream for realtime UI updates.
    Frontend can open: new EventSource('/api/events')
    """
    q: asyncio.Queue = asyncio.Queue()
    _SUBSCRIBERS.append(q)

    async def event_stream():
        try:
            # initial heartbeat so clients know stream is alive
            yield "event: hello\ndata: {}\n\n"
            while True:
                evt = await q.get()
                yield f"data: {json.dumps(evt)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _SUBSCRIBERS.remove(q)
            except Exception:
                pass

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

# ========= Schemas =========

class IngestPayload(BaseModel):
    session_id: str
    timestamp: str
    zone_ids: List[int]
    images: List[str]
    roles: Optional[List[Optional[str]]] = None
    primary_id: Optional[int] = None
    tz: Optional[str] = None

class ZonesPayload(BaseModel):
    session_id: str
    zones: List[dict]

class LabelPayload(BaseModel):
    session_id: str
    mosaic_rel_path: str
    label: str

class TrainPayload(BaseModel):
    session_id: str
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 16
    include_all: bool = False
    sessions: Optional[List[str]] = None

class VocabCreate(BaseModel):
    name: str
    parent: Optional[str] = None

class AnnotationPayload(BaseModel):
    session_id: str
    timestamp: str
    global_labels: Optional[List[str]] = None
    by_role: Optional[Dict[str, List[str]]] = None
    notes: Optional[str] = None
    # optional user-vs-model comparison payload from UI
    model_feedback: Optional[Dict[str, Any]] = None

class DeleteGroupPayload(BaseModel):
    session_id: str
    timestamp: str

class DeleteImagePayload(BaseModel):
    session_id: str
    timestamp: str
    path: Optional[str] = None
    filename: Optional[str] = None

class DeleteSessionPayload(BaseModel):
    session_id: str

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

# ========= Ingest (SCREENSHOT PATH FIXED HERE) =========

@app.post("/api/ingest")
def ingest(payload: IngestPayload):
    # use _dir_for_zones so zones-* sessions behave like before (no over-scaffolding)
    sdir = _dir_for_zones(payload.session_id)
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
    # NEW: notify listeners that a new capture exists
    _broadcast({"type": "capture", "session_id": payload.session_id, "timestamp": payload.timestamp, "tz": payload.tz})
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
    # Coarse notification that consolidation created/updated groups
    _broadcast({"type": "consolidated", "session_id": session_id, "mosaics": created})
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
    if (DATA_DIR / "sessions").exists():
        return DATA_DIR / "sessions"
    if DATA_DIR.name == "sessions":
        return DATA_DIR
    return DATA_DIR / "sessions"

def _global_model_dir() -> Path:
    return _sessions_root() / "_global" / "model"

def _global_model_exists() -> bool:
    gdir = _global_model_dir()
    return (gdir / "heads.pkl").exists() or (gdir / "weights.pth").exists()

def _session_path(session_id: str) -> Path:
    return _sessions_root() / session_id

def _ts_sort_key(val):
    s = str(val)
    try:
        return (0, int(s))
    except Exception:
        return (1, s)

def _collect_from_index_or_captures_or_flat(sdir: Path) -> List[Dict]:
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

def collect_groups_for_session(session_id: str, create_if_needed: bool = True) -> List[Dict]:
    if session_id.startswith("zones-"):
        sdir = _session_path(session_id)
    else:
        sdir = ensure_session_dirs(session_id) if create_if_needed else _session_path(session_id)

    ann_map = load_annotations(sdir)
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

# ---- helpers to attach lightweight predictions to group objects ----
def _model_meta() -> Dict[str, Any]:
    """
    Returns a small descriptor for the current global heads (if present).
    Non-failing and safe to call when model files are missing.
    """
    root = _global_model_dir()
    report = read_json(root / "train_report.json", {}) or {}
    mt_candidates = [p for p in [root / "heads.pkl", root / "weights.pth", root / "train_report.json"] if p.exists()]
    mtime = max((p.stat().st_mtime for p in mt_candidates), default=None)
    return {
        "name": report.get("mode") or "global-heads",
        # use mtime as a coarse version so it monotonically increases
        "version": int(mtime) if mtime else None
    }

def _group_pred_struct(session_id: str, ts: str) -> Dict[str, Any]:
    """
    Build the 'pred' object expected by the UI:
      { model:{name,version}, global:[{id,conf}, ...], by_role:{} }
    Uses existing global heads via _predict_group_with_global.
    Falls back to empty predictions when heads aren't available.
    """
    if not _global_model_exists():
        return {"model": None, "global": [], "by_role": {}}
    try:
        sdir = ensure_session_dirs(session_id)
        probs = _predict_group_with_global(sdir, str(ts))  # {label_id: prob}
        # sort by confidence desc and emit full list (UI may take top-N)
        global_list = [{"id": lid, "conf": float(p)} for lid, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)]
        return {"model": _model_meta(), "global": global_list, "by_role": {}}
    except Exception:
        # never fail the groups endpoint due to prediction errors
        return {"model": None, "global": [], "by_role": {}}

# ========= Groups & Annotations =========

@app.get("/api/groups")
def get_groups(session_id: str, include_pred: int = Query(1, ge=0, le=1)):
    groups = collect_groups_for_session(session_id)
    if include_pred:
        for g in groups:
            g["pred"] = _group_pred_struct(session_id, g.get("timestamp"))
    return {"session_id": session_id, "groups": groups}

@app.get("/api/groups_all")
def get_groups_all(only_unlabeled: int = Query(0, ge=0, le=1), include_pred: int = Query(1, ge=0, le=1)):
    sessions_root = _sessions_root()
    if not sessions_root.exists():
        return {"groups": []}

    all_groups: List[Dict] = []
    for sid_dir in sorted(sessions_root.iterdir()):
        if not sid_dir.is_dir():
            continue
        sid = sid_dir.name
        if sid.startswith("_") or sid.startswith("zones-"):
            continue
        try:
            all_groups.extend(collect_groups_for_session(sid, create_if_needed=False))
        except Exception:
            continue

    if only_unlabeled:
        all_groups = [g for g in all_groups if not g.get("labeled")]

    all_groups = sorted(all_groups, key=lambda g: _ts_sort_key(g.get("timestamp", "")))
    if include_pred:
        for g in all_groups:
            sid = g.get("session_id") or ""
            g["pred"] = _group_pred_struct(sid, g.get("timestamp"))
    return {"groups": all_groups}

@app.get("/api/annotations")
def get_annotations(session_id: str):
    sdir = _session_path(session_id)
    return load_annotations(sdir)

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
    # Non-intrusive: append model/user agreement info if provided.
    if payload.model_feedback is not None:
        try:
            fb = {
                "session_id": payload.session_id,
                "timestamp": str(payload.timestamp),
                "saved_at": int(time.time() * 1000),
                "feedback": payload.model_feedback,
                "labels": {"global": payload.global_labels, "by_role": payload.by_role},
            }
            (sdir / "feedback.jsonl").open("a", encoding="utf-8").write(json.dumps(fb) + "\n")
        except Exception:
            # Never fail the annotate call because feedback logging failed
            pass

    # Live incremental update (prototypes + thresholds) on save (always ON here)
    try:
        if _incremental_enabled():
            heads_dir = _global_model_dir()  # update the active global heads
            incremental_update_from_annotation(
                sdir,                               # session_dir
                payload.timestamp,                  # group timestamp
                payload.global_labels or [],        # authoritative labels
                payload.model_feedback,             # optional feedback blob
                heads_dir=heads_dir                 # target heads to update
            )
    except Exception:
        # Never fail annotate due to incremental update issues
        pass

    # Notify clients that annotations changed
    _broadcast({"type": "annotation_saved", "session_id": payload.session_id, "timestamp": str(payload.timestamp)})

    return {"ok": True, "annotation": rec}

# ========= Vocab =========

@app.get("/api/labels/vocab")
def get_vocab():
    return load_vocab()

@app.post("/api/labels/vocab")
def create_vocab(item: VocabCreate):
    label = add_label_to_vocab(item.name, item.parent)
    return {"ok": True, "label": label}

# ========= Deletion =========

def _safe_under(base: Path, p: Path) -> bool:
    try:
        return p.resolve().is_relative_to(base.resolve())
    except Exception:
        br = str(base.resolve()); pr = str(p.resolve())
        return pr.startswith(br.rstrip("/") + "/")

@app.post("/api/group/delete")
def delete_group(payload: DeleteGroupPayload):
    sdir = ensure_session_dirs(payload.session_id)
    ts = _norm_ts(payload.timestamp)

    cap_dir = sdir / "captures" / ts
    mos_jpg = sdir / "mosaics" / f"{ts}.jpg"

    removed = {"captures": False, "mosaic": False, "index_pruned": False, "annotation_pruned": False}

    if cap_dir.exists() and _safe_under(sdir, cap_dir):
        shutil.rmtree(cap_dir, ignore_errors=True)
        removed["captures"] = True

    if mos_jpg.exists() and _safe_under(sdir, mos_jpg):
        try: mos_jpg.unlink(missing_ok=True)
        except Exception: pass
        removed["mosaic"] = True

    gi_path = sdir / "group_index.json"
    gi = read_json(gi_path, None)
    if isinstance(gi, dict) and isinstance(gi.get("groups"), list):
        new_groups = [g for g in gi["groups"] if _norm_ts(g.get("timestamp")) != ts]
        if len(new_groups) != len(gi["groups"]):
            gi["groups"] = new_groups
            gi_path.write_text(json.dumps(gi, indent=2))
            removed["index_pruned"] = True

    ann = load_annotations(sdir)
    if isinstance(ann, dict) and ts in ann:
        ann.pop(ts, None)
        save_session_annotations(sdir.name, ann)
        removed["annotation_pruned"] = True

    # Notify clients that a group was deleted
    _broadcast({"type": "group_deleted", "session_id": payload.session_id, "timestamp": ts, "removed": removed})

    return {"ok": True, "removed": removed}

@app.post("/api/group/delete_image")
def delete_group_image(payload: DeleteImagePayload):
    if not payload.path and not payload.filename:
        raise HTTPException(status_code=400, detail="Provide either 'path' or 'filename'.")

    sdir = ensure_session_dirs(payload.session_id)
    ts = _norm_ts(payload.timestamp)

    if payload.path:
        rel = Path(payload.path)
        if rel.parts[0] != "captures":
            rel = Path("captures") / rel
        if ts not in rel.parts:
            rel = Path("captures") / ts / rel.name
        target = sdir / rel
    else:
        target = sdir / "captures" / ts / payload.filename

    if not _safe_under(sdir, target):
        raise HTTPException(status_code=400, detail="Invalid path.")

    file_removed = False
    try:
        if target.exists():
            target.unlink()
            file_removed = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

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
        try: shutil.rmtree(tdir, ignore_errors=True)
        except Exception: pass
        group_empty = True

    if group_empty:
        ann = load_annotations(sdir)
        if isinstance(ann, dict) and ts in ann:
            ann.pop(ts, None)
            save_session_annotations(sdir.name, ann)
        mos_jpg = sdir / "mosaics" / f"{ts}.jpg"
        if mos_jpg.exists():
            try: mos_jpg.unlink(missing_ok=True)
            except Exception: pass

    # Notify clients about image delete (and possible group removal)
    _broadcast({
        "type": "image_deleted",
        "session_id": payload.session_id,
        "timestamp": ts,
        "file_removed": file_removed,
        "group_empty": group_empty
    })

    return {"ok": True, "file_removed": file_removed, "group_empty": group_empty, "timestamp": ts}

# ========= Multi-session training support =========

def _build_merged_training_session(session_ids: List[str]) -> Path:
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
        if sid.startswith("zones-"):
            continue

        sdir = ensure_session_dirs(sid)
        base_groups = _collect_from_index_or_captures_or_flat(sdir)
        ann_map = load_annotations(sdir)

        for g in base_groups:
            ts = str(g.get("timestamp", ""))
            ann = ann_map.get(ts) or {}
            gl = ann.get("global") or []
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
                zone_id = z.get("id") or (len(zones_out) + 1)
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
    anchor_sid = payload.session_id

    if payload.include_all:
        sroot = _sessions_root()
        if not sroot.exists():
            raise HTTPException(status_code=400, detail="No sessions directory found.")
        session_ids = [
            d.name for d in sorted(sroot.iterdir())
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith("zones-")
        ]
    elif payload.sessions:
        session_ids = [sid for sid in dict.fromkeys(payload.sessions) if not str(sid).startswith("zones-")]
    else:
        session_ids = [anchor_sid]

    if not session_ids:
        raise HTTPException(status_code=400, detail="No sessions found to train on.")

    train_dir = _build_merged_training_session(session_ids)

    job_key = anchor_sid
    TRAIN_PROGRESS[job_key] = {
        "status": "running",
        "stage": "starting",
        "epoch": 0,
        "epochs": payload.epochs,
        "sessions": session_ids,
        "mode": None,
    }

    def _runner():
        try:
            from .training import train_model as _train_model, TrainConfig as _TrainConfig
            cfg = _TrainConfig(
                session_dir=train_dir,
                epochs=payload.epochs,
                lr=payload.lr,
                batch_size=payload.batch_size
            )
            _train_model(cfg, TRAIN_PROGRESS[job_key])

            groot = _sessions_root() / "_global" / "model"
            groot.mkdir(parents=True, exist_ok=True)

            src_heads = train_dir / "model" / "heads.pkl"
            if src_heads.exists():
                shutil.copy2(src_heads, groot / "heads.pkl")

            src_report = train_dir / "model" / "train_report.json"
            if src_report.exists():
                shutil.copy2(src_report, groot / "train_report.json")

            src_weights = train_dir / "model" / "weights.pth"
            if src_weights.exists():
                shutil.copy2(src_weights, groot / "weights.pth")

            TRAIN_PROGRESS[job_key].update({"status": "done"})

        except Exception as e:
            TRAIN_PROGRESS[job_key].update({"status": "error", "error": str(e)})
        finally:
            try:
                p = train_dir.resolve()
                if p.parent.name.startswith("_merged_train"):
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

    threading.Thread(target=_runner, daemon=True).start()
    return {"ok": True, "sessions": session_ids, "job_dir": str(train_dir)}

@app.get("/api/train/status")
def train_status(session_id: str):
    state = dict(TRAIN_PROGRESS.get(session_id, {"status": "idle"}))
    state["global_model"] = _global_model_exists()
    return state

# ---- MODEL INFO (for LivePrediction UI) ----
@app.get("/api/model/info")
def model_info():
    root = _global_model_dir()
    has_heads = (root / "heads.pkl").exists()
    has_weights = (root / "weights.pth").exists()
    exists = has_heads or has_weights
    report = read_json(root / "train_report.json", {})
    mt_candidates = [p for p in [root / "heads.pkl", root / "weights.pth", root / "train_report.json"] if p.exists()]
    mtime = max((p.stat().st_mtime for p in mt_candidates), default=None)
    trained_at_human = None
    if mtime:
        trained_at_human = time.strftime("%d %b %Y %H:%M", time.localtime(mtime))
    return {
        "exists": bool(exists),
        "has_heads": bool(has_heads),
        "has_weights": bool(has_weights),
        "trained_at": mtime,
        "trained_at_human": trained_at_human,
        "labels": report.get("labels") or [],
        "metrics": report.get("metrics") or {},
        "mode": report.get("mode"),
        "n_groups": report.get("n_groups"),
    }

@app.post("/api/model/reset")
def model_reset():
    root = _sessions_root() / "_global"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    return {"ok": True}

# ========= Predict =========

@app.post("/api/predict")
def predict(session_id: str, file: UploadFile = File(...)):
    gweights = _global_model_dir() / "weights.pth"
    sdir = ensure_session_dirs(session_id)
    sweights = sdir / "model" / "weights.pth"
    weights = gweights if gweights.exists() else (sweights if sweights.exists() else None)
    model, preprocess = load_model(weights if weights is not None else None)
    img = Image.open(file.file).convert("RGB")

    if weights is None:
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
        return {"label": label, "probs": probs, "note": "heuristic (build a model for real predictions)"}

    out = predict_pil(model, preprocess, img)
    return out

@app.post("/api/session/delete")
def delete_session(payload: DeleteSessionPayload):
    sroot = _sessions_root()
    sdir = _session_path(payload.session_id)
    if not sdir.exists():
        return {"ok": True, "removed": False, "note": "session not found"}
    if not _safe_under(sroot, sdir):
        raise HTTPException(status_code=400, detail="Invalid session path")
    shutil.rmtree(sdir, ignore_errors=True)
    save_session_annotations(payload.session_id, {})
    return {"ok": True, "removed": True, "session_id": payload.session_id}

# ========= Group prediction (GLOBAL heads) =========

def _get_group_by_ts_local(session_dir: Path, ts: str) -> Optional[Dict]:
    gi = read_json(session_dir / "group_index.json", {"groups": []})
    for g in gi.get("groups", []):
        if str(g.get("timestamp")) == str(ts):
            return g
    return None

def _predict_group_with_global(session_dir: Path, timestamp: str) -> Dict[str, float]:
    gdir = _global_model_dir()
    heads_path = gdir / "heads.pkl"
    if not heads_path.exists():
        raise HTTPException(status_code=400, detail="No global model found (heads.pkl). Build the model first.")
    import pickle
    with open(heads_path, "rb") as f:
        heads = pickle.load(f)

    g = _get_group_by_ts_local(session_dir, timestamp)
    if not g:
        raise HTTPException(status_code=400, detail=f"Group {timestamp} not found in session.")

    backbone, tf, feat_dim, _ = _load_backbone()
    x = _group_vector(session_dir, g, heads["roles"], feat_dim, backbone, tf).reshape(1, -1)

    scores: Dict[str, float] = {}
    if heads["type"] == "lightgbm_dump":
        for lid in heads["label_ids"]:
            kind, payload = heads["models"][lid]
            if kind == "prior":
                p = float(payload)
            else:
                booster = _rehydrate_booster(payload)
                p = float(booster.predict(x)[0])
            scores[lid] = p
    else:
        def cos(a, b):
            na = np.linalg.norm(a) + 1e-9
            nb = np.linalg.norm(b) + 1e-9
            return float((a @ b) / (na * nb))
        for lid in heads["label_ids"]:
            c = heads["models"][lid]
            if c is None:
                scores[lid] = 0.0
            else:
                sim = cos(x[0], np.asarray(c, dtype=np.float32))
                scores[lid] = float((sim + 1.0) / 2.0)
    return scores

@app.get("/api/predict_group")
def predict_group(session_id: str, timestamp: Optional[str] = None):
    sdir = ensure_session_dirs(session_id)
    if timestamp is None:
        gi = (sdir / "group_index.json")
        gidx = read_json(gi, {"groups": []})
        if not gidx.get("groups"):
            raise HTTPException(status_code=400, detail="No groups available.")
        latest = sorted(gidx["groups"], key=lambda g: int(str(g.get('timestamp', '0'))))[-1]
        timestamp = str(latest.get("timestamp"))

    # compute probs via global heads
    probs = _predict_group_with_global(sdir, str(timestamp))

    # load thresholds and decide top label
    import pickle
    with open(_global_model_dir() / "heads.pkl", "rb") as f:
        hp = pickle.load(f)
    thresholds = hp.get("thresholds", {})
    preds = [lid for lid, p in probs.items() if p >= thresholds.get(lid, 0.5)]
    top = max(probs, key=probs.get) if probs else None

    # ---- persist the model suggestion alongside annotations (non-destructive) ----
    ann_map = load_annotations(sdir)     # central slice for this session
    rec = dict(ann_map.get(str(timestamp), {}))
    rec["model"] = {
        "top": top,
        "probs": probs,
        "ts": int(time.time() * 1000),
    }
    ann_map[str(timestamp)] = rec
    save_session_annotations(sdir.name, ann_map)

    return {
        "session_id": session_id,
        "timestamp": str(timestamp),
        "top": top,
        "preds": preds,
        "probs": probs,
        "thresholds": thresholds
    }

# ========= Evaluation (GLOBAL heads) =========

@app.get("/api/eval")
def eval_heads(session_id: str):
    sdir = ensure_session_dirs(session_id)
    if not (_global_model_dir() / "heads.pkl").exists():
        raise HTTPException(status_code=400, detail="No global model found. Build the model first.")

    ann = load_annotations(sdir)
    gi = read_json(sdir / "group_index.json", {"groups": []})
    ts_list = [str(g["timestamp"]) for g in gi.get("groups", []) if str(g.get("timestamp")) in ann]
    if not ts_list:
        return {"session_id": session_id, "n_eval": 0, "per_label": {}, "overall": {}}

    import pickle
    with open(_global_model_dir() / "heads.pkl", "rb") as f:
        heads = pickle.load(f)
    thresholds = heads.get("thresholds", {})
    label_ids = heads.get("label_ids", [])

    per_label = {lid: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lid in label_ids}
    preds_by_ts: Dict[str, Dict[str, float]] = {}

    for ts in ts_list:
        probs = _predict_group_with_global(sdir, ts)
        preds_by_ts[ts] = probs
        true = set(ann[ts].get("global") or [])
        for lid in label_ids:
            p = probs.get(lid, 0.0)
            th = thresholds.get(lid, 0.5)
            pred = (p >= th)
            is_true = (lid in true)
            if is_true:
                per_label[lid]["support"] += 1
            if pred and is_true:
                per_label[lid]["tp"] += 1
            elif pred and not is_true:
                per_label[lid]["fp"] += 1
            elif (not pred) and is_true:
                per_label[lid]["fn"] += 1

    per_label_metrics = {}
    micro_tp = micro_fp = micro_fn = 0
    for lid, d in per_label.items():
        tp, fp, fn, sup = d["tp"], d["fp"], d["fn"], d["support"]
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        per_label_metrics[lid] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(sup)}
        micro_tp += tp; micro_fp += fp; micro_fn += fn
    micro_prec = micro_tp / (micro_tp + micro_fp + 1e-9)
    micro_rec  = micro_tp / (micro_tp + micro_fn + 1e-9)
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + 1e-9)

    return {
        "session_id": session_id,
        "n_eval": len(ts_list),
        "per_label": per_label_metrics,
        "overall": {"micro_precision": float(micro_prec), "micro_recall": float(micro_rec), "micro_f1": float(micro_f1)},
        "preds": preds_by_ts,
        "thresholds": thresholds,
        "labels": label_ids
    }

# ========= (Optional) Apply feedback.jsonl in batch =========

@app.post("/api/train/apply_feedback")
def train_apply_feedback(session_id: str, limit: Optional[int] = None):
    """
    Replays feedback.jsonl for `session_id` into the active GLOBAL heads incrementally.
    Safe no-op if feedback.jsonl doesn't exist or heads are missing.
    """
    sdir = ensure_session_dirs(session_id)
    heads_dir = _global_model_dir()
    try:
        n = apply_feedback_log(sdir, heads_dir, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "applied": int(n)}
