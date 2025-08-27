# backend/main.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from pathlib import Path
from PIL import Image
import time, shutil

from .utils import (
    DATA_DIR,
    # write-path helpers (mkdir side-effects)
    ensure_process_dirs, ensure_session_dirs, group_dir, captures_dir, model_dir,
    # path-only helpers (NO mkdir)
    process_root_path, sessions_dir_path, session_root_path, captures_dir_path,
    # images & zones & vocab & annotations
    decode_data_url_to_pil, save_zone_image,
    save_zones_global, load_zones_global,
    read_vocab, write_vocab,
    load_annotations, save_annotations_for_session,
    # listing & deletion (non-creating)
    list_groups, delete_group, delete_group_image,
    delete_session_dir, delete_process_dir,
)

# ──────────────────────────────────────────────────────────────────────────────
# App + Static
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="GPT5 Trading CV API")

# Serve files so frontend can use /data/<process>/sessions/<session>/...
app.mount("/data", StaticFiles(directory=str(DATA_DIR), html=False), name="data")
# Dev/prod compatibility for clients using /api/data/...
app.mount("/api/data", StaticFiles(directory=str(DATA_DIR), html=False), name="api-data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_process_id(
    x_process_id: Optional[str] = Header(None),
    q_process_id: Optional[str] = Query(None),
    body_process_id: Optional[str] = None,
) -> str:
    """
    Resolve current process id from header 'x-process-id', query 'process_id',
    or an explicit body-provided override (for callers that only send JSON).
    IMPORTANT: no implicit mkdir here. We only create directories in write paths.
    """
    return (body_process_id or x_process_id or q_process_id or "default")

def _latest_timestamp_dir(cap_dir: Path) -> Optional[Path]:
    if not cap_dir.exists():
        return None
    dirs = [p for p in cap_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.name)
    return dirs[-1]

def _dummy_group_pred_from_dir(ts_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Lightweight deterministic prediction for a folder based on *.jpg count.
    Shape matches what Labeler.jsx expects for g.pred.
    """
    if ts_dir is None or (not ts_dir.exists()):
        return None
    n = len([p for p in ts_dir.glob("*.jpg")])
    probs = {"bullish": 0.5 if n % 2 else 0.2, "bearish": 0.3, "neutral": 0.2 if n % 2 else 0.5}
    s = sum(probs.values()) or 1.0
    probs = {k: v / s for k, v in probs.items()}
    top = max(probs, key=lambda k: probs[k])
    global_list = sorted(
        [{"id": k, "conf": float(v)} for k, v in probs.items()],
        key=lambda x: x["conf"],
        reverse=True
    )
    return {"top": top, "probs": probs, "global": global_list}

def _get_or_make_group_pred(pid: str, session_id: str, ts: str) -> Optional[Dict[str, Any]]:
    """
    Read folder prediction from annotations if present; otherwise compute a dummy one and persist it.
    Stored under annotations[ts].model_pred.
    """
    sdir = session_root_path(pid, session_id)  # path-only (no mkdir)
    sess_map = load_annotations(sdir) or {}
    rec = sess_map.get(str(ts)) or {}

    model_pred = rec.get("model_pred")
    if isinstance(model_pred, dict) and "top" in model_pred and "probs" in model_pred:
        if "global" not in model_pred and isinstance(model_pred.get("probs"), dict):
            gp = model_pred["probs"]
            model_pred["global"] = sorted(
                [{"id": k, "conf": float(v)} for k, v in gp.items()],
                key=lambda x: x["conf"],
                reverse=True
            )
        return model_pred

    ts_dir = captures_dir_path(pid, session_id) / str(ts)  # path-only
    pred = _dummy_group_pred_from_dir(ts_dir)
    if pred is None:
        return None
    rec["model_pred"] = {**pred, "at": int(time.time())}
    sess_map[str(ts)] = rec
    save_annotations_for_session(pid, session_id, sess_map)
    return pred

def _get_labeled_flag(pid: str, session_id: str, ts: str) -> bool:
    sdir = session_root_path(pid, session_id)  # path-only
    sess_map = load_annotations(sdir) or {}
    entry = sess_map.get(str(ts)) or {}
    gl = entry.get("global") or []
    return bool(gl and isinstance(gl, list) and len(gl) > 0)

# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────
class IngestPayload(BaseModel):
    session_id: str
    timestamp: str
    zone_ids: List[int]
    images: List[str]           # data URLs, aligned with zone_ids
    roles: Optional[List[Optional[str]]] = None
    primary_id: Optional[int] = None
    tz: Optional[str] = None
    # NEW: allow callers to send process_id in JSON
    process_id: Optional[str] = None

class AnnotatePayload(BaseModel):
    session_id: str
    timestamp: str | int
    global_labels: Optional[List[str]] = None
    by_role: Optional[Dict[str, List[str]]] = None
    notes: Optional[str] = None
    process_id: Optional[str] = None

class VocabCreate(BaseModel):
    name: str

class DeleteGroupPayload(BaseModel):
    session_id: str
    timestamp: str | int
    process_id: Optional[str] = None

class DeleteImagePayload(BaseModel):
    session_id: str
    timestamp: str | int
    rel: Optional[str] = None  # "captures/<ts>/<file>"
    path: Optional[str] = None
    filename: Optional[str] = None
    process_id: Optional[str] = None

class SessionDeletePayload(BaseModel):
    session_id: str
    process_id: Optional[str] = None

class ProcessDeletePayload(BaseModel):
    process_id: str

# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health(x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None)):
    pid = _get_process_id(x_process_id, process_id)
    return {"ok": True, "process": pid}

# ──────────────────────────────────────────────────────────────────────────────
# Zones (global layout for a process)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/zones/load")
def zones_load(
    zones_id: Optional[str] = Query(None), session_id: Optional[str] = Query(None),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None)
):
    pid = _get_process_id(x_process_id, process_id)
    if not (zones_id or session_id):
        raise HTTPException(status_code=422, detail="Missing zones_id or session_id")
    return load_zones_global(pid)

@app.post("/api/zones/save")
def zones_save(
    payload: Dict[str, Any],
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None)
):
    # Allow body override of process_id for clients that only send JSON
    pid = _get_process_id(x_process_id, process_id, payload.get("process_id"))
    zones = payload.get("zones")
    if zones is None:
        raise HTTPException(status_code=422, detail="Missing 'zones' in body")
    save_zones_global(pid, zones)
    return {"ok": True}

# ──────────────────────────────────────────────────────────────────────────────
# Ingest (save zone crops under captures/<timestamp>)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/ingest")
def ingest(
    payload: IngestPayload,
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id, payload.process_id)
    session_id = payload.session_id
    ts = str(payload.timestamp)

    # Target directory (mkdir for write)
    gdir = group_dir(pid, session_id, ts)

    # Save each image as zone_<id>.jpg, using the ids provided
    if len(payload.zone_ids) != len(payload.images):
        raise HTTPException(status_code=400, detail="zone_ids and images length mismatch")

    saved = []
    for zid, data_url in zip(payload.zone_ids, payload.images):
        try:
            img = decode_data_url_to_pil(data_url)
            out = gdir / f"zone_{int(zid)}.jpg"
            save_zone_image(img, out, quality=92)
            saved.append(out.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed saving zone {zid}: {e}")

    # Compute + persist model prediction so folder cards can show it immediately
    try:
        _ = _get_or_make_group_pred(pid, session_id, ts)
    except Exception:
        pass

    return {
        "ok": True,
        "session_id": session_id,
        "timestamp": ts,
        "saved": saved,
        "rel_dir": f"captures/{ts}",
    }

# ──────────────────────────────────────────────────────────────────────────────
# Groups (per session) & Groups All (across sessions)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/groups")
def groups(
    session_id: str = Query(...),
    include_pred: int = Query(0, ge=0, le=1),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    rows = list_groups(pid, session_id)  # read-only (no mkdir)
    out: List[Dict[str, Any]] = []
    for r in rows:
        ts = r["timestamp"]
        zones = [{"id": i, "role": f"zone_{i}", "path": rel} for i, rel in enumerate(r["images"])]
        rec: Dict[str, Any] = {
            "timestamp": ts,
            "session_id": session_id,
            "zones": zones,
            "labeled": _get_labeled_flag(pid, session_id, ts),
        }
        if include_pred:
            pred = _get_or_make_group_pred(pid, session_id, ts)
            if pred:
                rec["pred"] = pred
        out.append(rec)
    return {"session_id": session_id, "groups": out}

@app.get("/api/groups_all")
def groups_all(
    only_unlabeled: int = Query(0, ge=0, le=1),
    unlabeled: Optional[int] = Query(None),   # legacy alias -> maps to only_unlabeled
    include_pred: int = Query(0, ge=0, le=1),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    if unlabeled is not None:
        only_unlabeled = 1 if int(unlabeled) else 0

    all_groups: List[Dict[str, Any]] = []
    sess_root = sessions_dir_path(pid)  # path-only (no mkdir)
    if sess_root.exists():
        for sdir in sorted([p for p in sess_root.iterdir() if p.is_dir()]):
            sid = sdir.name
            for r in list_groups(pid, sid):  # read-only
                ts = r["timestamp"]
                zones = [{"id": i, "role": f"zone_{i}", "path": rel} for i, rel in enumerate(r["images"])]
                item: Dict[str, Any] = {
                    "timestamp": ts,
                    "session_id": sid,
                    "zones": zones,
                    "labeled": _get_labeled_flag(pid, sid, ts),
                }
                if include_pred:
                    pred = _get_or_make_group_pred(pid, sid, ts)
                    if pred:
                        item["pred"] = pred
                all_groups.append(item)

    if only_unlabeled:
        all_groups = [g for g in all_groups if not g.get("labeled")]

    return {"groups": all_groups}

# ──────────────────────────────────────────────────────────────────────────────
# Annotations
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/annotations")
def annotations_get(
    session_id: str = Query(...),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    sdir = session_root_path(pid, session_id)  # path-only (no mkdir)
    return load_annotations(sdir)

@app.post("/api/annotations/annotate")
def annotations_annotate(
    payload: AnnotatePayload,
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id, payload.process_id)
    sid = payload.session_id
    sdir = session_root_path(pid, sid)  # path-only read
    sess_map = load_annotations(sdir) or {}

    ts_key = str(payload.timestamp)
    entry = sess_map.get(ts_key, {}) if isinstance(sess_map.get(ts_key), dict) else {}
    if payload.global_labels is not None:
        entry["global"] = list(payload.global_labels)
    if payload.by_role is not None:
        entry["by_role"] = payload.by_role
    if payload.notes is not None:
        entry["notes"] = payload.notes
    # Keep/seed model_pred
    if "model_pred" not in entry:
        try:
            pred = _get_or_make_group_pred(pid, sid, ts_key)
            if pred:
                entry["model_pred"] = {**pred, "at": int(time.time())}
        except Exception:
            pass

    sess_map[ts_key] = entry
    save_annotations_for_session(pid, sid, sess_map)
    return {"ok": True, "timestamp": ts_key, "session_id": sid, "annotation": entry}

# ──────────────────────────────────────────────────────────────────────────────
# Vocab (objects with id/name to match Labeler.jsx expectations)
# ──────────────────────────────────────────────────────────────────────────────
def _vocab_objects(lst: List[str]) -> List[Dict[str, str]]:
    return [{"id": s, "name": s} for s in lst if isinstance(s, str) and s.strip()]

@app.get("/api/vocab")
def vocab_get(
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    lst = read_vocab(pid)  # list[str]
    return {"labels": _vocab_objects(lst)}

@app.post("/api/vocab")
def vocab_post(
    item: VocabCreate,
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    vocab = read_vocab(pid)  # list[str]
    name = (item.name or "").strip()
    created = False
    if name and name not in vocab:
        vocab.append(name)
        write_vocab(pid, vocab)
        created = True
    label_obj = {"id": name, "name": name} if name else None
    return {"ok": True, "created": created, "labels": _vocab_objects(vocab), "label": label_obj}

# ──────────────────────────────────────────────────────────────────────────────
# Delete group / image / session / process
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/group/delete")
def group_delete(
    payload: DeleteGroupPayload,
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id, payload.process_id)
    ok = delete_group(pid, payload.session_id, str(payload.timestamp))
    return {"ok": ok}

@app.post("/api/group/delete_image")
def group_delete_image(
    payload: DeleteImagePayload,
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id, payload.process_id)
    rel = payload.rel or payload.path
    if not rel and not payload.filename:
        raise HTTPException(status_code=400, detail="Provide 'rel' (or 'path') or 'filename'")
    if rel:
        ok = delete_group_image(pid, payload.session_id, str(payload.timestamp), rel)
    else:
        rel = f"captures/{payload.timestamp}/{payload.filename}"
        ok = delete_group_image(pid, payload.session_id, str(payload.timestamp), rel)
    return {"ok": ok}

@app.post("/api/session/delete")
def session_delete(
    payload: SessionDeletePayload,
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None)
):
    pid = _get_process_id(x_process_id, process_id, payload.process_id)
    ok = delete_session_dir(pid, payload.session_id)
    return {"ok": ok}

@app.post("/api/process/delete")
def process_delete(
    payload: ProcessDeletePayload,
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None)
):
    # Always trust explicit body id; fallback to header/query only if body missing
    pid = _get_process_id(x_process_id, process_id, payload.process_id or None)
    ok = delete_process_dir(pid)
    return {"ok": ok}

# ──────────────────────────────────────────────────────────────────────────────
# Predict (stubbed – deterministic)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict(
    session_id: str = Query(...),
    file: UploadFile = File(...),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    _ = _get_process_id(x_process_id, process_id)  # read pid but no mkdir
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    gray = img.convert("L")
    mean = sum(gray.getdata()) / (gray.width * gray.height)
    b = max(min((mean - 100.0) / 80.0, 1.0), -1.0)
    probs = {"bullish": max(0.0, b), "bearish": max(0.0, -b), "neutral": max(0.0, 1.0 - abs(b))}
    s = sum(probs.values()) or 1.0
    probs = {k: v / s for k, v in probs.items()}
    label = max(probs, key=probs.get)
    return {"label": label, "probs": probs}

@app.get("/api/predict_group")
def predict_group(
    session_id: str = Query(...),
    timestamp: Optional[str] = Query(None),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    cap = captures_dir_path(pid, session_id)  # path-only (no mkdir)
    ts_dir = cap / str(timestamp) if timestamp else _latest_timestamp_dir(cap)
    if ts_dir is None or not ts_dir.exists():
        raise HTTPException(status_code=400, detail="No groups found.")
    pred = _dummy_group_pred_from_dir(ts_dir) or {}
    return {"timestamp": ts_dir.name, **pred}

# Legacy alias
@app.get("/api/predictGroup")
def predict_group_alias(
    session_id: str = Query(...),
    timestamp: Optional[str] = Query(None),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    return predict_group(session_id=session_id, timestamp=timestamp, x_process_id=x_process_id, process_id=process_id)

# ──────────────────────────────────────────────────────────────────────────────
# Training stubs (now ensure model dir exists + write a tiny metadata file)
# ──────────────────────────────────────────────────────────────────────────────
_TRAIN_STATE: Dict[str, Dict[str, Any]] = {}

@app.post("/api/train")
def train_start(
    session_id: str = Query(...),
    include_all: bool = Query(False),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    # ensure model dir to make it visible on disk
    mdir = model_dir(pid)
    try:
        (mdir / "metadata.json").write_text(
            json.dumps({"started_at": int(time.time()), "include_all": bool(include_all)}, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass
    _TRAIN_STATE[pid + ":" + session_id] = {
        "status": "running",
        "include_all": bool(include_all),
        "started_at": int(time.time())
    }
    return {"ok": True}

@app.get("/api/train/status")
def train_status(
    session_id: str = Query(...),
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    key = pid + ":" + session_id
    state = _TRAIN_STATE.get(key, {"status": "idle"})
    return state

@app.get("/api/model/info")
def model_info(
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    mdir = model_dir(pid)  # ensure exists
    exists = mdir.exists()
    return {
        "exists": bool(exists),
        "has_heads": False,
        "has_weights": False,
        "trained_at": None,
        "trained_at_human": None,
        "labels": [],
        "metrics": {},
        "mode": None,
        "n_groups": None,
    }

@app.post("/api/model/reset")
def model_reset(
    x_process_id: Optional[str] = Header(None), process_id: Optional[str] = Query(None),
):
    pid = _get_process_id(x_process_id, process_id)
    mdir = model_dir(pid)  # mkdir for write
    try:
        if mdir.exists():
            shutil.rmtree(mdir, ignore_errors=True)
        mdir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return {"ok": True}
