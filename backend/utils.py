# backend/utils.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import json, re, base64, io, shutil
from PIL import Image

# -----------------------
# Data root
# -----------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = (ROOT_DIR / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)  # create data/ root only

# -----------------------
# Path-only helpers (NO mkdir side-effects)
# -----------------------
def process_root_path(process_id: str) -> Path:
    return DATA_DIR / str(process_id)

def zones_dir_path(process_id: str) -> Path:
    return process_root_path(process_id) / "zones"

def model_dir_path(process_id: str) -> Path:
    return process_root_path(process_id) / "model"

def sessions_dir_path(process_id: str) -> Path:
    return process_root_path(process_id) / "sessions"

def session_root_path(process_id: str, session_id: str) -> Path:
    return sessions_dir_path(process_id) / str(session_id)

def captures_dir_path(process_id: str, session_id: str) -> Path:
    return session_root_path(process_id, session_id) / "captures"

def group_dir_path(process_id: str, session_id: str, timestamp: str) -> Path:
    return captures_dir_path(process_id, session_id) / str(timestamp)

# -----------------------
# Creating helpers (mkdir side-effects) — use for writes
# -----------------------
def process_root(process_id: str) -> Path:
    p = process_root_path(process_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def zones_dir(process_id: str) -> Path:
    d = zones_dir_path(process_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def model_dir(process_id: str) -> Path:
    d = model_dir_path(process_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def sessions_dir(process_id: str) -> Path:
    d = sessions_dir_path(process_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def session_root(process_id: str, session_id: str) -> Path:
    d = session_root_path(process_id, session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def captures_dir(process_id: str, session_id: str) -> Path:
    d = captures_dir_path(process_id, session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def group_dir(process_id: str, session_id: str, timestamp: str) -> Path:
    d = group_dir_path(process_id, session_id, timestamp)
    d.mkdir(parents=True, exist_ok=True)
    return d

# NOTE: function names retained but now point to .json instead of .js
def vocab_js_path(process_id: str) -> Path:
    # JSON location (no mkdir)
    return process_root_path(process_id) / "vocab.json"

def annotations_js_path(process_id: str) -> Path:
    # JSON location (no mkdir)
    return process_root_path(process_id) / "annotations.json"

# Legacy (read-only) paths for migration
def _vocab_legacy_js_path(process_id: str) -> Path:
    return process_root_path(process_id) / "vocab.js"

def _annotations_legacy_js_path(process_id: str) -> Path:
    return process_root_path(process_id) / "annotations.js"

def zones_global_json(process_id: str) -> Path:
    # NO mkdir here — writer ensures parent exists
    return zones_dir_path(process_id) / "zones-global.json"

def ensure_process_dirs(process_id: str) -> None:
    """
    Explicitly create the process folder tree + empty JSON files.
    Call this ONLY for write paths, not from generic request bootstrap.
    """
    process_root(process_id)
    zones_dir(process_id)
    model_dir(process_id)
    sessions_dir(process_id)

    # Ensure JSON files exist (migrate legacy on first read below anyway)
    vocab_p = vocab_js_path(process_id)
    if not vocab_p.exists():
        write_json_atomic(vocab_p, [])

    ann_p = annotations_js_path(process_id)
    if not ann_p.exists():
        write_json_atomic(ann_p, {})

def ensure_session_dirs(process_id: str, session_id: str) -> Path:
    """
    Explicitly create session folders (write path).
    """
    ensure_process_dirs(process_id)
    captures_dir(process_id, session_id)
    return session_root(process_id, session_id)

# -----------------------
# JSON helpers
# -----------------------
def read_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure parent for writes
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

# -----------------------
# Legacy JS export fallback (for migration)
# -----------------------
_JS_EXPORT = re.compile(r"^export\s+default\s+(.+?);\s*$", re.S)

def _read_js_export(path: Path, expect: str) -> Any:
    if not path.exists():
        return [] if expect == "array" else {}
    try:
        txt = path.read_text(encoding="utf-8").strip()
        m = _JS_EXPORT.match(txt)
        payload = m.group(1) if m else txt
        payload2 = re.sub(r",\s*([}\]])", r"\1", payload)  # strip trailing commas
        return json.loads(payload2)
    except Exception:
        return [] if expect == "array" else {}

def _read_json_with_legacy_fallback(json_path: Path, legacy_js_path: Path, default: Any, expect: str) -> Any:
    if json_path.exists():
        return read_json(json_path, default)
    legacy_data = _read_js_export(legacy_js_path, expect)
    try:
        write_json_atomic(json_path, legacy_data)
    except Exception:
        pass
    return legacy_data if legacy_data is not None else default

# -----------------------
# Central annotations adapter (JSON-backed)
# -----------------------
def load_annotations(session_dir: Path) -> Dict[str, Any]:
    """
    Return a dict for THIS session only: { "<timestamp>": { "global": [...], ... } }
    Read from data/<process>/annotations.json (shape: { "<sid>": { "<ts>": {...}, ... }, ... }).
    """
    try:
        sid = session_dir.name                      # does not require folder to exist
        proc = session_dir.parent.parent            # .../data/<process>
        ann_json = annotations_js_path(proc.name)   # no mkdir
        ann_js_legacy = _annotations_legacy_js_path(proc.name)
        all_ann = _read_json_with_legacy_fallback(ann_json, ann_js_legacy, {}, expect="object")
        sess_map = all_ann.get(sid, {}) if isinstance(all_ann, dict) else {}
        return sess_map if isinstance(sess_map, dict) else {}
    except Exception:
        return {}

def save_annotations_for_session(process_id: str, session_id: str, session_map: Dict[str, Any]) -> None:
    ann_json = annotations_js_path(process_id)           # no mkdir, writer ensures parent
    ann_js_legacy = _annotations_legacy_js_path(process_id)
    all_ann = _read_json_with_legacy_fallback(ann_json, ann_js_legacy, {}, expect="object")
    if not isinstance(all_ann, dict):
        all_ann = {}
    all_ann[session_id] = session_map
    write_json_atomic(ann_json, all_ann)

def read_vocab(process_id: str) -> List[str]:
    vocab_json = vocab_js_path(process_id)
    vocab_js_legacy = _vocab_legacy_js_path(process_id)
    arr = _read_json_with_legacy_fallback(vocab_json, vocab_js_legacy, [], expect="array")
    return list(arr) if isinstance(arr, list) else []

def write_vocab(process_id: str, labels: List[str]) -> None:
    uniq: List[str] = []
    for s in labels:
        s = str(s).strip()
        if s and s not in uniq:
            uniq.append(s)
    write_json_atomic(vocab_js_path(process_id), uniq)

# -----------------------
# Zones helpers
# -----------------------
def save_zones_global(process_id: str, zones: List[Dict[str, Any]]) -> None:
    """Save zones to zones-global.json with debugging"""
    p = zones_global_json(process_id)
    p.parent.mkdir(parents=True, exist_ok=True)  # ensure zones/ exists for writes
    
    # Debug log
    print(f"[save_zones_global] Saving {len(zones)} zones to {p}")
    for i, zone in enumerate(zones):
        print(f"  Zone {i}: id={zone.get('id')}, x={zone.get('x')}, y={zone.get('y')}, w={zone.get('w')}, h={zone.get('h')}")
    
    p.write_text(json.dumps(zones, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save_zones_global] Saved successfully")

def load_zones_global(process_id: str) -> List[Dict[str, Any]]:
    """Load zones from zones-global.json with debugging"""
    p = zones_global_json(process_id)  # no mkdir
    print(f"[load_zones_global] Loading zones from {p}")
    
    if not p.exists():
        print(f"[load_zones_global] File does not exist: {p}")
        return []
    
    try:
        zones = json.loads(p.read_text(encoding="utf-8"))
        print(f"[load_zones_global] Loaded {len(zones)} zones")
        for i, zone in enumerate(zones):
            print(f"  Zone {i}: id={zone.get('id')}, x={zone.get('x')}, y={zone.get('y')}, w={zone.get('w')}, h={zone.get('h')}")
        return zones
    except Exception as e:
        print(f"[load_zones_global] Failed to load zones: {e}")
        return []

# -----------------------
# Image helpers
# -----------------------
def decode_data_url_to_pil(data_url: str) -> Image.Image:
    m = re.match(r"^data:(image/\w+);base64,(.+)$", data_url)
    if not m:
        raise ValueError("Invalid data URL")
    raw = base64.b64decode(m.group(2))
    return Image.open(io.BytesIO(raw)).convert("RGB")

def save_zone_image(img: Image.Image, out_path: Path, quality: int = 92) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure write parent
    img.save(str(out_path), format="JPEG", quality=quality, optimize=True)

# -----------------------
# Deletion helpers (safe) + listing
# -----------------------
def _is_within(parent: Path, child: Path) -> bool:
    try:
        parent = parent.resolve()
        child = child.resolve()
        return str(child).startswith(str(parent))
    except Exception:
        return False

def list_groups(process_id: str, session_id: str) -> List[Dict[str, Any]]:
    cap = captures_dir_path(process_id, session_id)  # path-only
    if not cap.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for ts_dir in sorted([p for p in cap.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
        rels = [(Path("captures") / ts_dir.name / f.name).as_posix() for f in sorted(ts_dir.glob("*.jpg"))]
        rows.append({"timestamp": ts_dir.name, "images": rels})
    return rows

def delete_group(process_id: str, session_id: str, timestamp: str) -> bool:
    d = group_dir_path(process_id, session_id, timestamp)  # path-only
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not d.exists()

def delete_group_image(process_id: str, session_id: str, timestamp: str, rel_path: str) -> bool:
    base = session_root_path(process_id, session_id)  # path-only
    p = (base / rel_path).resolve()
    if not _is_within(base, p):
        return False
    if p.exists() and p.is_file():
        try:
            p.unlink()
            return True
        except:
            return False
    return False

def delete_session_dir(process_id: str, session_id: str) -> bool:
    sdir = session_root_path(process_id, session_id)
    if not _is_within(DATA_DIR, sdir):
        return False
    shutil.rmtree(sdir, ignore_errors=True)
    return not sdir.exists()

def delete_process_dir(process_id: str) -> bool:
    proot = process_root_path(process_id)
    if not _is_within(DATA_DIR, proot):
        return False
    shutil.rmtree(proot, ignore_errors=True)
    return not proot.exists()
