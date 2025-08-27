import base64
import io
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent

# Sessions root (used by backend to locate session folders)
DATA_DIR = BASE_DIR / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Global vocab (shared across sessions)
LABELS_DIR = BASE_DIR / "data" / "labels"
VOCAB_PATH = LABELS_DIR / "vocab.json"

# Centralized annotations store (one file for all sessions)
CENTRAL_ANN_PATH = BASE_DIR / "data" / "annotations.json"

# ========== JSON helpers (atomic, tolerant) ==========

def read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def write_json_atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

# ========== Central annotations helpers ==========

def _read_central_ann() -> Dict[str, Dict]:
    return read_json(CENTRAL_ANN_PATH, {})

def _write_central_ann(obj: Dict[str, Dict]) -> None:
    write_json_atomic(CENTRAL_ANN_PATH, obj)

def _load_session_ann_from_central(session_id: str) -> Dict[str, Dict]:
    data = _read_central_ann()
    sess = data.get(session_id)
    return sess if isinstance(sess, dict) else {}

def _save_session_ann_to_central(session_id: str, sess_map: Dict[str, Dict]) -> None:
    data = _read_central_ann()
    if sess_map and isinstance(sess_map, dict):
        data[session_id] = sess_map
    else:
        # delete empty key
        if session_id in data:
            data.pop(session_id, None)
    _write_central_ann(data)

# Public helpers used by main.py
def save_session_annotations(session_id: str, sess_map: Dict[str, Dict]):
    _save_session_ann_to_central(session_id, sess_map)

# ========== Session folder helpers ==========

def ensure_session_dirs(session_id: str) -> Path:
    """Create session folders lazily."""
    sdir = DATA_DIR / session_id
    (sdir / "captures").mkdir(parents=True, exist_ok=True)
    (sdir / "mosaics").mkdir(parents=True, exist_ok=True)
    (sdir / "model").mkdir(parents=True, exist_ok=True)
    (sdir / "embeddings").mkdir(parents=True, exist_ok=True)  # for future pipeline
    return sdir
    
def ensure_simple_dir(session_id: str) -> Path:
    """Create only the session folder (no captures/mosaics/model subdirs)."""
    sdir = DATA_DIR / session_id
    sdir.mkdir(parents=True, exist_ok=True)
    return sdir

# ========== Data-URL decode ==========

def decode_data_url_to_pil(data_url: str) -> Image.Image:
    """Decode a data URL (e.g. 'data:image/jpeg;base64,...') to PIL Image."""
    header, b64 = data_url.split(",", 1)
    binary = base64.b64decode(b64)
    return Image.open(io.BytesIO(binary)).convert("RGB")

# ========== Capture saving ==========

def save_zone_crops(session_dir: Path, timestamp: str, zone_ids: List[int], images: List[Image.Image]) -> List[str]:
    """
    Save crops under captures/<timestamp>/zone_{id}.jpg
    Return the relative paths (from session_dir) in the same order as input.
    """
    tdir = session_dir / "captures" / str(timestamp)
    tdir.mkdir(parents=True, exist_ok=True)
    rels: List[str] = []
    for zid, img in zip(zone_ids, images):
        name = f"zone_{zid}.jpg"
        img.save(tdir / name, quality=92)
        rels.append(str(Path("captures") / str(timestamp) / name))
    return rels

# ========== Mosaic builder ==========

def make_mosaic_horiz(images: List[Image.Image], pad: int = 4, bg=(20,20,20)) -> Image.Image:
    """Combine images horizontally with padding."""
    if not images:
        raise ValueError("No images to mosaic")
    heights = [im.height for im in images]
    max_h = max(heights)
    widths = [im.width for im in images]
    total_w = sum(widths) + pad * (len(images) + 1)
    mosaic = Image.new("RGB", (total_w, max_h + 2*pad), bg)
    x = pad
    for im in images:
        # Center vertically
        y = pad + (max_h - im.height)//2
        mosaic.paste(im, (x, y))
        x += im.width + pad
    return mosaic

# ========== Group index (per session) ==========

def _default_role_for(zid: int) -> str:
    return f"zone_{zid}"

def upsert_group_entry(
    session_dir: Path,
    timestamp: str | int,
    zone_ids: List[int],
    rel_paths: List[str],
    roles: Optional[List[Optional[str]]] = None,
    primary_id: Optional[int] = None,
    tz: Optional[str] = None
) -> Dict:
    """
    Ensure a group entry exists/updated for this timestamp.
    Merge zones by id; update role/path; keep groups sorted by timestamp.
    Note: timestamps here are numeric (per real session). Merged training sessions
    build their own index and may use non-numeric composite timestamps; this
    function is not used for merged sessions.
    """
    gi_path = session_dir / "group_index.json"
    index = read_json(gi_path, {"session_id": session_dir.name, "groups": []})
    ts = int(str(timestamp))

    # find group
    grp = None
    for g in index["groups"]:
        if int(g.get("timestamp", -1)) == ts:
            grp = g
            break

    # build incoming zones
    zlist = []
    for i, (zid, rel) in enumerate(zip(zone_ids, rel_paths)):
        role = None
        if roles and i < len(roles):
            role = roles[i]
        role = role or _default_role_for(zid)
        zlist.append({"id": int(zid), "role": role, "path": rel})

    if grp is None:
        grp = {"timestamp": ts, "tz": tz or "UTC", "zones": zlist}
        if primary_id is not None:
            grp["primary_id"] = int(primary_id)
        index["groups"].append(grp)
    else:
        # merge zones by id
        existing = {int(z["id"]): z for z in grp.get("zones", [])}
        for z in zlist:
            existing[int(z["id"])] = {"id": int(z["id"]), "role": z["role"], "path": z["path"]}
        grp["zones"] = sorted(existing.values(), key=lambda z: int(z["id"]))
        if primary_id is not None:
            grp["primary_id"] = int(primary_id)
        if tz:
            grp["tz"] = tz

    # sort by timestamp (numeric sessions only)
    index["groups"].sort(key=lambda g: int(g["timestamp"]))
    write_json_atomic(gi_path, index)
    return grp

# ========== Label vocab (global, user-driven) ==========

_slug_re = re.compile(r"[^a-z0-9]+")
def slugify(name: str) -> str:
    s = _slug_re.sub("-", name.strip().lower()).strip("-")
    return s or "label"

def load_vocab() -> Dict:
    return read_json(VOCAB_PATH, {"labels": []})

def add_label_to_vocab(name: str, parent: Optional[str] = None) -> Dict:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab()
    slug = slugify(name)
    lid = slug.replace("-", "_")
    for e in vocab["labels"]:
        if e.get("slug") == slug:
            # existing
            return e
    entry = {"id": lid, "name": name, "slug": slug, "parent": parent}
    vocab["labels"].append(entry)
    write_json_atomic(VOCAB_PATH, vocab)
    return entry

# ========== Annotations (CENTRAL, group-level) ==========

def load_annotations(session_dir: Path) -> Dict[str, Dict]:
    """
    Backward compatible: returns THIS session's map { "<ts>": {...} } from the central file.
    If a legacy per-session annotations.json exists, import it into central once.
    """
    session_id = session_dir.name
    sess = _load_session_ann_from_central(session_id)
    if sess:
        return sess

    # one-time lazy migration from legacy per-session file
    legacy = read_json(session_dir / "annotations.json", {})
    if isinstance(legacy, dict) and legacy:
        _save_session_ann_to_central(session_id, legacy)
        return legacy
    return {}

def upsert_annotation(
    session_dir: Path,
    timestamp: str | int,
    global_labels: Optional[List[str]],
    by_role: Optional[Dict[str, List[str]]],
    notes: Optional[str] = None
) -> Dict:
    session_id = session_dir.name
    ann = load_annotations(session_dir)  # this now reads central
    ts = str(int(str(timestamp)))

    rec = ann.get(ts, {})
    if global_labels is not None:
        seen = set()
        rec["global"] = [x for x in global_labels if not (x in seen or seen.add(x))]
    if by_role is not None:
        brec = {}
        for k, vals in by_role.items():
            seen = set()
            brec[k] = [x for x in vals if not (x in seen or seen.add(x))]
        rec["by_role"] = brec
    if notes is not None:
        rec["notes"] = notes

    ann[ts] = rec
    _save_session_ann_to_central(session_id, ann)
    return rec

def set_model_suggestion(session_dir: Path, timestamp: str | int, suggestion: Dict):
    """
    Non-destructively attach the model's suggestion to the central annotations
    for this session + timestamp.

    `suggestion` should look like:
      {
        "top": "<label_id>",
        "probs": {"label_id": float, ...},
        "ts": <unix_ms>  # optional, when the prediction was made
      }

    This preserves existing keys such as "global", "by_role", and "notes".
    Returns the updated annotation record for that timestamp.
    """
    session_id = session_dir.name
    ann = load_annotations(session_dir)
    ts = str(int(str(timestamp)))

    rec = dict(ann.get(ts, {}))
    rec["model"] = dict(suggestion) if isinstance(suggestion, dict) else {"value": suggestion}
    ann[ts] = rec

    _save_session_ann_to_central(session_id, ann)
    return rec
