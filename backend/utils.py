import base64
import io
import os
from pathlib import Path
from typing import List
from PIL import Image

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_session_dirs(session_id: str):
    """Create session folders lazily."""
    sdir = DATA_DIR / session_id
    (sdir / "captures").mkdir(parents=True, exist_ok=True)
    (sdir / "mosaics").mkdir(parents=True, exist_ok=True)
    (sdir / "model").mkdir(parents=True, exist_ok=True)
    return sdir


def decode_data_url_to_pil(data_url: str) -> Image.Image:
    """Decode a data URL (e.g. 'data:image/jpeg;base64,...') to PIL Image."""
    header, b64 = data_url.split(",", 1)
    binary = base64.b64decode(b64)
    return Image.open(io.BytesIO(binary)).convert("RGB")


def save_zone_crops(session_dir: Path, timestamp: str, zone_ids: List[int], images: List[Image.Image]):
    tdir = session_dir / "captures" / timestamp
    tdir.mkdir(parents=True, exist_ok=True)
    for zid, img in zip(zone_ids, images):
        img.save(tdir / f"zone_{zid}.jpg", quality=92)


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