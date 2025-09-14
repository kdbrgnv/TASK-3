# src/utils/image_tools.py
from typing import Sequence, Tuple, Union, Dict
from PIL import Image

# Типы, которые могут приходить как bbox:
# - [x1, y1, x2, y2]
# - {"x":..,"y":..,"w":..,"h":..}
# - [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]

BBox = Union[Sequence[float], Dict[str, float], Sequence[Sequence[float]]]

def _points_to_ltrb(pts: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    l, t, r, b = min(xs), min(ys), max(xs), max(ys)
    return int(l), int(t), int(r), int(b)

def _bbox_to_ltrb(bbox: BBox, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    if isinstance(bbox, dict):
        # формат dict
        x = float(bbox.get("x", bbox.get("left", 0)))
        y = float(bbox.get("y", bbox.get("top", 0)))
        w = float(bbox.get("w", bbox.get("width", 0)))
        h = float(bbox.get("h", bbox.get("height", 0)))
        l, t, r, b = x, y, x + w, y + h
    elif bbox and isinstance(bbox[0], (list, tuple)) and len(bbox[0]) == 2:
        # формат 4 точки
        l, t, r, b = _points_to_ltrb(bbox)
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        # формат [x1,y1,x2,y2]
        x1, y1, x2, y2 = map(float, bbox)
        l, t, r, b = x1, y1, x2, y2
    else:
        raise ValueError(f"Unsupported bbox format: {bbox}")

    # Зажимаем в границы
    l = max(0, min(int(l), img_w - 1))
    t = max(0, min(int(t), img_h - 1))
    r = max(0, min(int(r), img_w))
    b = max(0, min(int(b), img_h))
    if r < l: l, r = r, l
    if b < t: t, b = b, t
    return l, t, r, b

def safe_crop(image: Image.Image, bbox: BBox, expand: int = 2) -> Image.Image | None:
    """
    Возвращает кропнутый PIL.Image или None, если bbox некорректен.
    """
    if image is None or bbox is None:
        return None
    w, h = image.size
    try:
        l, t, r, b = _bbox_to_ltrb(bbox, w, h)
    except Exception:
        return None

    # Немного расширим рамку
    l = max(0, l - expand)
    t = max(0, t - expand)
    r = min(w, r + expand)
    b = min(h, b + expand)

    if r <= l or b <= t:
        return None
    return image.crop((l, t, r, b))
