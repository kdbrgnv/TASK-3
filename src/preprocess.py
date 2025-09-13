from PIL import Image, ImageEnhance, ImageFilter
import fitz
from pathlib import Path
from typing import List


def pdf_to_images(path: str | Path, dpi: int = 200) -> List[Image.Image]:
    doc = fitz.open(str(path))
    pages = []
    for p in doc:
        pix = p.get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        pages.append(Image.frombytes(mode, [pix.width, pix.height], pix.samples).convert("RGB"))
        return pages


def preprocess_light(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img