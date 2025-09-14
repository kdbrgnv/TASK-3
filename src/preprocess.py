# src/preprocess.py
from pathlib import Path
from typing import List
from PIL import Image, ImageEnhance, ImageFilter
import fitz


def pdf_to_images(path: str | Path, dpi: int = 200) -> List[Image.Image]:
    """
    Конвертирует PDF в список изображений PIL.Image.
    :param path: путь к PDF-файлу
    :param dpi: разрешение (по умолчанию 200)
    :return: список изображений
    """
    doc = fitz.open(str(path))
    pages: List[Image.Image] = []

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
        pages.append(img)

    return pages


def preprocess_light(img: Image.Image) -> Image.Image:
    """
    Лёгкая предобработка изображения: увеличение контраста,
    повышение резкости и фильтрация шумов.
    """
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img
