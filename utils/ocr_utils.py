# src/ocr_utils.py
import cv2
import numpy as np
from PIL import Image

def preprocess_for_ocr(pil: Image.Image, *,
                       target_max_side: int = 1280,
                       use_clahe: bool = True,
                       do_unsharp: bool = True,
                       invert_if_needed: bool = True,
                       add_padding: int = 8,
                       return_rgb: bool = True) -> Image.Image:
    """
    Готовит изображение для OCR:
    - grayscale + CLAHE
    - адаптивная бинаризация
    - лёгкая морфология
    - масшт. до target_max_side (ап/даун)
    - авто-инверт фона при необходимости
    - небольшой белый паддинг
    - возврат PIL.Image (RGB по умолчанию)
    """
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)

    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    h, w = bw.shape[:2]
    max_side = max(h, w)
    scale = target_max_side / max_side
    if scale != 1.0:
        interp = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        bw = cv2.resize(bw, (new_w, new_h), interpolation=interp)

    if invert_if_needed:
        white = cv2.countNonZero(bw)
        black = bw.size - white
        if white < black:
            bw = cv2.bitwise_not(bw)

    if add_padding > 0:
        bw = cv2.copyMakeBorder(
            bw, add_padding, add_padding, add_padding, add_padding,
            borderType=cv2.BORDER_CONSTANT, value=255
        )

    if do_unsharp:
        blur = cv2.GaussianBlur(bw, (0, 0), 1.0)
        bw = cv2.addWeighted(bw, 1.5, blur, -0.5, 0)

    if return_rgb:
        rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)
    else:
        return Image.fromarray(bw)
