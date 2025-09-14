# src/ocr_utils.py
import cv2
import numpy as np
from PIL import Image
from typing import Literal, Tuple

Mode = Literal["soft", "binary"]

def _to_rgb(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def _from_rgb(bgr: np.ndarray, gray: bool = False) -> Image.Image:
    if gray:
        return Image.fromarray(bgr)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def _auto_invert_if_needed(gray: np.ndarray) -> np.ndarray:
    # если фон тёмный — инвертируем
    white = cv2.countNonZero(gray)
    black = gray.size - white
    return cv2.bitwise_not(gray) if white < black else gray

def _clahe_rgb(bgr: np.ndarray) -> np.ndarray:
    # CLAHE по Y-каналу (лучше для шрифтов)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    yuv = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def _deskew(gray: np.ndarray, max_angle: float = 10.0) -> Tuple[np.ndarray, float]:
    """
    Грубый deskew по Хаффу. Возвращает (выравненное изображение, угол).
    Работает быстро и достаточно для типовых договоров.
    """
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=max(120, int(0.15*max(h, w))))
    angle_deg = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for rho_theta in lines[:180]:  # ограничим
            rho, theta = rho_theta[0]
            a = (theta * 180.0 / np.pi) - 90.0  # приводим к «горизонтальной» системе
            # нормализуем к диапазону [-45; 45]
            if a > 45: a -= 90
            if a < -45: a += 90
            if abs(a) <= max_angle:
                angles.append(a)
        if angles:
            angle_deg = float(np.median(angles))
            M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return gray, angle_deg

def preprocess_for_ocr(
    pil: Image.Image,
    *,
    mode: Mode = "soft",            # "soft" — для PaddleOCR (рекомендуется), "binary" — как было
    target_width: int = 2400,       # апскейл для мелких шрифтов (1.6–2.4k обычно ок)
    add_padding: int = 8,
    do_unsharp: bool = True,
    return_rgb: bool = True,
) -> Image.Image:
    """
    Подготовка изображений под OCR.

    mode="soft"  (по умолчанию): RGB + CLAHE(Y) + bilateral + deskew + лёгкая резкость.
        Лучшее качество для PaddleOCR по русскому.

    mode="binary": старый путь (бинаризация, морфология) — пригодится для пост-обработки/масок.
    """
    bgr = _to_rgb(pil)

    # --- Универсальный апскейл до target_width ---
    h, w = bgr.shape[:2]
    if w < target_width:
        scale = target_width / float(w)
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    if mode == "soft":
        # 1) Контраст (CLAHE по Y)
        bgr = _clahe_rgb(bgr)

        # 2) Мягкое шумоподавление (сохраняем границы букв)
        bgr = cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)

        # 3) Deskew (по серому)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray, _ = _deskew(gray, max_angle=10.0)

        # 4) Лёгкая нерезкая маска, чуть повышаем чёткость
        if do_unsharp:
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
            # вставим обратно как Y-канал
            yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            y = sharp
            bgr = cv2.cvtColor(cv2.merge([y, u, v]), cv2.COLOR_YUV2BGR)

        # 5) Паддинг
        if add_padding > 0:
            bgr = cv2.copyMakeBorder(bgr, add_padding, add_padding, add_padding, add_padding,
                                     borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Возвращаем RGB (для PaddleOCR лучше RGB)
        return _from_rgb(bgr, gray=False) if return_rgb else Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

    # ===== mode == "binary" =====
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Мягкое шумоподавление
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)

    # Адаптивная бинаризация
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Морфология для очистки точек/мусора
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    # Deskew
    bw, _ = _deskew(bw, max_angle=10.0)

    # Авто-инверсия (текст всегда тёмный на светлом)
    bw = _auto_invert_if_needed(bw)

    # Паддинг
    if add_padding > 0:
        bw = cv2.copyMakeBorder(bw, add_padding, add_padding, add_padding, add_padding,
                                borderType=cv2.BORDER_CONSTANT, value=255)

    # Нерезкая маска
    if do_unsharp:
        blur = cv2.GaussianBlur(bw, (0, 0), 1.0)
        bw = cv2.addWeighted(bw, 1.5, blur, -0.5, 0)

    # RGB или GRAY
    if return_rgb:
        return Image.fromarray(cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB))
    return Image.fromarray(bw)
