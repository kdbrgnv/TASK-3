# src/ocr_utils.py
import cv2
import numpy as np
from PIL import Image

def preprocess_for_ocr(pil: Image.Image) -> Image.Image:
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # адаптивная бинаризация
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 15)
    # небольшой морф-открытие для мусора
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    # ресайз если слишком маленькое
    h,w = bw.shape
    scale = 1280 / max(h,w) if max(h,w) < 1280 else 1.0
    if scale<1.0:
        bw = cv2.resize(bw, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(bw)
