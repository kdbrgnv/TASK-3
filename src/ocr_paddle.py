# src/ocr_paddle.py
from typing import List, Dict
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

class PaddleEngine:
    def __init__(self, lang: str = "ru"):
        # Никаких спорных аргументов — только язык
        self.ocr = PaddleOCR(lang=lang)

    def run(self, img: Image.Image) -> List[Dict]:
        """
        Возвращает список:
        [{"text": "...", "conf": 0.98, "bbox": [x1,y1,x2,y2]}]
        """
        # На всякий случай приводим к RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img)
        out: List[Dict] = []

        # Новый API (если у инстанса есть .predict)
        if hasattr(self.ocr, "predict"):
            results = self.ocr.predict(arr) or []
            for res in results:
                for line in res.get("ocr_info", []):
                    text = line.get("text", "")
                    conf = float(line.get("score", 0.0))
                    box = line.get("box")  # [x1,y1,x2,y2,x3,y3,x4,y4]
                    if box and len(box) == 8:
                        xs = box[0::2]; ys = box[1::2]
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                    else:
                        x1 = y1 = x2 = y2 = 0
                    out.append({"text": text, "conf": conf, "bbox": [int(x1), int(y1), int(x2), int(y2)]})
            return out

        # Старый API: .ocr(image, cls=True)
        results = self.ocr.ocr(arr, cls=True) or []
        if results and results[0]:
            for line in results[0]:
                # line: [points, (text, score)]
                points = line[0]                  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text, conf = line[1]
                xs = [p[0] for p in points]; ys = [p[1] for p in points]
                x1, y1 = min(xs), min(ys); x2, y2 = max(xs), max(ys)
                out.append({"text": text, "conf": float(conf), "bbox": [int(x1), int(y1), int(x2), int(y2)]})
        return out
