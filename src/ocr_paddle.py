from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from typing import List, Dict

class PaddleEngine:
    def __init__(self, lang: str = "ru"):
        self.ocr = PaddleOCR(lang=lang, det=True, rec=True, use_angle_cls=True, show_log=False)

    def run(self, img: Image.Image) -> List[Dict]:
        arr = np.array(img)
        res = self.ocr.ocr(arr, cls=True)
        out = []
        if res and res[0]:
            for line in res[0]:
                (x1,y1),(x2,y2),(x3,y3),(x4,y4) = line[0]
                text, conf = line[1]
                out.append({"text": text, "conf": float(conf), "bbox": [int(x1),int(y1),int(x3),int(y3)]})
        return out