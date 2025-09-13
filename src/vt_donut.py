from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json, re
class DonutEngine:
    def __init__(self, model_id: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
        self.processor = DonutProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.eval()
    def infer(self, img: Image.Image, question: str = "Extract key fields"):
        prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        pv = self.processor(img, return_tensors="pt").pixel_values
        task_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        out = self.model.generate(pv, decoder_input_ids=task_ids, max_new_tokens=256)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", text, flags=re.S)
        return json.loads(m.group(0)) if m else {"raw": text}