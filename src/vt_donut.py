# src/vt_donut.py
import os
# Полностью выключаем accelerate-путь, чтобы не получить meta-тензоры
os.environ.setdefault("TRANSFORMERS_USE_ACCELERATE", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from typing import Optional, List
from huggingface_hub import list_repo_files, snapshot_download, hf_hub_download
from safetensors.torch import load_file as safe_load
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
from PIL import Image
import torch, json, re

# Кандидаты: сначала твой, затем док-варианты с safetensors
CANDIDATES: List[str] = [
    "naver-clova-ix/donut-base-finetuned-docvqa",
    "nielsr/donut-docvqa-demo",
    "fairuzafnan/donut-docvqa",
]

def _find_safetensors_filename(repo_id: str) -> Optional[str]:
    files = list_repo_files(repo_id)
    for name in files:
        if name.endswith(".safetensors"):
            return name
    return None

class DonutEngine:
    def __init__(self, model_id: Optional[str] = None):
        # 0) Выбираем репозиторий, в котором есть *.safetensors
        self.model_id = model_id or self._pick_model_with_safetensors()

        # 1) Скачиваем снапшот локально (без *.bin)
        local_dir = snapshot_download(
            repo_id=self.model_id,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "tokenizer.*",
                "vocab*",
                "merges.txt",
                "special_tokens_map.json",
                "generation_config.json",
                "config.json",
            ],
            ignore_patterns=["*.bin", "*.pt", "*.msgpack"],
        )

        # 2) Процессор — slow токенайзер (SentencePiece), чтобы не ловить ValueError
        self.processor = DonutProcessor.from_pretrained(local_dir, use_fast=False)
        # некоторые чекпойнты требуют pad_token
        try:
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        except Exception:
            pass

        # 3) Находим и скачиваем точный safetensors-файл
        st_name = _find_safetensors_filename(self.model_id)
        if st_name is None:
            raise RuntimeError(f"В репозитории '{self.model_id}' нет *.safetensors (нужен другой чекпойнт).")
        weights_path = hf_hub_download(repo_id=self.model_id, filename=st_name, force_download=True)
        state_dict = safe_load(weights_path, device="cpu")

        # 4) Создаём модель из конфига (реальные тензоры сразу на CPU), грузим веса вручную
        cfg: VisionEncoderDecoderConfig = VisionEncoderDecoderConfig.from_pretrained(local_dir)

        # Отключаем SDPA везде (иначе может тронуть meta-маски)
        for obj in (cfg, getattr(cfg, "encoder", None), getattr(cfg, "decoder", None)):
            if obj is None:
                continue
            for k in ("attn_implementation", "_attn_implementation"):
                try:
                    setattr(obj, k, "eager")
                except Exception:
                    pass

        self.model = VisionEncoderDecoderModel(config=cfg)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        try:
            self.model.tie_weights()
        except Exception:
            pass

        # Дублируем отключение SDPA на уровне инстанса
        try:
            self.model.config.attn_implementation = "eager"
        except Exception:
            setattr(self.model.config, "_attn_implementation", "eager")
        dec = getattr(self.model, "decoder", None)
        if getattr(dec, "config", None) is not None:
            setattr(dec.config, "attn_implementation", "eager")
            setattr(dec.config, "_attn_implementation", "eager")

        self.model.eval()  # всё на CPU

    def _pick_model_with_safetensors(self) -> str:
        last_err = None
        for rid in CANDIDATES:
            try:
                if _find_safetensors_filename(rid):
                    return rid
            except Exception as e:
                last_err = e
        hint = f" (последняя ошибка: {last_err})" if last_err else ""
        raise RuntimeError("Не найден DocVQA чекпойнт с *.safetensors." + hint)

    def infer(self, img: Image.Image, question: str = "Extract key fields"):
        # Donut DocVQA промпт
        prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

        # Подготовка входов (CPU)
        enc = self.processor(img, return_tensors="pt")
        pixel_values = enc.pixel_values  # CPU

        task_ids = self.processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids  # CPU

        # Генерация
        with torch.no_grad():
            out_ids = self.model.generate(
                pixel_values,
                decoder_input_ids=task_ids,
                max_new_tokens=256,
                use_cache=True,
            )

        # Декод и извлечение JSON, если есть
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {"raw": text, "error": "Invalid JSON in output"}
        return {"raw": text}
