"""Vision-language generation backends (report section 3.2.5).

A prompt is a list of parts: ("text", str) or ("image", path) - kept in order, so text and
pictures are interleaved exactly as Z_input = [X_T(Q); X_T(R_text); X_V(R_vis)] in eq. 23.
"""
from __future__ import annotations

import base64
import io

from .config import Config


def _load_image(path: str, max_side: int = 1280):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    return im


class Backend:
    name = "base"
    supports_vision = False

    def generate(self, parts, system: str = "", max_new_tokens: int = 512) -> str:
        raise NotImplementedError


class ExtractiveBackend(Backend):
    """No model: the Reasoner falls back to extractive answers. Great for demos, CI and low-end laptops."""
    name = "extractive"

    def generate(self, parts, system: str = "", max_new_tokens: int = 512) -> str:
        raise RuntimeError("The extractive backend cannot generate free text.")


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible server: Ollama (`ollama pull qwen2.5vl:7b`), vLLM, LM Studio, OpenRouter..."""
    name = "openai_compat"
    supports_vision = True

    def __init__(self, cfg: Config):
        self.base, self.model, self.key = cfg.api_base.rstrip("/"), cfg.api_model, cfg.api_key

    @staticmethod
    def _data_uri(path: str) -> str:
        buf = io.BytesIO()
        _load_image(path).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def generate(self, parts, system: str = "", max_new_tokens: int = 512) -> str:
        import requests
        content = []
        for kind, val in parts:
            if kind == "text":
                content.append({"type": "text", "text": val})
            else:
                content.append({"type": "image_url", "image_url": {"url": self._data_uri(val)}})
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        r = requests.post(f"{self.base}/chat/completions",
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "messages": msgs, "max_tokens": max_new_tokens,
                                "temperature": 0.1}, timeout=600)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class QwenLocalBackend(Backend):
    """Qwen2.5-VL loaded in-process with Hugging Face transformers."""
    name = "qwen_local"
    supports_vision = True

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        if cfg.load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.qwen_model, **kwargs)
        # limit visual tokens per image (dynamic-resolution patches) to keep VRAM bounded
        self.processor = AutoProcessor.from_pretrained(cfg.qwen_model, min_pixels=256 * 28 * 28,
                                                       max_pixels=1024 * 28 * 28)

    def generate(self, parts, system: str = "", max_new_tokens: int = 512) -> str:
        import torch
        content, images = [], []
        for kind, val in parts:
            if kind == "text":
                content.append({"type": "text", "text": val})
            else:
                content.append({"type": "image"})
                images.append(_load_image(val))
        msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
               [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()


def make_backend(cfg: Config) -> Backend:
    b = cfg.backend.lower()
    if b == "openai_compat":
        return OpenAICompatBackend(cfg)
    if b == "qwen_local":
        return QwenLocalBackend(cfg)
    return ExtractiveBackend()
