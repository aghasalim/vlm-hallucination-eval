"""Thin wrappers around BLIP (generation) and CLIP (verification).

Models are loaded lazily and cached, because the Streamlit app and the eval
harness both import this module and neither should pay for a model it does not
use.

Note: the `.eval()` calls below are PyTorch's inference-mode switch (disables
dropout/batchnorm updates). They are unrelated to Python's builtin eval.
"""
from __future__ import annotations

import functools

import torch
from PIL import Image

from . import config


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _inference_mode(model):
    model.eval()
    return model


@functools.lru_cache(maxsize=1)
def _captioner():
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL).to(device())
    return proc, _inference_mode(model)


@functools.lru_cache(maxsize=1)
def _vqa():
    from transformers import BlipForQuestionAnswering, BlipProcessor
    proc = BlipProcessor.from_pretrained(config.VQA_MODEL)
    model = BlipForQuestionAnswering.from_pretrained(config.VQA_MODEL).to(device())
    return proc, _inference_mode(model)


@functools.lru_cache(maxsize=1)
def _clip():
    from transformers import CLIPModel, CLIPProcessor
    proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device())
    return proc, _inference_mode(model)


@torch.no_grad()
def caption(image: Image.Image, max_new_tokens: int = 40) -> str:
    proc, model = _captioner()
    inputs = proc(image, return_tensors="pt").to(device())
    # Beam search rather than sampling: the measurement must be deterministic,
    # and a hallucination rate that changes between runs is not a measurement.
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3)
    return proc.decode(out[0], skip_special_tokens=True).strip()


@torch.no_grad()
def ask(image: Image.Image, question: str, max_new_tokens: int = 8) -> str:
    proc, model = _vqa()
    inputs = proc(image, question, return_tensors="pt").to(device())
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3)
    return proc.decode(out[0], skip_special_tokens=True).strip().lower()


@torch.no_grad()
def clip_scores(image: Image.Image, texts: list[str]) -> list[float]:
    """Raw cosine similarities between the image and each text, x100.

    Deliberately *not* softmaxed here. A softmax over an arbitrary candidate list
    makes the score depend on which other candidates happen to be in the list,
    which is fine for classification and wrong for "is this claim true".
    """
    proc, model = _clip()
    inputs = proc(text=texts, images=image, return_tensors="pt",
                  padding=True, truncation=True).to(device())
    out = model(**inputs)
    img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return (img @ txt.T).squeeze(0).mul(100).cpu().tolist()


def yes_no(answer: str) -> bool | None:
    """Map a free-text VQA answer onto a boolean. Returns None if it is neither,
    so that unparseable answers are counted rather than silently read as 'no'."""
    a = answer.strip().lower().strip(".")
    if a.startswith("yes"):
        return True
    if a.startswith("no"):
        return False
    return None
