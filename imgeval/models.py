"""Lazy-loaded vision models (Hugging Face). Loaded once, on the GPU when available.

CLIP ViT-L/14 is the same backbone as the RQ1 taste model, so taste vectors live in its space.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_ID = "openai/clip-vit-large-patch14"
DINO_ID = "facebook/dinov2-base"
OWL_ID = "google/owlv2-base-patch16-ensemble"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"


@lru_cache(maxsize=None)
def _clip():
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval(), CLIPProcessor.from_pretrained(CLIP_ID)


@lru_cache(maxsize=None)
def _dino():
    from transformers import AutoImageProcessor, AutoModel
    return AutoModel.from_pretrained(DINO_ID).to(DEVICE).eval(), AutoImageProcessor.from_pretrained(DINO_ID)


@lru_cache(maxsize=None)
def _owl():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    return Owlv2ForObjectDetection.from_pretrained(OWL_ID).to(DEVICE).eval(), Owlv2Processor.from_pretrained(OWL_ID)


@lru_cache(maxsize=None)
def _depth():
    from transformers import pipeline
    return pipeline("depth-estimation", model=DEPTH_ID, device=0 if DEVICE == "cuda" else -1)


def _norm(x: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy()


@torch.no_grad()
def clip_embed(images: list[Image.Image]) -> np.ndarray:
    model, proc = _clip()
    inp = proc(images=[i.convert("RGB") for i in images], return_tensors="pt").to(DEVICE)
    out = model.get_image_features(**inp)
    if not isinstance(out, torch.Tensor):  # newer transformers return a model output object
        out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
    return _norm(out)


@torch.no_grad()
def dino_embed(images: list[Image.Image]) -> np.ndarray:
    model, proc = _dino()
    inp = proc(images=[i.convert("RGB") for i in images], return_tensors="pt").to(DEVICE)
    return _norm(model(**inp).last_hidden_state[:, 0])  # CLS token, as in DreamBench's DINO score


@torch.no_grad()
def detect(image: Image.Image, queries: list[str], threshold: float = 0.1) -> list[dict]:
    """Open-vocabulary detection. Returns [{query, score, box(x0,y0,x1,y1)}] sorted by score."""
    model, proc = _owl()
    image = image.convert("RGB")
    inp = proc(text=[queries], images=image, return_tensors="pt").to(DEVICE)
    out = model(**inp)
    # OWLv2 pads to a square, so boxes are relative to the padded size.
    side = max(image.size)
    post = getattr(proc, "post_process_grounded_object_detection", None) or proc.post_process_object_detection
    res = post(outputs=out, threshold=threshold, target_sizes=torch.tensor([[side, side]], device=DEVICE))[0]
    dets = []
    for s, lab, box in zip(res["scores"], res["labels"], res["boxes"]):
        x0, y0, x1, y1 = [float(v) for v in box]
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(image.width), x1), min(float(image.height), y1)
        if x1 - x0 > 4 and y1 - y0 > 4:
            dets.append({"query": queries[int(lab)], "score": float(s), "box": (x0, y0, x1, y1)})
    return sorted(dets, key=lambda d: -d["score"])


def depth(image: Image.Image) -> np.ndarray:
    return np.asarray(_depth()(image.convert("RGB"))["predicted_depth"].squeeze().cpu(), dtype=np.float32)
