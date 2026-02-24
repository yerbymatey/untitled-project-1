"""Shared embedding helpers for text and vision features."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from io import BytesIO
from typing import Dict, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Default model identifiers can be overridden via environment variables to support experimentation.
TEXT_MODEL_NAME: str = os.getenv("TEXT_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
VISION_MODEL_NAME: str = os.getenv("VISION_EMBED_MODEL", "nomic-ai/nomic-embed-vision-v1.5")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))


def _select_device() -> torch.device:
    """Select the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = _select_device()


@lru_cache(maxsize=1)
def _get_text_components() -> Tuple[AutoTokenizer, AutoModel]:
    """Load and cache the text tokenizer/model pair."""
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _get_vision_components() -> Tuple[AutoImageProcessor, AutoModel]:
    """Load and cache the vision processor/model pair."""
    processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(VISION_MODEL_NAME, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    return processor, model


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply attention-aware mean pooling to model outputs."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def download_image(url: str, timeout: int = 10) -> Image.Image:
    """Fetch an image from a URL and return a PIL image."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def get_text_embedding(text: str) -> torch.Tensor:
    """Generate a normalized embedding for the supplied text."""
    # Allow empty strings by feeding a single space through the model so we still
    # produce a consistent embedding for "no text" cases.
    if not text:
        text = " "

    tokenizer, model = _get_text_components()
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        embedding = mean_pooling(output, encoded["attention_mask"])
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0)


def _extract_image_embedding(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract an image embedding tensor from a model forward pass."""
    if hasattr(outputs, "image_embeds"):
        return outputs.image_embeds
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0]
    raise ValueError("Vision model output does not contain an image embedding")


def get_image_embedding(media_url: str, image_desc: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return joint (image+text) and image-only embeddings for a media item."""
    if not media_url:
        raise ValueError("media_url is required")

    processor, model = _get_vision_components()
    image = download_image(media_url)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        image_embedding = _extract_image_embedding(outputs)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)

    image_embedding = image_embedding.squeeze(0)

    if image_desc:
        try:
            text_embedding = get_text_embedding(image_desc)
            joint_embedding = F.normalize((image_embedding + text_embedding) / 2, p=2, dim=0)
        except Exception as exc:
            logger.warning("Falling back to image-only joint embedding for %s: %s", media_url, exc)
            joint_embedding = image_embedding.clone()
    else:
        joint_embedding = image_embedding.clone()

    return joint_embedding, image_embedding


__all__ = [
    "DEVICE",
    "EMBEDDING_DIM",
    "TEXT_MODEL_NAME",
    "VISION_MODEL_NAME",
    "download_image",
    "get_image_embedding",
    "get_text_embedding",
    "mean_pooling",
]
