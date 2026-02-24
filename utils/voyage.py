import logging
import os
from typing import List, Optional, Tuple

import requests
import time
import random

logger = logging.getLogger(__name__)


VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_EMBED_ENDPOINT = os.getenv(
    "VOYAGE_EMBED_URL", "https://api.voyageai.com/v1/embeddings"
)
VOYAGE_MM_ENDPOINT = os.getenv(
    "VOYAGE_MULTIMODAL_URL", "https://api.voyageai.com/v1/multimodalembeddings"
)
VOYAGE_CTX_ENDPOINT = os.getenv(
    "VOYAGE_CONTEXTUAL_URL", "https://api.voyageai.com/v1/contextualizedembeddings"
)
VOYAGE_RERANK_ENDPOINT = os.getenv(
    "VOYAGE_RERANK_URL", "https://api.voyageai.com/v1/rerank"
)


def _headers() -> dict:
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set")
    return {"Authorization": f"Bearer {VOYAGE_API_KEY}", "Content-Type": "application/json"}


def _post_with_retries(url: str, payload: dict, headers: dict, timeout: int = 60, max_retries: int = 5) -> dict:
    """POST with simple exponential backoff on 429/5xx, honoring Retry-After when present."""
    attempt = 0
    backoff = 1.0
    while True:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                # Too Many Requests, obey Retry-After or backoff
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                wait += random.uniform(0, 0.25)
                if attempt >= max_retries:
                    r.raise_for_status()
                time.sleep(wait)
                attempt += 1
                backoff = min(backoff * 2, 16)
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code and 500 <= code < 600 and attempt < max_retries:
                wait = backoff + random.uniform(0, 0.25)
                time.sleep(wait)
                attempt += 1
                backoff = min(backoff * 2, 16)
                continue
            raise


def _parse_embeddings_from_response(data: dict) -> List[List[float]]:
    """Best-effort extraction of embeddings from Voyage API responses.

    Handles shapes like:
      {"data": [{"embedding": [...]}, ...]}
      {"data": [{"data": [{"embedding": [...]}]}]}  # defensive
    Returns list of vectors or empty list if none found.
    """
    vectors: List[List[float]] = []
    rows = data.get("data", []) if isinstance(data, dict) else []
    for row in rows:
        if isinstance(row, dict):
            if "embedding" in row:
                vectors.append(row["embedding"])  # type: ignore[arg-type]
            elif "data" in row and isinstance(row["data"], list):
                for inner in row["data"]:
                    if isinstance(inner, dict) and "embedding" in inner:
                        vectors.append(inner["embedding"])  # type: ignore[arg-type]
        elif isinstance(row, list):
            # Sometimes API might return raw vectors (unlikely, but safe)
            if row and isinstance(row[0], (float, int)):
                vectors.append(row)  # type: ignore[list-item]
    if not vectors:
        logger.warning("Voyage response had no embeddings; keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))
    return vectors


def voyage_multimodal_embeddings(
    inputs: List[dict],
    model: str = "voyage-multimodal-3.5",
    input_type: Optional[str] = "document",
    timeout: int = 60,
) -> List[List[float]]:
    """Call Voyage multimodal embeddings endpoint.

    Args:
        inputs: A list of {"content": [...]} items with pieces of type text or image_url.
        model: The multimodal embedding model.
        input_type: Optional input_type (null, query, document).
        timeout: HTTP timeout.
    Returns:
        List of embeddings (lists of floats), in the same order as inputs.
    """
    payload = {"inputs": inputs, "model": model}
    if input_type is not None:
        payload["input_type"] = input_type

    data = _post_with_retries(VOYAGE_MM_ENDPOINT, payload, _headers(), timeout=timeout)
    return _parse_embeddings_from_response(data)


def voyage_embeddings(
    texts: List[str],
    model: str = "voyage-4",
    input_type: Optional[str] = "document",
    output_dimension: Optional[int] = None,
    timeout: int = 60,
) -> List[List[float]]:
    """Call Voyage text embeddings endpoint (v1/embeddings).

    Args:
        texts: List of strings to embed. Max 1000 items, max 320K tokens for voyage-4.
        model: Voyage text model, default voyage-4.
        input_type: null, query, or document.
        output_dimension: Optional dimension (256, 512, 1024, 2048).
        timeout: HTTP timeout.
    Returns:
        List of embeddings in the same order as inputs.
    """
    payload: dict = {"input": texts, "model": model}
    if input_type is not None:
        payload["input_type"] = input_type
    if output_dimension is not None:
        payload["output_dimension"] = int(output_dimension)

    data = _post_with_retries(VOYAGE_EMBED_ENDPOINT, payload, _headers(), timeout=timeout)
    return _parse_embeddings_from_response(data)


def voyage_contextualized_embeddings(
    inputs: List[List[str]],
    model: str = "voyage-context-3",
    input_type: Optional[str] = "document",
    output_dimension: Optional[int] = None,
    timeout: int = 60,
) -> List[List[float]]:
    """Call Voyage contextualized chunk embeddings (legacy, for voyage-context-3).

    Args:
        inputs: List of lists of strings. Use [[text]] for single-item documents.
        model: Voyage context model.
        input_type: null, query, or document.
        output_dimension: Optional dimension.
        timeout: HTTP timeout.
    Returns:
        embeddings list aligned with the inputs order.
    """
    payload: dict = {"inputs": inputs, "model": model}
    if input_type is not None:
        payload["input_type"] = input_type
    if output_dimension is not None:
        payload["output_dimension"] = int(output_dimension)

    data = _post_with_retries(VOYAGE_CTX_ENDPOINT, payload, _headers(), timeout=timeout)
    return _parse_embeddings_from_response(data)


def voyage_rerank(
    query: str,
    documents: List[str],
    model: str = "rerank-2.5",
    top_k: Optional[int] = None,
    return_documents: bool = False,
    timeout: int = 60,
) -> List[dict]:
    """Call Voyage reranker API and return results.

    Args:
        query: The query string.
        documents: List of document strings to rerank.
        model: Rerank model name.
        top_k: Optional limit of results returned.
        return_documents: Whether to include documents in response.
        timeout: HTTP timeout.
    Returns:
        List of {index, relevance_score[, document]} sorted by relevance.
    """
    payload = {
        "query": query,
        "documents": documents,
        "model": model,
        "return_documents": return_documents,
    }
    if top_k is not None:
        payload["top_k"] = int(top_k)

    data = _post_with_retries(VOYAGE_RERANK_ENDPOINT, payload, _headers(), timeout=timeout)
    return data.get("data", [])
