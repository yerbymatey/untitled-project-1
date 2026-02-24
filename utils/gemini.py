import base64
import logging
import os
from typing import Optional

import requests
import time
import random

logger = logging.getLogger(__name__)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENDPOINT = os.getenv(
    "GEMINI_GENERATE_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
)


def _detect_mime_type(url: str, timeout: int = 10) -> str:
    """Attempt to detect the image mime type using a HEAD request, fallback to jpeg."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        ct = resp.headers.get("content-type") or resp.headers.get("Content-Type")
        if ct and ct.lower().startswith("image/"):
            return ct.split(";")[0].strip()
    except Exception as e:
        logger.debug("MIME type detection failed for %s: %s", url, e)
    return "image/jpeg"


def _fetch_base64(url: str, timeout: int = 30) -> str:
    """Download the image and return a base64 string without newlines."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("ascii")


def _post_with_retries(url: str, headers: dict, json_body: dict, timeout: int = 60, max_retries: int = 5) -> dict:
    """POST with exponential backoff on 429/5xx; honors Retry-After when present."""
    attempt = 0
    backoff = 1.0
    while True:
        try:
            r = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            if r.status_code == 429:
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


def describe_image_with_gemini(image_url: str, prompt: Optional[str] = None, timeout: int = 60) -> str:
    """Call Gemini to produce a semantic description of an image at a hosted URL.

    Args:
        image_url: Publicly reachable URL for the image.
        prompt: Optional instruction to guide the captioning.
        timeout: HTTP timeout in seconds.

    Returns:
        The model's text description.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    mime_type = _detect_mime_type(image_url)
    image_b64 = _fetch_base64(image_url)

    if not prompt:
        prompt = (
            "Analyze this image and provide two things:\n\n"
            "1. A concise semantic description of what the image shows and why someone "
            "might find it notable or worth saving (2-3 sentences). Focus on the core meaning, "
            "implications, or takeaway rather than superficial visual details.\n\n"
            "2. If the image contains any readable text — such as a screenshot of an article, "
            "research paper, website, social media post, code snippet, chart with labels, "
            "or any other text-bearing content — extract and reproduce that text as accurately "
            "as possible. Preserve paragraph structure. If there is no readable text, write "
            "\"none\".\n\n"
            "Format your response exactly as:\n"
            "DESCRIPTION: [your visual/semantic description]\n"
            "TEXT_CONTENT: [extracted text or \"none\"]"
        )

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ]
    }

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        data = _post_with_retries(GEMINI_ENDPOINT, headers, body, timeout=timeout)
        # Navigate Gemini response structure for the text output
        candidates = data.get("candidates") or []
        if candidates:
            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )
            for p in parts:
                if "text" in p:
                    return p["text"].strip()
        # Fallback: try older response shape
        if "contents" in data:
            for c in data["contents"]:
                for p in c.get("parts", []):
                    if "text" in p:
                        return p["text"].strip()
        raise ValueError("No text candidate returned by Gemini")
    except requests.HTTPError as e:
        # Bubble up 403/404 handling to callers to decide on retry vs. marking
        raise


def parse_gemini_response(raw_text: str) -> dict:
    """Parse a structured Gemini response into description and extracted text.

    Expected format:
        DESCRIPTION: ...
        TEXT_CONTENT: ...

    Returns dict with keys 'image_desc' and 'extr_text' (None if no text found).
    Falls back to treating the entire response as image_desc if parsing fails.
    """
    desc = raw_text.strip()
    extr = None

    text_marker = "TEXT_CONTENT:"
    desc_marker = "DESCRIPTION:"

    ti = raw_text.find(text_marker)
    di = raw_text.find(desc_marker)

    if di != -1 and ti != -1 and di < ti:
        # Both markers present in expected order — split them
        desc = raw_text[di + len(desc_marker):ti].strip()
        extr_raw = raw_text[ti + len(text_marker):].strip()
        if extr_raw.lower() not in ("none", "\"none\"", "n/a", ""):
            extr = extr_raw
    elif di != -1 and ti != -1 and ti < di:
        # Reversed order — TEXT_CONTENT before DESCRIPTION
        extr_raw = raw_text[ti + len(text_marker):di].strip()
        desc = raw_text[di + len(desc_marker):].strip()
        if extr_raw.lower() not in ("none", "\"none\"", "n/a", ""):
            extr = extr_raw
    elif di != -1:
        # Only DESCRIPTION marker
        desc = raw_text[di + len(desc_marker):].strip()
    # else: no markers, use full text as desc (fallback)

    return {"image_desc": desc, "extr_text": extr}
