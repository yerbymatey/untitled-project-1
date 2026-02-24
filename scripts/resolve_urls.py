#!/usr/bin/env python3
"""Resolve t.co shortlinks to expanded URLs and optionally fetch page metadata.

For each URL in the urls table with NULL expanded_url:
1. Follow redirects to get the final URL
2. Optionally fetch og:title, og:description, and a content snippet

Usage:
    python -m scripts.resolve_urls [--fetch-metadata] [--limit N] [--dry-run]
"""
import argparse
import logging
import re
import sys
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from db.session import get_db_session

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# Don't resolve twitter/x.com URLs (they're internal references)
SKIP_DOMAINS = {"x.com", "twitter.com", "t.co"}
REQUEST_TIMEOUT = 10
RATE_DELAY = 0.3  # seconds between requests


def resolve_redirect(url: str) -> Optional[str]:
    """Follow redirects on a URL and return the final destination."""
    try:
        resp = requests.head(
            url,
            allow_redirects=True,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; bookmark-resolver/1.0)"},
        )
        final = resp.url
        # Don't store if it just resolved to another t.co
        if urlparse(final).netloc in SKIP_DOMAINS:
            return final
        return final
    except Exception as e:
        logger.debug(f"Failed to resolve {url}: {e}")
        return None


def fetch_page_metadata(url: str) -> Dict[str, Optional[str]]:
    """Fetch og:title, og:description, and a content snippet from a URL."""
    result = {"title": None, "description": None, "content_snippet": None}
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; bookmark-resolver/1.0)"},
            stream=True,
        )
        # Only read first 100KB to avoid downloading huge pages
        content = b""
        for chunk in resp.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 102400:
                break
        html = content.decode("utf-8", errors="replace")

        # Extract <title>
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            result["title"] = title_match.group(1).strip()[:500]

        # Extract og:title (preferred over <title>)
        og_title = re.search(
            r'<meta\s+(?:property|name)=["\']og:title["\']\s+content=["\']([^"\']*)["\']',
            html,
            re.IGNORECASE,
        )
        if og_title:
            result["title"] = og_title.group(1).strip()[:500]

        # Extract og:description
        og_desc = re.search(
            r'<meta\s+(?:property|name)=["\']og:description["\']\s+content=["\']([^"\']*)["\']',
            html,
            re.IGNORECASE,
        )
        if og_desc:
            result["description"] = og_desc.group(1).strip()[:1000]

        # Fallback: meta description
        if not result["description"]:
            meta_desc = re.search(
                r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']*)["\']',
                html,
                re.IGNORECASE,
            )
            if meta_desc:
                result["description"] = meta_desc.group(1).strip()[:1000]

        # Content snippet: strip tags, grab first ~500 chars of body text
        body_match = re.search(r"<body[^>]*>(.*)</body>", html, re.IGNORECASE | re.DOTALL)
        if body_match:
            body_text = re.sub(r"<script[^>]*>.*?</script>", "", body_match.group(1), flags=re.DOTALL | re.IGNORECASE)
            body_text = re.sub(r"<style[^>]*>.*?</style>", "", body_text, flags=re.DOTALL | re.IGNORECASE)
            body_text = re.sub(r"<[^>]+>", " ", body_text)
            body_text = re.sub(r"\s+", " ", body_text).strip()
            if len(body_text) > 50:
                result["content_snippet"] = body_text[:500]

    except Exception as e:
        logger.debug(f"Failed to fetch metadata for {url}: {e}")

    return result


def get_unresolved_urls(limit: int = 1000) -> List[Tuple[str, str]]:
    """Get URLs from the db that have no expanded_url."""
    with get_db_session() as session:
        session.execute(
            """
            SELECT tweet_id, url
            FROM urls
            WHERE expanded_url IS NULL
            ORDER BY tweet_id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [(row["tweet_id"], row["url"]) for row in session.fetchall()]


def main():
    parser = argparse.ArgumentParser(description="Resolve t.co URLs and fetch metadata")
    parser.add_argument("--limit", type=int, default=1000, help="Max URLs to process")
    parser.add_argument("--fetch-metadata", action="store_true", help="Also fetch og:title/description")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    unresolved = get_unresolved_urls(limit=args.limit)
    logger.info(f"Found {len(unresolved)} URLs to resolve")

    if not unresolved:
        return

    resolved_count = 0
    metadata_count = 0

    with get_db_session() as session:
        for tweet_id, tco_url in unresolved:
            expanded = resolve_redirect(tco_url)
            if not expanded:
                time.sleep(RATE_DELAY)
                continue

            if args.dry_run:
                logger.info(f"  {tco_url} -> {expanded}")
                resolved_count += 1
                time.sleep(RATE_DELAY)
                continue

            # Parse display_url from expanded
            parsed = urlparse(expanded)
            display = parsed.netloc + parsed.path[:50]

            update_params = [expanded, display, tweet_id, tco_url]
            update_sql = """
                UPDATE urls
                SET expanded_url = %s, display_url = %s
                WHERE tweet_id = %s AND url = %s
            """

            # Optionally fetch metadata
            if args.fetch_metadata and parsed.netloc not in SKIP_DOMAINS:
                meta = fetch_page_metadata(expanded)
                if any(meta.values()):
                    update_sql = """
                        UPDATE urls
                        SET expanded_url = %s, display_url = %s,
                            title = %s, description = %s, content_snippet = %s
                        WHERE tweet_id = %s AND url = %s
                    """
                    update_params = [
                        expanded, display,
                        meta["title"], meta["description"], meta["content_snippet"],
                        tweet_id, tco_url,
                    ]
                    metadata_count += 1

            session.execute(update_sql, update_params)
            resolved_count += 1
            time.sleep(RATE_DELAY)

    prefix = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"{prefix}Resolved {resolved_count} URLs" + (f", fetched metadata for {metadata_count}" if args.fetch_metadata else ""))


if __name__ == "__main__":
    main()
