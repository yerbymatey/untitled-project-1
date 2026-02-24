#!/usr/bin/env python3
"""Import bookmarks from portable JSON export back into PostgreSQL."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import struct
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from db.session import get_db_session


logger = logging.getLogger(__name__)


def _configure_logging(quiet: bool = False) -> None:
    """Configure logging and reduce DB session noise."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr)
    logging.getLogger("db.session").setLevel(logging.WARNING)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse ISO-8601 timestamp strings for PostgreSQL parameters."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _decode_embedding(value: Any) -> Optional[List[float]]:
    """Decode an embedding from export JSON."""
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                return []
            return [float(item.strip()) for item in inner.split(",") if item.strip()]
        raise ValueError("Unsupported embedding string format")
    if isinstance(value, dict):
        data = value.get("data")
        shape = value.get("shape", [])
        if data is None:
            return None
        if value.get("dtype") not in (None, "float32"):
            raise ValueError("Only float32 embeddings are supported")
        if value.get("encoding") not in (None, "base64"):
            raise ValueError("Only base64 embeddings are supported")
        dim = int(shape[0]) if shape else 0
        raw_bytes = base64.b64decode(data.encode("ascii"))
        if dim == 0 and raw_bytes:
            raise ValueError("Embedding shape metadata missing")
        if dim == 0:
            return []
        expected_bytes = dim * 4
        if len(raw_bytes) != expected_bytes:
            raise ValueError("Embedding byte size does not match shape")
        return list(struct.unpack(f"<{dim}f", raw_bytes))
    raise ValueError(f"Unsupported embedding payload type: {type(value)!r}")


def _vector_to_pg(values: Optional[List[float]]) -> Optional[str]:
    """Convert a Python float list into pgvector text format."""
    if values is None:
        return None
    return str([float(item) for item in values])


def _read_json_document(input_path: Optional[str]) -> Any:
    """Read JSON from a file path or stdin."""
    if input_path and input_path != "-":
        return json.loads(Path(input_path).read_text(encoding="utf-8"))
    return json.load(sys.stdin)


def _normalize_bookmarks(document: Any) -> List[Dict[str, Any]]:
    """Normalize export JSON into bookmark records."""
    bookmarks = document.get("bookmarks")
    if isinstance(bookmarks, list):
        return bookmarks
    if isinstance(document, list):
        return document
    raise ValueError("Input JSON does not contain a bookmarks array")


def _collect_rows(bookmarks: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect and deduplicate relational rows for import."""
    users_by_id: Dict[str, Tuple[Any, ...]] = {}
    user_urls: set[Tuple[str, str, Optional[str], Optional[str]]] = set()
    tweets_by_id: Dict[str, Tuple[Any, ...]] = {}
    hashtags_by_tweet: Dict[str, set[str]] = {}
    urls_rows: set[Tuple[str, str, Optional[str], Optional[str]]] = set()
    media_rows: Dict[Tuple[str, str], Tuple[Any, ...]] = {}

    for bookmark in bookmarks:
        tweet = bookmark.get("tweet") or {}
        user = bookmark.get("user") or {}

        tweet_id = tweet.get("id") or bookmark.get("id")
        if not tweet_id:
            logger.warning("Skipping bookmark with missing tweet id")
            continue

        user_id = tweet.get("user_id") or user.get("id")
        if user_id:
            users_by_id[user_id] = (
                user_id,
                user.get("name"),
                bool(user.get("verified", False)),
                int(user.get("followers_count") or 0),
                int(user.get("following_count") or 0),
                user.get("description"),
            )
            for description_url in user.get("description_urls", []) or []:
                if isinstance(description_url, str):
                    url = description_url
                    expanded_url = None
                    display_url = None
                elif isinstance(description_url, dict):
                    url = description_url.get("url")
                    expanded_url = description_url.get("expanded_url")
                    display_url = description_url.get("display_url")
                else:
                    continue
                if url:
                    user_urls.add(
                        (
                            user_id,
                            url,
                            expanded_url,
                            display_url,
                        )
                    )

        tweet_embedding = _decode_embedding(tweet.get("embedding")) if "embedding" in tweet else None
        tweets_by_id[tweet_id] = (
            tweet_id,
            user_id,
            tweet.get("text"),
            _parse_timestamp(tweet.get("created_at")),
            int(tweet.get("retweet_count") or 0),
            int(tweet.get("favorite_count") or 0),
            int(tweet.get("reply_count") or 0),
            int(tweet.get("quote_count") or 0),
            bool(tweet.get("is_quote_status", False)),
            tweet.get("quoted_tweet_id"),
            tweet.get("url"),
            bool(tweet.get("has_media", False)),
            _vector_to_pg(tweet_embedding),
        )

        tags = hashtags_by_tweet.setdefault(tweet_id, set())
        for tag in bookmark.get("hashtags", []) or []:
            if tag:
                tags.add(str(tag))

        for url_item in bookmark.get("urls", []) or []:
            url_value = url_item.get("url")
            if url_value:
                urls_rows.add(
                    (
                        tweet_id,
                        url_value,
                        url_item.get("expanded_url"),
                        url_item.get("display_url"),
                    )
                )

        for media_item in bookmark.get("media", []) or []:
            media_url = media_item.get("media_url")
            if not media_url:
                continue

            joint_embedding = _decode_embedding(media_item.get("joint_embedding")) if "joint_embedding" in media_item else None
            image_embedding = _decode_embedding(media_item.get("image_embedding")) if "image_embedding" in media_item else None
            media_rows[(tweet_id, media_url)] = (
                tweet_id,
                media_url,
                media_item.get("type"),
                media_item.get("alt_text"),
                media_item.get("image_desc"),
                _vector_to_pg(joint_embedding),
                _vector_to_pg(image_embedding),
                media_item.get("extr_text"),
            )

    return {
        "users": list(users_by_id.values()),
        "user_urls": list(user_urls),
        "tweets": list(tweets_by_id.values()),
        "hashtags_by_tweet": hashtags_by_tweet,
        "urls": list(urls_rows),
        "media": list(media_rows.values()),
    }


def _insert_hashtag_links(session, hashtags_by_tweet: Dict[str, set[str]]) -> None:
    """Insert hashtags and tweet_hashtag links."""
    all_tags = sorted({tag for tags in hashtags_by_tweet.values() for tag in tags})
    if not all_tags:
        return

    session.execute_values(
        """
        INSERT INTO hashtags (tag)
        VALUES %s
        ON CONFLICT (tag) DO NOTHING
        """,
        [(tag,) for tag in all_tags],
    )

    placeholders = ",".join(["%s"] * len(all_tags))
    session.execute(
        f"SELECT id, tag FROM hashtags WHERE tag IN ({placeholders})",
        tuple(all_tags),
    )
    id_by_tag = {row["tag"]: row["id"] for row in session.fetchall()}

    link_values = []
    for tweet_id, tags in hashtags_by_tweet.items():
        for tag in tags:
            hashtag_id = id_by_tag.get(tag)
            if hashtag_id is not None:
                link_values.append((tweet_id, hashtag_id))

    if not link_values:
        return

    session.execute_values(
        """
        INSERT INTO tweet_hashtags (tweet_id, hashtag_id)
        VALUES %s
        ON CONFLICT (tweet_id, hashtag_id) DO NOTHING
        """,
        link_values,
    )


def _import_rows(rows: Dict[str, Any]) -> None:
    """Import normalized row data into PostgreSQL with upserts."""
    with get_db_session() as session:
        if rows["users"]:
            session.execute_values(
                """
                INSERT INTO users (
                    id, name, verified, followers_count, following_count, description
                )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    verified = EXCLUDED.verified,
                    followers_count = EXCLUDED.followers_count,
                    following_count = EXCLUDED.following_count,
                    description = EXCLUDED.description
                """,
                rows["users"],
            )

        if rows["user_urls"]:
            session.execute_values(
                """
                INSERT INTO user_description_urls (
                    user_id, url, expanded_url, display_url
                )
                VALUES %s
                ON CONFLICT (user_id, url) DO UPDATE SET
                    expanded_url = COALESCE(EXCLUDED.expanded_url, user_description_urls.expanded_url),
                    display_url = COALESCE(EXCLUDED.display_url, user_description_urls.display_url)
                """,
                rows["user_urls"],
            )

        if rows["tweets"]:
            session.execute_values(
                """
                INSERT INTO tweets (
                    id, user_id, text, created_at, retweet_count, favorite_count,
                    reply_count, quote_count, is_quote_status, quoted_tweet_id, url,
                    has_media, embedding
                )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    text = EXCLUDED.text,
                    created_at = EXCLUDED.created_at,
                    retweet_count = EXCLUDED.retweet_count,
                    favorite_count = EXCLUDED.favorite_count,
                    reply_count = EXCLUDED.reply_count,
                    quote_count = EXCLUDED.quote_count,
                    is_quote_status = EXCLUDED.is_quote_status,
                    quoted_tweet_id = EXCLUDED.quoted_tweet_id,
                    url = EXCLUDED.url,
                    has_media = EXCLUDED.has_media,
                    embedding = COALESCE(EXCLUDED.embedding, tweets.embedding)
                """,
                rows["tweets"],
                template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
            )

        _insert_hashtag_links(session, rows["hashtags_by_tweet"])

        if rows["urls"]:
            session.execute_values(
                """
                INSERT INTO urls (tweet_id, url, expanded_url, display_url)
                VALUES %s
                ON CONFLICT (tweet_id, url) DO UPDATE SET
                    expanded_url = COALESCE(EXCLUDED.expanded_url, urls.expanded_url),
                    display_url = COALESCE(EXCLUDED.display_url, urls.display_url)
                """,
                rows["urls"],
            )

        if rows["media"]:
            session.execute_values(
                """
                INSERT INTO media (
                    tweet_id, media_url, type, alt_text, image_desc, joint_embedding,
                    image_embedding, extr_text
                )
                VALUES %s
                ON CONFLICT (tweet_id, media_url) DO UPDATE SET
                    type = EXCLUDED.type,
                    alt_text = EXCLUDED.alt_text,
                    image_desc = COALESCE(EXCLUDED.image_desc, media.image_desc),
                    joint_embedding = COALESCE(EXCLUDED.joint_embedding, media.joint_embedding),
                    image_embedding = COALESCE(EXCLUDED.image_embedding, media.image_embedding),
                    extr_text = COALESCE(EXCLUDED.extr_text, media.extr_text)
                """,
                rows["media"],
                template="(%s,%s,%s,%s,%s,%s::vector,%s::vector,%s)",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import bookmarks from JSON export")
    parser.add_argument(
        "--input",
        default="-",
        help="Input JSON file path (default: stdin)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logging",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.quiet)

    document = _read_json_document(args.input)
    bookmarks = _normalize_bookmarks(document)
    rows = _collect_rows(bookmarks)
    _import_rows(rows)

    logger.info(
        "Imported %s bookmarks (%s users, %s tweets, %s media rows)",
        len(bookmarks),
        len(rows["users"]),
        len(rows["tweets"]),
        len(rows["media"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
