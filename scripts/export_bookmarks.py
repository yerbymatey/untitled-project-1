#!/usr/bin/env python3
"""Export bookmarks from PostgreSQL to portable formats."""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import struct
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from db.session import get_db_session


logger = logging.getLogger(__name__)


def _configure_logging(quiet: bool = False) -> None:
    """Configure script logging and suppress noisy DB info logs."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr)
    logging.getLogger("db.session").setLevel(logging.WARNING)


def _parse_vector_text(vector_text: Optional[str]) -> Optional[List[float]]:
    """Convert pgvector text format ('[1,2,3]') into Python floats."""
    if not vector_text:
        return None
    values = vector_text.strip().strip("[]")
    if not values:
        return []
    return [float(part.strip()) for part in values.split(",") if part.strip()]


def _encode_vector_base64(values: Optional[List[float]]) -> Optional[Dict[str, Any]]:
    """Encode vector values as little-endian float32 bytes in base64."""
    if values is None:
        return None
    packed = struct.pack(f"<{len(values)}f", *values) if values else b""
    return {
        "encoding": "base64",
        "dtype": "float32",
        "endianness": "little",
        "shape": [len(values)],
        "data": base64.b64encode(packed).decode("ascii"),
    }


def _build_filters(
    since: Optional[date],
    query: Optional[str],
) -> Tuple[str, List[Any]]:
    """Build SQL WHERE/LIMIT clauses for tweet export filters."""
    where_clauses: List[str] = []
    params: List[Any] = []

    if since is not None:
        where_clauses.append("t.created_at >= %s")
        params.append(since)

    if query:
        where_clauses.append("t.text ILIKE %s")
        params.append(f"%{query}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return where_sql, params


def _fetch_export_rows(
    since: Optional[date],
    limit: Optional[int],
    query: Optional[str],
) -> List[Dict[str, Any]]:
    """Fetch bookmark rows and related entities from PostgreSQL."""
    where_sql, params = _build_filters(since=since, query=query)

    sql = f"""
        SELECT
            t.id,
            t.user_id,
            t.text,
            t.created_at,
            t.retweet_count,
            t.favorite_count,
            t.reply_count,
            t.quote_count,
            t.is_quote_status,
            t.quoted_tweet_id,
            t.url,
            t.has_media,
            t.embedding::text AS embedding_text,
            u.name AS user_name,
            u.verified AS user_verified,
            u.followers_count AS user_followers_count,
            u.following_count AS user_following_count,
            u.description AS user_description
        FROM tweets t
        LEFT JOIN users u ON u.id = t.user_id
        {where_sql}
        ORDER BY t.created_at DESC NULLS LAST, t.id DESC
    """
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    with get_db_session() as session:
        session.execute(sql, tuple(params))
        rows = list(session.fetchall())
        if not rows:
            return []

        tweet_ids = [row["id"] for row in rows]
        user_ids = list({row["user_id"] for row in rows if row.get("user_id")})

        placeholder_tweets = ",".join(["%s"] * len(tweet_ids))

        session.execute(
            f"""
            SELECT th.tweet_id, h.tag
            FROM tweet_hashtags th
            JOIN hashtags h ON h.id = th.hashtag_id
            WHERE th.tweet_id IN ({placeholder_tweets})
            ORDER BY th.tweet_id, h.tag
            """,
            tuple(tweet_ids),
        )
        hashtag_rows = session.fetchall()

        session.execute(
            f"""
            SELECT tweet_id, url, expanded_url, display_url
            FROM urls
            WHERE tweet_id IN ({placeholder_tweets})
            ORDER BY tweet_id, url
            """,
            tuple(tweet_ids),
        )
        url_rows = session.fetchall()

        session.execute(
            f"""
            SELECT
                tweet_id,
                media_url,
                type,
                alt_text,
                image_desc,
                extr_text,
                joint_embedding::text AS joint_embedding_text,
                image_embedding::text AS image_embedding_text
            FROM media
            WHERE tweet_id IN ({placeholder_tweets})
            ORDER BY tweet_id, media_url
            """,
            tuple(tweet_ids),
        )
        media_rows = session.fetchall()

        user_url_rows: Sequence[Dict[str, Any]] = []
        if user_ids:
            placeholder_users = ",".join(["%s"] * len(user_ids))
            session.execute(
                f"""
                SELECT user_id, url, expanded_url, display_url
                FROM user_description_urls
                WHERE user_id IN ({placeholder_users})
                ORDER BY user_id, url
                """,
                tuple(user_ids),
            )
            user_url_rows = session.fetchall()

    hashtags_by_tweet: Dict[str, List[str]] = {}
    for item in hashtag_rows:
        hashtags_by_tweet.setdefault(item["tweet_id"], []).append(item["tag"])

    urls_by_tweet: Dict[str, List[Dict[str, Any]]] = {}
    for item in url_rows:
        urls_by_tweet.setdefault(item["tweet_id"], []).append(
            {
                "url": item["url"],
                "expanded_url": item["expanded_url"],
                "display_url": item["display_url"],
            }
        )

    media_by_tweet: Dict[str, List[Dict[str, Any]]] = {}
    for item in media_rows:
        media_by_tweet.setdefault(item["tweet_id"], []).append(
            {
                "media_url": item["media_url"],
                "type": item["type"],
                "alt_text": item["alt_text"],
                "image_desc": item["image_desc"],
                "extr_text": item["extr_text"],
                "joint_embedding": _parse_vector_text(item["joint_embedding_text"]),
                "image_embedding": _parse_vector_text(item["image_embedding_text"]),
            }
        )

    user_urls_by_user: Dict[str, List[Dict[str, Any]]] = {}
    for item in user_url_rows:
        user_urls_by_user.setdefault(item["user_id"], []).append(
            {
                "url": item["url"],
                "expanded_url": item["expanded_url"],
                "display_url": item["display_url"],
            }
        )

    payload: List[Dict[str, Any]] = []
    for row in rows:
        user_id = row.get("user_id")
        payload.append(
            {
                "tweet": {
                    "id": row["id"],
                    "user_id": user_id,
                    "text": row["text"],
                    "created_at": row["created_at"],
                    "retweet_count": row["retweet_count"],
                    "favorite_count": row["favorite_count"],
                    "reply_count": row["reply_count"],
                    "quote_count": row["quote_count"],
                    "is_quote_status": row["is_quote_status"],
                    "quoted_tweet_id": row["quoted_tweet_id"],
                    "url": row["url"],
                    "has_media": row["has_media"],
                    "embedding": _parse_vector_text(row["embedding_text"]),
                },
                "user": {
                    "id": user_id,
                    "name": row["user_name"],
                    "verified": row["user_verified"],
                    "followers_count": row["user_followers_count"],
                    "following_count": row["user_following_count"],
                    "description": row["user_description"],
                    "description_urls": user_urls_by_user.get(user_id, []),
                },
                "hashtags": hashtags_by_tweet.get(row["id"], []),
                "urls": urls_by_tweet.get(row["id"], []),
                "media": media_by_tweet.get(row["id"], []),
            }
        )

    return payload


def _make_json_document(
    records: List[Dict[str, Any]],
    include_embeddings: bool,
    since: Optional[date],
    query: Optional[str],
    limit: Optional[int],
) -> Dict[str, Any]:
    """Build a self-contained JSON export document."""
    bookmarks: List[Dict[str, Any]] = []
    for item in records:
        tweet = dict(item["tweet"])
        user = dict(item["user"])
        media_rows = [dict(media_item) for media_item in item["media"]]

        if include_embeddings:
            tweet["embedding"] = _encode_vector_base64(tweet.get("embedding"))
            for media_item in media_rows:
                media_item["joint_embedding"] = _encode_vector_base64(media_item.get("joint_embedding"))
                media_item["image_embedding"] = _encode_vector_base64(media_item.get("image_embedding"))
        else:
            tweet.pop("embedding", None)
            for media_item in media_rows:
                media_item.pop("joint_embedding", None)
                media_item.pop("image_embedding", None)

        bookmarks.append(
            {
                "tweet": tweet,
                "user": user,
                "hashtags": item["hashtags"],
                "urls": item["urls"],
                "media": media_rows,
            }
        )

    return {
        "schema_version": "1.0",
        "source": "twitter-bookmarks-postgres",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "include_embeddings": include_embeddings,
        "filters": {
            "since": since.isoformat() if since else None,
            "query": query,
            "limit": limit,
        },
        "total_bookmarks": len(bookmarks),
        "bookmarks": bookmarks,
    }


def _json_default(value: Any) -> Any:
    """JSON serializer for date/time objects."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _to_csv_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten bookmark records for CSV export."""
    rows: List[Dict[str, Any]] = []
    for item in records:
        user = item.get("user", {})
        user_label = user.get("id") or user.get("name") or ""
        rows.append(
            {
                "tweet_id": item["tweet"]["id"],
                "text": item["tweet"]["text"] or "",
                "url": item["tweet"]["url"] or "",
                "created_at": item["tweet"]["created_at"].isoformat() if item["tweet"]["created_at"] else "",
                "user": user_label,
                "media_urls": "|".join(media.get("media_url", "") for media in item.get("media", []) if media.get("media_url")),
                "hashtags": "|".join(item.get("hashtags", [])),
            }
        )
    return rows


def _to_parquet_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create parquet-ready row dicts with array columns for embeddings."""
    parquet_rows: List[Dict[str, Any]] = []
    for item in records:
        tweet = item["tweet"]
        user = item["user"]
        media_rows = []
        for media_item in item["media"]:
            media_rows.append(
                {
                    "media_url": media_item["media_url"],
                    "type": media_item["type"],
                    "alt_text": media_item["alt_text"],
                    "image_desc": media_item["image_desc"],
                    "extr_text": media_item["extr_text"],
                    "joint_embedding": media_item.get("joint_embedding"),
                    "image_embedding": media_item.get("image_embedding"),
                }
            )

        parquet_rows.append(
            {
                "tweet_id": tweet["id"],
                "user_id": tweet["user_id"],
                "text": tweet["text"],
                "created_at": tweet["created_at"],
                "retweet_count": tweet["retweet_count"],
                "favorite_count": tweet["favorite_count"],
                "reply_count": tweet["reply_count"],
                "quote_count": tweet["quote_count"],
                "is_quote_status": tweet["is_quote_status"],
                "quoted_tweet_id": tweet["quoted_tweet_id"],
                "url": tweet["url"],
                "has_media": tweet["has_media"],
                "embedding": tweet.get("embedding"),
                "user_name": user.get("name"),
                "user_verified": user.get("verified"),
                "user_followers_count": user.get("followers_count"),
                "user_following_count": user.get("following_count"),
                "user_description": user.get("description"),
                "user_description_urls": user.get("description_urls", []),
                "hashtags": item.get("hashtags", []),
                "urls": item.get("urls", []),
                "media": media_rows,
            }
        )
    return parquet_rows


def _write_text_output(payload: str, output_path: Optional[str]) -> None:
    """Write text payload to stdout or file."""
    if output_path:
        Path(output_path).write_text(payload, encoding="utf-8")
        return
    sys.stdout.write(payload)


def _export_json(records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    doc = _make_json_document(
        records=records,
        include_embeddings=not args.no_embeddings,
        since=args.since,
        query=args.query,
        limit=args.limit,
    )
    payload = json.dumps(doc, ensure_ascii=False, indent=2, default=_json_default)
    _write_text_output(payload + "\n", args.output)


def _export_csv(records: List[Dict[str, Any]], output_path: Optional[str]) -> None:
    rows = _to_csv_rows(records)
    fieldnames = ["tweet_id", "text", "url", "created_at", "user", "media_urls", "hashtags"]

    output_buffer = io.StringIO()
    writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _write_text_output(output_buffer.getvalue(), output_path)


def _export_parquet(records: List[Dict[str, Any]], output_path: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet export requires optional dependency 'pyarrow'. "
            "Install it with `pip install pyarrow`."
        ) from exc

    parquet_rows = _to_parquet_rows(records)
    table = pa.Table.from_pylist(parquet_rows)
    pq.write_table(table, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export bookmarks from PostgreSQL")
    parser.add_argument(
        "--format",
        choices=["json", "csv", "parquet"],
        default="json",
        help="Export format (default: json)",
    )
    parser.add_argument(
        "--output",
        help="Output path (default: stdout for json/csv; required for parquet)",
    )
    parser.add_argument(
        "--since",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        help="Only export bookmarks created at/after this date (YYYY-MM-DD)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of exported bookmarks")
    parser.add_argument("--query", help="Only export bookmarks whose text matches this term (ILIKE)")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embeddings in JSON export",
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

    if args.format == "parquet" and not args.output:
        parser.error("--output is required for parquet format")

    if args.no_embeddings and args.format != "json":
        logger.info("--no-embeddings only applies to json export")

    records = _fetch_export_rows(since=args.since, limit=args.limit, query=args.query)

    if args.format == "json":
        _export_json(records, args)
    elif args.format == "csv":
        _export_csv(records, args.output)
    else:
        _export_parquet(records, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
