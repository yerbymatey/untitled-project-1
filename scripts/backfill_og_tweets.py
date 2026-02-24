#!/usr/bin/env python3
"""Backfill OG tweets and expanded URLs from existing JSON backup files.

Parses bookmarks_*.json files, extracts:
1. Original tweets from quoted statuses (inserts as independent rows)
2. Expanded URLs that were not captured in earlier scrapes

Usage:
    python -m scripts.backfill_og_tweets [--dry-run] [--file FILE]
"""
import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from db.session import get_db_session
from pipelines.ingest import BookmarkIngester

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)


def load_bookmarks_from_files(file_paths: List[str]) -> List[Dict]:
    """Load and deduplicate bookmarks from multiple JSON files."""
    seen_ids = set()
    all_bookmarks = []

    for path in sorted(file_paths):
        try:
            with open(path) as f:
                data = json.load(f)
            bookmarks = data.get("bookmarks", [])
            new_count = 0
            for bm in bookmarks:
                bm_id = bm.get("id")
                if bm_id and bm_id not in seen_ids:
                    seen_ids.add(bm_id)
                    all_bookmarks.append(bm)
                    new_count += 1
            logger.info(f"Loaded {path}: {len(bookmarks)} total, {new_count} new")
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")

    return all_bookmarks


def backfill_expanded_urls(bookmarks: List[Dict], dry_run: bool = False) -> int:
    """Update urls table with expanded_url from bookmark data."""
    updates = []
    for bm in bookmarks:
        tweet_id = bm.get("id")
        for url_obj in bm.get("urls", []):
            tco = url_obj.get("url")
            expanded = url_obj.get("expanded_url")
            display = url_obj.get("display_url")
            if tco and expanded:
                updates.append((expanded, display, tweet_id, tco))

        # Also handle quoted tweet URLs
        qs = bm.get("quoted_status")
        if qs:
            qt_id = qs.get("id")
            for url_obj in qs.get("urls", []):
                tco = url_obj.get("url")
                expanded = url_obj.get("expanded_url")
                display = url_obj.get("display_url")
                if tco and expanded:
                    updates.append((expanded, display, qt_id, tco))

    if not updates:
        logger.info("No expanded URLs to backfill")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Would update {len(updates)} URL rows with expanded_url")
        return len(updates)

    updated = 0
    with get_db_session() as session:
        for expanded, display, tweet_id, tco in updates:
            session.execute(
                """
                UPDATE urls
                SET expanded_url = COALESCE(%s, expanded_url),
                    display_url = COALESCE(%s, display_url)
                WHERE tweet_id = %s AND url = %s
                  AND expanded_url IS NULL
                """,
                (expanded, display, tweet_id, tco),
            )
            updated += session.cursor.rowcount

    logger.info(f"Updated {updated} URL rows with expanded_url")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill OG tweets and expanded URLs")
    parser.add_argument(
        "--file",
        type=str,
        help="Specific JSON file to process (default: all bookmarks_*.json in repo root)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument(
        "--skip-og", action="store_true", help="Skip OG tweet insertion"
    )
    parser.add_argument(
        "--skip-urls", action="store_true", help="Skip URL backfill"
    )
    args = parser.parse_args()

    # Find files
    if args.file:
        files = [args.file]
    else:
        files = glob.glob("bookmarks_*.json") + glob.glob("test/bookmarks_*.json")

    if not files:
        logger.error("No bookmark JSON files found")
        sys.exit(1)

    logger.info(f"Processing {len(files)} file(s)")
    bookmarks = load_bookmarks_from_files(files)
    logger.info(f"Total unique bookmarks: {len(bookmarks)}")

    # 1. Insert OG tweets
    if not args.skip_og:
        ingester = BookmarkIngester()
        og_tweets = ingester._extract_og_tweets(bookmarks)
        qt_count = sum(1 for bm in bookmarks if bm.get("quoted_status"))
        logger.info(f"Found {qt_count} quote tweets, {len(og_tweets)} unique OG tweets")

        if og_tweets and not args.dry_run:
            with get_db_session() as session:
                # Collect users from OG tweets
                users = ingester._collect_users(og_tweets)
                ingester._insert_users(session, users)
                ingester._insert_tweets(session, og_tweets)
                logger.info(f"Inserted/updated {len(og_tweets)} OG tweets")
        elif args.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(og_tweets)} OG tweets")

    # 2. Backfill expanded URLs
    if not args.skip_urls:
        backfill_expanded_urls(bookmarks, dry_run=args.dry_run)

    logger.info("Backfill complete")


if __name__ == "__main__":
    main()
