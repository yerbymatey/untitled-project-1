import glob
import json
from datetime import datetime
from typing import Dict, List

from pipelines.ingest import BookmarkIngester

TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_twitter_date(date_str):
    """Convert Twitter date format to PostgreSQL timestamp format"""
    try:
        return datetime.strptime(date_str, TWITTER_DATE_FORMAT)
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None


def _normalize_created_at(created_at):
    """Normalize serialized timestamps into datetime values."""
    if not created_at or isinstance(created_at, datetime):
        return created_at

    if isinstance(created_at, str):
        try:
            # Handle timezone suffix used by many JSON serializers.
            return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            return parse_twitter_date(created_at)

    return created_at


def _prepare_bookmarks(bookmarks: List[Dict]) -> List[Dict]:
    """Normalize bookmark values for ingestion."""
    prepared = []
    for bookmark in bookmarks:
        if not isinstance(bookmark, dict):
            continue
        normalized = bookmark.copy()
        normalized["created_at"] = _normalize_created_at(normalized.get("created_at"))
        prepared.append(normalized)
    return prepared


def _count_processable_bookmarks(bookmarks: List[Dict], ingester: BookmarkIngester) -> int:
    """Keep historical return behavior of reporting processable bookmarks."""
    count = 0
    for bookmark in bookmarks:
        user = bookmark.get("user")
        if not user or not bookmark.get("id"):
            continue

        handle = ingester._resolve_screen_name(user, bookmark.get("url"))
        user_id = user.get("id") or handle
        if user_id:
            count += 1

    return count


def import_bookmarks_file(file_path):
    """Import bookmarks from a JSON file into the database"""
    print(f"Importing data from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bookmarks = data.get("bookmarks", [])
    if not bookmarks:
        print("No bookmarks found in the file")
        return 0

    prepared_bookmarks = _prepare_bookmarks(bookmarks)
    ingester = BookmarkIngester()

    try:
        processed_count = _count_processable_bookmarks(prepared_bookmarks, ingester)
        ingester.ingest_bookmarks(prepared_bookmarks, save_to_file=False)
        print(f"Successfully imported {processed_count} bookmarks from {file_path}")
        return processed_count

    except Exception as e:
        print(f"Error importing data: {e}")
        return 0


def import_all_bookmarks():
    """Import all bookmark JSON files in the current directory"""
    bookmark_files = glob.glob("bookmarks_*.json")

    if not bookmark_files:
        print("No bookmark files found in the current directory")
        return

    total_imported = 0
    for file_path in bookmark_files:
        count = import_bookmarks_file(file_path)
        total_imported += count

    print(f"Total imported bookmarks: {total_imported}")


if __name__ == "__main__":
    import_all_bookmarks()
