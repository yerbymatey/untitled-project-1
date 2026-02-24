import argparse
import logging
from typing import Dict, List, Tuple

from db.session import get_db_session
from utils.vector_config import EMBEDDING_DIM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VECTOR_COLUMNS: List[Tuple[str, str]] = [
    ("tweets", "embedding"),
    ("media", "joint_embedding"),
    ("media", "image_embedding"),
]


def _fetch_configured_dimension(session, table: str, column: str) -> int:
    session.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = %s
          AND a.attname = %s
          AND a.attisdropped = FALSE
        """,
        (table, column),
    )
    row = session.fetchone()
    if not row:
        raise RuntimeError(f"Column not found: {table}.{column}")
    atttypmod = row["atttypmod"]
    if atttypmod is None or atttypmod <= 0:
        return 0
    # pgvector stores vector(n) typmod as n + 4.
    return atttypmod - 4


def _fetch_stored_dimensions(session, table: str, column: str) -> Dict[int, int]:
    session.execute(
        f"""
        SELECT vector_dims({column}) AS dimension, COUNT(*) AS count
        FROM {table}
        WHERE {column} IS NOT NULL
        GROUP BY vector_dims({column})
        ORDER BY count DESC, dimension ASC
        """
    )
    rows = session.fetchall()
    return {int(row["dimension"]): int(row["count"]) for row in rows}


def _log_state(session) -> Dict[Tuple[str, str], Dict[str, object]]:
    state: Dict[Tuple[str, str], Dict[str, object]] = {}
    logger.info("Current embedding column state:")
    for table, column in VECTOR_COLUMNS:
        configured_dim = _fetch_configured_dimension(session, table, column)
        stored_dims = _fetch_stored_dimensions(session, table, column)
        total_rows = sum(stored_dims.values())
        state[(table, column)] = {
            "configured_dim": configured_dim,
            "stored_dims": stored_dims,
            "total_rows": total_rows,
        }
        logger.info(
            "  %s.%s -> configured vector(%s), populated rows=%s, stored dims=%s",
            table,
            column,
            configured_dim,
            total_rows,
            stored_dims if stored_dims else "{}",
        )
    return state


def _wipe_embeddings(session) -> None:
    for table, column in VECTOR_COLUMNS:
        session.execute(f"UPDATE {table} SET {column} = NULL WHERE {column} IS NOT NULL")
        logger.info("Nullified %s.%s rows: %s", table, column, session.cursor.rowcount)


def _alter_vector_dimensions(session, target_dimension: int) -> None:
    for table, column in VECTOR_COLUMNS:
        session.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector({target_dimension})"
        )
        logger.info("Updated %s.%s type to vector(%s)", table, column, target_dimension)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and migrate embedding vector dimensions. "
            "Defaults to read-only dry run."
        )
    )
    parser.add_argument(
        "--target-dimension",
        type=int,
        default=EMBEDDING_DIM,
        help=f"Target vector dimension (default: EMBEDDING_DIM={EMBEDDING_DIM})",
    )
    parser.add_argument(
        "--wipe-old",
        action="store_true",
        help="Set all existing embedding vectors to NULL before re-encoding.",
    )
    parser.add_argument(
        "--no-alter-columns",
        action="store_true",
        help="Skip ALTER TABLE ... TYPE vector(target_dimension).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply migration writes. Without this flag, only inspection and planned actions are shown.",
    )
    args = parser.parse_args()

    with get_db_session() as session:
        state = _log_state(session)
        target_dimension = int(args.target_dimension)

        need_alter = any(
            int(meta["configured_dim"]) != target_dimension for meta in state.values()
        ) and not args.no_alter_columns

        has_mismatched_data = False
        for meta in state.values():
            stored_dims = meta["stored_dims"]
            if any(dim != target_dimension for dim in stored_dims):
                has_mismatched_data = True
                break

        logger.info("Planned actions:")
        if args.wipe_old:
            logger.info("  - Wipe existing embeddings (set vectors to NULL)")
        else:
            logger.info("  - No embedding wipe requested")

        if args.no_alter_columns:
            logger.info("  - Skip column type changes")
        elif need_alter:
            logger.info("  - Alter vector columns to vector(%s)", target_dimension)
        else:
            logger.info("  - Column dimensions already match target")

        if has_mismatched_data and not args.wipe_old:
            logger.warning(
                "Stored vectors include non-target dimensions and --wipe-old is not set. "
                "Re-encoding required vectors may still fail until old vectors are nulled."
            )

        if not args.apply:
            logger.info("Dry run complete. Re-run with --apply to execute migration actions.")
            return

        if args.wipe_old:
            _wipe_embeddings(session)

        if need_alter:
            _alter_vector_dimensions(session, target_dimension)

        logger.info("Migration complete. Final state:")
        _log_state(session)


if __name__ == "__main__":
    main()
