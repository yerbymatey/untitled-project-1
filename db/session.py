from contextlib import contextmanager
from typing import Generator
import threading
import os
import time

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2 import OperationalError

from utils.config import DB_CONFIG, DB_SCHEMA, REQUIRED_TABLES, PGVECTOR_REQUIRED
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseSession:
    _schema_verified = False
    _schema_lock = threading.Lock()

    def __init__(self, verify_schema: bool = True):
        self.conn = None
        self.cursor = None
        self._ensure_database(verify_schema=verify_schema)

    def _ensure_database(self, verify_schema: bool = True):
        """Ensure database is reachable and schema is correct."""
        try:
            self.connect()

            # Only verify schema once across all sessions.
            if verify_schema and not type(self)._schema_verified:
                with type(self)._schema_lock:
                    if not type(self)._schema_verified:
                        self._verify_schema()
                        type(self)._schema_verified = True
        except Exception as e:
            logger.error(f"Error ensuring database: {e}")
            raise

    def _verify_schema(self):
        """Verify and update database schema if needed."""
        try:
            # Keep this separate so we can show a targeted error message.
            try:
                self.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as extension_error:
                requirement_note = (
                    "PGVECTOR_REQUIRED=true" if PGVECTOR_REQUIRED else "PGVECTOR_REQUIRED=false"
                )
                raise RuntimeError(
                    "pgvector extension is required by the current schema but could not be created. "
                    "Use a PostgreSQL instance that supports pgvector, then retry. "
                    f"Current setting: {requirement_note}."
                ) from extension_error

            # Execute schema creation/update.
            self.execute(DB_SCHEMA)
            self.commit()

            # Verify tables exist.
            self.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = {row['table_name'] for row in self.fetchall()}

            if not REQUIRED_TABLES.issubset(tables):
                missing = REQUIRED_TABLES - tables
                raise Exception(f"Missing required tables: {missing}")

            logger.info("Database schema verified successfully")
        except Exception as e:
            logger.error(f"Error verifying schema: {e}")
            raise

    def connect(self):
        """Establish database connection."""
        retries = int(os.getenv("DB_CONNECT_RETRIES", "5"))
        delay_seconds = float(os.getenv("DB_CONNECT_RETRY_DELAY", "1.0"))
        if "dsn" not in DB_CONFIG:
            missing = [key for key in ("dbname", "user", "password") if not DB_CONFIG.get(key)]
            if missing:
                missing_keys = ", ".join(missing)
                raise ValueError(
                    f"Missing database configuration ({missing_keys}). "
                    "Set DATABASE_URL or POSTGRES_* variables."
                )

        for attempt in range(1, retries + 1):
            try:
                self.conn = psycopg2.connect(**DB_CONFIG)
                self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
                logger.info("Successfully connected to database")
                return
            except OperationalError as e:
                if attempt == retries:
                    logger.error(
                        "Failed to connect to database after %s attempt(s): %s",
                        retries,
                        e,
                    )
                    raise
                logger.warning(
                    "Database connection attempt %s/%s failed: %s. Retrying in %.1fs...",
                    attempt,
                    retries,
                    e,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                raise

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def commit(self):
        """Commit the current transaction."""
        if self.conn:
            self.conn.commit()
            logger.debug("Transaction committed")

    def rollback(self):
        """Rollback the current transaction."""
        if self.conn:
            self.conn.rollback()
            logger.debug("Transaction rolled back")

    def execute(self, query: str, params: tuple = None):
        """Execute a query with optional parameters."""
        try:
            self.cursor.execute(query, params)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.rollback()
            raise

    def execute_values(self, query: str, values: list):
        """Execute a query with multiple values."""
        try:
            execute_values(self.cursor, query, values)
        except Exception as e:
            logger.error(f"Error executing values query: {e}")
            self.rollback()
            raise

    def fetchall(self) -> list:
        """Fetch all results from the last query."""
        return self.cursor.fetchall()

    def fetchone(self) -> dict:
        """Fetch one result from the last query."""
        return self.cursor.fetchone()


@contextmanager
def get_db_session() -> Generator[DatabaseSession, None, None]:
    """Context manager for database sessions."""
    session = DatabaseSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
