from contextlib import contextmanager
from typing import Generator
import threading

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from db.schema import get_schema_statements
from utils.config import DB_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseSession:
    _schema_verified = False
    _schema_lock = threading.Lock()
    
    def __init__(self):
        self.conn = None
        self.cursor = None
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure database exists and schema is correct"""
        try:
            self.connect()
            
            # Only verify schema once across all sessions
            if not self._schema_verified:
                with self._schema_lock:
                    if not self._schema_verified:
                        self._verify_schema()
                        self._schema_verified = True
                        
        except psycopg2.OperationalError as e:
            if "database" in str(e).lower() and "does not exist" in str(e).lower():
                # Database doesn't exist, create it
                self._create_database()
            else:
                raise
        except Exception as e:
            logger.error(f"Error ensuring database: {e}")
            raise
    
    def _create_database(self):
        """Create the database if it doesn't exist"""
        config = DB_CONFIG.copy()
        db_name = config.pop('dbname')
        
        try:
            conn = psycopg2.connect(**config)
            conn.autocommit = True
            cursor = conn.cursor()

            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            exists = cursor.fetchone() is not None
            
            if not exists:
                logger.info(f"Creating database '{db_name}'...")
                cursor.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Database '{db_name}' created successfully")
            
            cursor.close()
            conn.close()

            self.connect()
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    def _verify_schema(self):
        """Verify and update database schema if needed"""
        try:
            # Execute schema creation/update
            for statement in get_schema_statements():
                self.execute(statement)
            self.commit()
            
            # Verify tables exist
            self.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = {row['table_name'] for row in self.fetchall()}
            
            required_tables = {
                'users', 'tweets', 'hashtags', 'tweet_hashtags',
                'urls', 'media', 'user_description_urls'
            }
            
            if not required_tables.issubset(tables):
                missing = required_tables - tables
                raise Exception(f"Missing required tables: {missing}")
            
            logger.info("Database schema verified successfully")
            
        except Exception as e:
            logger.error(f"Error verifying schema: {e}")
            raise
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Successfully connected to database")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def commit(self):
        """Commit the current transaction"""
        if self.conn:
            self.conn.commit()
            logger.debug("Transaction committed")
    
    def rollback(self):
        """Rollback the current transaction"""
        if self.conn:
            self.conn.rollback()
            logger.debug("Transaction rolled back")
    
    def execute(self, query: str, params: tuple = None):
        """Execute a query with optional parameters"""
        try:
            self.cursor.execute(query, params)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.rollback()
            raise
    
    def execute_values(self, query: str, values: list):
        """Execute a query with multiple values"""
        try:
            execute_values(self.cursor, query, values)
        except Exception as e:
            logger.error(f"Error executing values query: {e}")
            self.rollback()
            raise
    
    def fetchall(self) -> list:
        """Fetch all results from the last query"""
        return self.cursor.fetchall()
    
    def fetchone(self) -> dict:
        """Fetch one result from the last query"""
        return self.cursor.fetchone()

@contextmanager
def get_db_session() -> Generator[DatabaseSession, None, None]:
    """Context manager for database sessions"""
    session = DatabaseSession()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close() 
