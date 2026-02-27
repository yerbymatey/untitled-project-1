from db.session import get_db_session
from utils.config import DB_SCHEMA, REQUIRED_TABLES


def setup_database():
    """Set up the complete database schema including pgvector support."""
    with get_db_session() as session:
        session.execute(DB_SCHEMA)
        session.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = {row["table_name"] for row in session.fetchall()}
        missing_tables = REQUIRED_TABLES - tables
        if missing_tables:
            raise RuntimeError(f"Database schema setup incomplete. Missing: {missing_tables}")

        session.commit()
        print("✓ Database schema is set up")


def main():
    """Main function to run schema checks and setup."""
    setup_database()
    print("\n✓ Database is correctly configured")


if __name__ == "__main__":
    main()
