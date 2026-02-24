from db.session import get_db_session

def check_media_table_schema():
    with get_db_session() as session:
        try:
            session.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'media'
            """)
            columns = session.fetchall()
            print("Media table columns:")
            for col in columns:
                print(f"  - {col['column_name']} ({col['data_type']})")
                
            # Also check if there are any rows
            session.execute("SELECT COUNT(*) FROM media")
            count = session.fetchone()
            print(f"\nTotal rows in media table: {count['count'] if count else 0}")
            
        except Exception as e:
            print(f"Error checking media table schema: {e}")

if __name__ == "__main__":
    check_media_table_schema()