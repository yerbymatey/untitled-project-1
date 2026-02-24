from db.session import get_db_session

def add_extr_text_column():
    with get_db_session() as session:
        try:
            # Check if the column already exists
            session.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'media' AND column_name = 'extr_text'
            """)
            result = session.fetchone()
            
            if result:
                print("The 'extr_text' column already exists in the media table.")
            else:
                # Add the extr_text column
                session.execute("""
                    ALTER TABLE media
                    ADD COLUMN extr_text TEXT
                """)
                session.commit()
                print("Successfully added 'extr_text' column to the media table.")
                
            # Verify the schema
            session.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'media'
                ORDER BY ordinal_position
            """)
            columns = session.fetchall()
            print("\nUpdated media table columns:")
            for col in columns:
                print(f"  - {col['column_name']} ({col['data_type']})")
                
        except Exception as e:
            print(f"Error updating media table schema: {e}")
            session.rollback()

if __name__ == "__main__":
    add_extr_text_column()