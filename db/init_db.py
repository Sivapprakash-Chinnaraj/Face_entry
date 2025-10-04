# db/init_db.py
from modules.database import init_db

def main():
    db_path = "db/visitors.db"
    init_db(db_path)
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    main()
