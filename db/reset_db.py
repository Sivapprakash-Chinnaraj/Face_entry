# db/reset_db.py
import os
from modules.database import init_db

DB_PATH = "visitors.db"

def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Old database removed.")
    init_db(DB_PATH)
    print("New database created at", DB_PATH)

if __name__ == "__main__":
    main()