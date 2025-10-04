# logs/clean_logs.py
import os
import shutil

BASE_DIR = "logs"

def clean_logs():
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
        print("Logs folder cleared.")
    os.makedirs(BASE_DIR, exist_ok=True)
    open(os.path.join(BASE_DIR, "events.log"), "w").close()
    print("Logs folder reset with new events.log")

if __name__ == "__main__":
    clean_logs()
