# logs/view_logs.py
import os

LOG_PATH = "events.log"

def tail_log(n=20):
    """Print last n lines from events.log."""
    if not os.path.exists(LOG_PATH):
        print("Log file not found:", LOG_PATH)
        return
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()
        for line in lines[-n:]:
            print(line.strip())

if __name__ == "__main__":
    tail_log(20)
