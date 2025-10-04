# main.py
import cv2
import os
import sqlite3
import csv
import json

from modules.detector import Detector
from modules.recognizer import Recognizer
from modules.tracker import SimpleTracker
from modules.logger import log_face_event, log_system
from modules.database import Database, init_db
from data.manager import DataManager
from modules.utils import load_config, crop_face, get_timestamp

# ---------------- Video Writer ---------------- #
class VideoWriter:
    def __init__(self, output_path, frame_size, fps=20):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

# ---------------- Report Generator ---------------- #
def export_reports(db_path="db/visitors.db", out_dir="outputs/reports"):
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Export visitors
    c.execute("SELECT * FROM visitors")
    visitors = c.fetchall()
    visitor_cols = [desc[0] for desc in c.description]
    with open(os.path.join(out_dir, "visitors.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(visitor_cols)
        writer.writerows(visitors)

    # Export events
    c.execute("SELECT * FROM events")
    events = c.fetchall()
    event_cols = [desc[0] for desc in c.description]
    with open(os.path.join(out_dir, "events.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(event_cols)
        writer.writerows(events)

    # Export summary JSON
    summary = {
        "unique_visitors": len(visitors),
        "total_events": len(events),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    conn.close()
    print(f"✅ Reports exported to {out_dir}")
    print(f"📊 Unique visitors: {len(visitors)}, Total events: {len(events)}")

# ---------------- Video Processing ---------------- #
def process_video(video_path, detector, recognizer, tracker, db, dm, cfg):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {video_path}")
        return

    # Setup video writer
    basename = os.path.basename(video_path)
    out_path = os.path.join("outputs", "processed_videos", f"processed_{basename}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = VideoWriter(out_path, (w, h), fps=fps)

    print(f"▶️ Processing {video_path} ...")
    log_system(f"Started processing {video_path}")

    frame_count = 0
    face_id_map = {}  # tracker_id -> db_face_id

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Skip frames for performance
        if frame_count % cfg.get("frame_skip", 5) != 0:
            continue

        # Step 1: Detect faces
        detections = detector.detect(frame)
        bboxes = [d["bbox"] for d in detections]

        # Step 2: Update tracker
        tracked_objects, exited_ids = tracker.update(bboxes)

        # Step 3: Handle exited objects
        for tid in exited_ids:
            if tid in face_id_map:
                fid = face_id_map[tid]
                log_face_event(fid, "exit", tracked_objects.get(tid, [0,0,1,1]), frame)
                del face_id_map[tid]

        # Step 4: Handle active tracked objects
        for tid, bbox in tracked_objects.items():
            face_img = crop_face(frame, bbox)
            emb = recognizer.get_embedding(face_img)
            if emb is None:
                continue

            if tid not in face_id_map:
                match_id, sim = db.find_match(emb, threshold=cfg.get("match_threshold", 0.6))
                if match_id:
                    face_id_map[tid] = match_id
                    log_system(f"Recognized existing face {match_id} (sim={sim:.2f})")
                else:
                    ts = get_timestamp()
                    face_path = dm.save_face(db.get_unique_count() + 1, face_img)
                    dm.save_embedding(db.get_unique_count() + 1, emb)
                    new_id = db.register_face(emb, face_path, ts)
                    face_id_map[tid] = new_id
                    log_face_event(new_id, "entry", bbox, frame)
                    log_system(f"Registered new face {new_id}")

        # Draw and save to processed video
        display = frame.copy()
        for tid, bbox in tracked_objects.items():
            x1, y1, x2, y2 = map(int, bbox)
            fid = face_id_map.get(tid, -1)
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(display, f"ID:{fid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        writer.write(display)

        # Optional live view (press q to quit)
        cv2.imshow("Face Tracker", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    log_system(f"Finished processing {video_path}")
    print(f"✅ Finished {video_path}")

# ---------------- Main ---------------- #
def main(video_folder="videos"):
    cfg = load_config()
    db_path = cfg.get("db_path", "db/visitors.db")
    init_db(db_path)

    # Init components
    detector = Detector()
    recognizer = Recognizer()
    tracker = SimpleTracker(
        max_disappeared=cfg.get("track_disappeared_frames", 30),
        distance_threshold=cfg.get("distance_threshold", 80),
    )
    db = Database(db_path)
    dm = DataManager()

    if not os.path.exists(video_folder):
        print(f"❌ Folder not found: {video_folder}")
        return

    # Loop through all video files
    for file in os.listdir(video_folder):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            process_video(os.path.join(video_folder, file), detector, recognizer, tracker, db, dm, cfg)

    cv2.destroyAllWindows()

    # Export reports
    export_reports(db_path=db_path, out_dir="outputs/reports")

if __name__ == "__main__":
    # Change "videos" to your folder containing videos
    main(video_folder="videos")