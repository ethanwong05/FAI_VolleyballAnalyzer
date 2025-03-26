import os
import pandas as pd

# Paths (adjust base_dir if needed)
base_dir = "/Users/shiven/Downloads/VolleyballRallyAnalyzer/data"
videos_dir = os.path.join(base_dir, "videos_full")
annotations = []

print(f"Scanning: {videos_dir}")

for video_id in os.listdir(videos_dir):
    video_path = os.path.join(videos_dir, video_id)
    if not os.path.isdir(video_path):
        continue
    annot_file = os.path.join(video_path, "annotations.txt")
    if not os.path.exists(annot_file):
        print(f"No annotations.txt in {video_path}")
        continue

    with open(annot_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            frame_id = parts[0].replace('.jpg', '')  # e.g., "62645"
            activity = parts[1]  # e.g., "r-pass"
            outcome = 1 if "winpoint" in activity.lower() else 0
            player_data = parts[2:]
            print(f"Processing {video_id}/{frame_id}, activity: {activity}")

            frame_folder = os.path.join(video_path, frame_id)
            if not os.path.isdir(frame_folder):
                print(f"Frame folder missing: {frame_folder}")
                continue
            for i in range(-5, 5):  # 5 before, target, 4 after
                frame_num = int(frame_id) + i
                frame_file = f"{frame_num}.jpg"  # Matches dataset naming
                frame_path = os.path.join(frame_folder, frame_file)
                if not os.path.exists(frame_path):
                    print(f"Frame not found: {frame_path}")
                    continue

                for j in range(0, len(player_data) - 4, 5):
                    try:
                        x, y, w, h, action = player_data[j:j+5]
                        annotations.append([frame_path, video_id, frame_id, frame_num, x, y, w, h, action, activity, outcome])
                    except ValueError:
                        print(f"Malformed player data: {player_data[j:j+5]}")

df = pd.DataFrame(annotations, columns=["frame_path", "video_id", "frame_id", "frame_num", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "action", "activity", "outcome"])
df.to_csv(os.path.join(base_dir, "annotations.csv"), index=False)
print(f"Saved {len(df)} annotations")
