import os
import pandas as pd

base_dir = "/Users/shiven/Downloads/VolleyballRallyAnalyzer/data"
df = pd.read_csv(os.path.join(base_dir, "annotations.csv"))
df = df.sort_values(['video_id', 'frame_num'])

rally_id = 0
rally_start = 0
rally_ids = []

for i, row in df.iterrows():
    if row['outcome'] == 1 and i > rally_start:
        rally_ids.extend([rally_id] * (i - rally_start + 1))
        rally_start = i + 1
        rally_id += 1
    elif i == len(df) - 1:
        rally_ids.extend([rally_id] * (i - rally_start + 1))

df['rally_id'] = rally_ids
df.to_csv(os.path.join(base_dir, "annotations.csv"), index=False)
print(f"Segmented {rally_id + 1} rallies")
