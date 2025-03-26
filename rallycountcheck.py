import pandas as pd

df = pd.read_csv("/Users/shiven/Downloads/VolleyballRallyAnalyzer/data/annotations.csv")
print(f"Total rallies: {len(df['rally_id'].unique())}")
print(f"Rows per rally (avg): {len(df) / len(df['rally_id'].unique())}")
