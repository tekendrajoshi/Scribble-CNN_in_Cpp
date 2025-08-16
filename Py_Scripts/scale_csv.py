import pandas as pd

# Load CSV
df = pd.read_csv("suffled_data.csv")

# Convert all values > 0 to 1 (except for the first column if it is image paths)
# Assuming first column is image paths, rest are numeric labels
for col in df.columns[1:]:
    df[col] = df[col].apply(lambda x: 1 if x > 0 else x)

# Save modified CSV
df.to_csv("scaled_data.csv", index=False)

print("Modified CSV saved as scaled_data.csv")
