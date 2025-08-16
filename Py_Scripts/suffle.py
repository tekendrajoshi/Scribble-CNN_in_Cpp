import pandas as pd

# Load CSV
df = pd.read_csv("mnist_style.csv")

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save back to new CSV
df.to_csv("image_labels_shuffled.csv", index=False)

print("Shuffled CSV saved as image_labels_shuffled.csv")
