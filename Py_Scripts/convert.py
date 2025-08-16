import os
import csv
from PIL import Image  # Pillow library

# --- CONFIGURATION ---
csv_input_path = r"C:\Users\WELCOME\Desktop\C++\image_labels.csv"  # path to CSV
output_csv_path = r"C:\Users\WELCOME\Desktop\C++\mnist_style.csv"  # output CSV path
image_size = (28, 28)  # target image size (MNIST standard)

# --- FUNCTION TO LOAD AND PROCESS IMAGE ---
def process_image(image_path):
    """
    Loads image, converts to grayscale, resizes to 28x28,
    and returns a flattened list of pixel values (0-255)
    """
    img = Image.open(image_path).convert("L")  # convert to grayscale
    img = img.resize(image_size)
    pixels = list(img.getdata())  # flatten to 1D list
    return pixels

# --- READ CSV AND PROCESS ---
with open(csv_input_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    rows = []
    for row in reader:
        if len(row) < 2:
            continue
        image_path, label = row[0].strip(), row[1].strip()
        # ensure full path if relative
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(csv_input_path), image_path)
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found: {image_path}")
            continue
        pixels = process_image(image_path)
        rows.append([label] + pixels)

# --- WRITE OUTPUT CSV ---
with open(output_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Optional: write header
    header = ["label"] + [f"{i//28 + 1}x{i%28 + 1}" for i in range(28*28)]
    writer.writerow(header)
    # Write all data rows
    writer.writerows(rows)

print(f"MNIST-style CSV saved at: {output_csv_path}")
