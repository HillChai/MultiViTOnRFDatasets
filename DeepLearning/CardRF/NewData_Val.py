import os
import random
import shutil

# Paths
source_dir = "/mnt/ssd/CardRFDataset/CardRF/LOS/Train/NewData"
destination_dir = "/mnt/ssd/CardRFDataset/CardRF/LOS/Test/NewData"

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Collect all `.npz` files
npz_files = []
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".npz"):
            npz_files.append(os.path.join(root, file))

# Calculate 20% of total files
num_to_select = int(0.2 * len(npz_files))
selected_files = random.sample(npz_files, num_to_select)

# Move selected files while preserving folder structure
for file in selected_files:
    relative_path = os.path.relpath(file, source_dir)  # Preserve subdir structure
    destination_path = os.path.join(destination_dir, relative_path)

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)  # Create subdirectories
    shutil.move(file, destination_path)  # Move file

print(f"✅ Moved {len(selected_files)} files to {destination_dir}")

