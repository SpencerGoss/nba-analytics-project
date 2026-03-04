"""
Package data for Google Colab training.

Creates nba_data.zip from the data/ folder so you can upload it to Google Drive
and use it in the Colab training notebook.

Usage:
    python scripts/package_for_colab.py
"""

import os
import shutil
import sys

# Make sure we're running from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

data_dir = os.path.join(project_root, "data")
zip_name = "nba_data"
zip_path = os.path.join(project_root, f"{zip_name}.zip")

if not os.path.isdir(data_dir):
    print("ERROR: data/ folder not found. Run this script from the project root.")
    sys.exit(1)

print("Packaging data/ folder for Colab...")
print()

# Show what's being zipped
total_size = 0
file_count = 0
for dirpath, dirnames, filenames in os.walk(data_dir):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        size = os.path.getsize(fp)
        total_size += size
        file_count += 1

print(f"  Files: {file_count}")
print(f"  Total size (uncompressed): {total_size / (1024**2):.0f} MB")
print()

# Create the zip
print("Creating zip file...")
shutil.make_archive(
    os.path.join(project_root, zip_name),
    "zip",
    project_root,
    "data"
)

zip_size = os.path.getsize(zip_path)
print(f"  Created: {zip_path}")
print(f"  Zip size: {zip_size / (1024**2):.0f} MB")
print()

print("=" * 50)
print("NEXT STEPS")
print("=" * 50)
print()
print("1. Copy nba_data.zip to your Google Drive root folder")
print("   (just drag it into drive.google.com)")
print()
print("2. Open the Colab notebook:")
print("   notebooks/train_on_colab.ipynb")
print()
print("3. Select a GPU runtime (T4 is free)")
print()
print("4. Run the cells in order — the notebook will")
print("   pull the zip from your Drive automatically")
print()
