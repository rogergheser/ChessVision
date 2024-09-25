import gdown
import tarfile
import os

# Google Drive file ID from the tar file's shareable link
file_id = "17JB6Gs1zAyzGcsdJgqlqwyjR0OXZye6W"  # Update this with the actual tar file ID
output_filename = "dataset.tar.gz"  # Set the filename for download

# Set the root destination folder where the files will be extracted
root_dst_folder = "data"

# Construct Google Drive download URL
drive_link = f"https://drive.google.com/uc?id={file_id}"

# Download the file from Google Drive
print(f"Downloading dataset from Google Drive...")
gdown.download(drive_link, output_filename, quiet=False)

# Extract the tar file
print(f"Extracting {output_filename} to {root_dst_folder}...")
with tarfile.open(output_filename, "r:gz") as tar:
    tar.extractall(path=root_dst_folder)

# Remove the tar file after extraction
print(f"Removing {output_filename}...")
os.remove(output_filename)

print("Download, extraction, and cleanup completed successfully!")
