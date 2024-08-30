from huggingface_hub import snapshot_download
import os
import shutil

# Replace with the name of your model repository
model_name = "ChallengerSpaceShuttle/continued-trained-gemma2-2b"

# Use the provided local directory
local_dir = snapshot_download(model_name)
# local_dir = "/home/motsepe-jr/.cache/huggingface/hub/models--ChallengerSpaceShuttle--continued-trained-gemma2-2b/snapshots/7b0d31bf88b0134909dc82ab58346981acba90e9"

print(f"Model files located at: {local_dir}")

# Create a new directory in the current working directory to store the copied files
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "../checkpoints/google/gemma-2-2b")
os.makedirs(output_dir, exist_ok=True)

print(f"Copying files to: {output_dir}")

# Copy files to the current directory
for root, dirs, files in os.walk(local_dir):
    for file in files:
        src_path = os.path.join(root, file)
        dst_path = os.path.join(output_dir, file)
        
        print(f"Copying: {file}")
        print(f"  From: {src_path}")
        print(f"  To: {dst_path}")
        
        if os.path.islink(src_path):
            # If it's a symlink, copy the target file
            target = os.readlink(src_path)
            absolute_target = os.path.join(os.path.dirname(src_path), target)
            shutil.copy2(absolute_target, dst_path)
        else:
            # If it's a regular file, copy it directly
            shutil.copy2(src_path, dst_path)
        
        print(f"  Copied successfully.")
        print()

print(f"All files have been copied to: {output_dir}")