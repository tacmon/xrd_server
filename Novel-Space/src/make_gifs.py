import os
import sys

# Set working directory to project root for easy path access
# Get the absolute path of the directory containing this script (src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Novel-Space/)
base_dir = os.path.dirname(script_dir)
# Change the current working directory to Novel-Space/
os.chdir(base_dir)
# Add the project root to sys.path so autoXRD can be imported
root_dir = os.path.dirname(base_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import imageio.v2 as imageio
import re

def numerical_sort(value):
    """
    Split the filename into parts of strings and numbers.
    This allows numerical sorting of filenames (e.g., 1.png, 2.png, ..., 10.png).
    """
    parts = re.split(r'(\d+)', value)
    return [int(text) if text.isdigit() else text.lower() for text in parts]

def create_gif(directory, output_filename, duration):
    """
    Create a GIF from images in a directory.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    images = []
    # Get all .png files in the directory
    filenames = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=numerical_sort)
    
    if not filenames:
        print(f"No image files found in {directory}.")
        return

    from PIL import Image
    print(f"Creating {output_filename} from {len(filenames)} images in {directory} (duration={duration}s)...")
    
    # Load all images with PIL
    pil_images = []
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        pil_images.append(Image.open(filepath))
    
    if not pil_images:
        return

    # Convert duration from seconds to milliseconds for Pillow
    duration_ms = int(duration * 1000)
    
    # The first image acts as the base, the rest are appended
    pil_images[0].save(
        output_filename,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"Successfully saved {output_filename} with frame duration {duration_ms}ms")

if __name__ == "__main__":
    base_dir = "figure/real_data"
    
    # Task (1): 0.25s per frame
    create_gif(os.path.join(base_dir, "参考"), os.path.join(base_dir, "gif/参考.gif"), duration=0.3)
    
    # Task (2): 0.1s per frame
    create_gif(os.path.join(base_dir, "AlN"), os.path.join(base_dir, "gif/AlN.gif"), duration=0.15)
    
    # Task (3): 0.25s per frame
    create_gif(os.path.join(base_dir, "BST"), os.path.join(base_dir, "gif/BST.gif"), duration=0.15)
    
    # Task (4): 0.25s per frame
    create_gif(os.path.join(base_dir, "CST"), os.path.join(base_dir, "gif/CST.gif"), duration=0.15)
