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

import csv

def extract_angle_ranges(directory, output_file):
    results = []
    # Support various extensions like .txt, .xy, .gk while ignoring hidden files
    valid_exts = ['.txt', '.xy', '.gk']
    files = [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in valid_exts) and not f.startswith('.')]
    files.sort()

    for filename in files:
        filepath = os.path.join(directory, filename)
        angles = []
        try:
            # Use errors='ignore' to handle non-UTF-8 characters in headers
            with open(filepath, 'r', errors='ignore') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            angle = float(parts[0])
                            angles.append(angle)
                        except ValueError:
                            continue
            if angles:
                results.append({
                    'filename': filename,
                    'min_angle': min(angles),
                    'max_angle': max(angles)
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'min_angle', 'max_angle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    spectra_dir = "./Spectra"
    output_path = "./angle_ranges.csv"
    extract_angle_ranges(spectra_dir, output_path)
    print(f"Results saved to {output_path}")
