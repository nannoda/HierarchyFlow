import os
import sys
import random

def find_png_files(directory, ratio=0.8):
    """Recursively finds all .png files in the given directory and splits them into train and test sets."""
    png_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                abs_path = os.path.abspath(os.path.join(root, file))
                png_files.append(abs_path)
    
    if not png_files:
        print("No PNG files found.")
        return
    
    random.shuffle(png_files)
    split_idx = int(len(png_files) * ratio)
    train_files = png_files[:split_idx]
    test_files = png_files[split_idx:]
    
    with open("train.txt", "w") as train_f:
        train_f.write("\n".join(train_files) + "\n")
    
    with open("test.txt", "w") as test_f:
        test_f.write("\n".join(test_files) + "\n")
    
    print(f"Train paths saved to train.txt ({len(train_files)} files)")
    print(f"Test paths saved to test.txt ({len(test_files)} files)")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <directory_path> [ratio]")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print("Error: Provided path is not a directory.")
        sys.exit(1)
    
    ratio = float(sys.argv[2]) if len(sys.argv) == 3 else 0.8
    if not (0 < ratio < 1):
        print("Error: Ratio must be between 0 and 1.")
        sys.exit(1)
    
    find_png_files(directory_path, ratio)