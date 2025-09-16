import os
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import random

def generate_density_map(img, points, sigma=15):
    """
    Create a density map from point annotations.
    """
    density = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for x, y in points:
        if x < img.shape[1] and y < img.shape[0]:
            density[int(y), int(x)] += 1

    density = gaussian_filter(density, sigma=sigma, mode='constant')
    return density

def split_dataset(img_dir, train_file, val_file, split_ratio=0.8, seed=42):
    """
    Splits dataset into train/validation lists.
    """
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    random.seed(seed)
    random.shuffle(all_imgs)

    train_size = int(split_ratio * len(all_imgs))
    train_imgs = all_imgs[:train_size]
    val_imgs = all_imgs[train_size:]

    with open(train_file, "w") as f:
        f.write("\n".join(train_imgs))
    with open(val_file, "w") as f:
        f.write("\n".join(val_imgs))

    print(f"✅ Dataset split: {len(train_imgs)} train, {len(val_imgs)} val")

if __name__ == "__main__":
    img_dir = "data/pellets/images"
    ann_file = "data/pellets/annotations.json"
    den_dir = "data/pellets/density_maps"

    os.makedirs(den_dir, exist_ok=True)

    # Load annotations
    with open(ann_file, "r") as f:
        annotations = json.load(f)

    # Generate density maps
    for img_name, ann in annotations.items():
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping {img_name}, not found")
            continue

        # Extract pellet coordinates
        points = []
        for region in ann["regions"]:
            cx = region["shape_attributes"]["cx"]
            cy = region["shape_attributes"]["cy"]
            points.append((cx, cy))

        density = generate_density_map(img, points)

        # Save density map
        out_path = os.path.join(den_dir, img_name.replace(".jpg", ".npy"))
        np.save(out_path, density)

        print(f"✅ Density map saved for {img_name}")

    # After density maps, split dataset
    split_dataset(
        img_dir,
        "data/pellets/train_list.txt",
        "data/pellets/val_list.txt"
    )
