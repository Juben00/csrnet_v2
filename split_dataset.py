import os
import random

def split_dataset(img_dir, train_file, val_file, split_ratio=0.8, seed=42):
    """
    Splits dataset into train/validation lists.

    Args:
        img_dir (str): Path to images folder
        train_file (str): Output file for training image names
        val_file (str): Output file for validation image names
        split_ratio (float): Ratio of training images (default 0.8)
        seed (int): Random seed for reproducibility
    """
    # Collect all .jpg images
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    
    random.seed(seed)
    random.shuffle(all_imgs)

    # Split into train/val
    train_size = int(split_ratio * len(all_imgs))
    train_imgs = all_imgs[:train_size]
    val_imgs = all_imgs[train_size:]

    # Save file lists
    with open(train_file, "w") as f:
        f.write("\n".join(train_imgs))
    with open(val_file, "w") as f:
        f.write("\n".join(val_imgs))

    print(f"✅ Split complete: {len(train_imgs)} train, {len(val_imgs)} val")

if __name__ == "__main__":
    img_dir = "data/pellets/images"
    train_file = "data/pellets/train_list.txt"
    val_file = "data/pellets/val_list.txt"

    split_dataset(img_dir, train_file, val_file)
