# CSRNet Feed Pellet Counting

1. Put images in `data/images/`
2. Put annotation JSON in `data/annotations.json` (see README for supported formats)
3. Prepare folders:
   - data/images/
   - data/density/   (optional: density maps saved here)
   - checkpoints/
   - outputs/

Run training:
python train.py --data_root data --ann_file data/annotations.json --epochs 100 --batch_size 8

Test/evaluate:
python test.py --data_root data --ann_file data/annotations.json --checkpoint checkpoints/best.pth

Visualize:
python visualize.py --img_path data/images/example.jpg --checkpoint checkpoints/best.pth

📖 Step-by-Step Guide to Training CSRNet

Here’s the full workflow for your pellet counting model:

Step 1: Organize Data

Place images in:

data/pellets/images/


Place your annotation JSON in:

data/pellets/annotations.json

Step 2: Generate Density Maps + Split Dataset

Run:

python generate_density_maps.py


This will:

Create .npy density maps in data/pellets/density_maps/

Generate train_list.txt (80%) and val_list.txt (20%)

Step 3: Train CSRNet

Run:

python train.py


train.py will:

Load train_list.txt and val_list.txt

Train CSRNet using images and density maps

Save model checkpoints (e.g. best_model.pth)

Step 4: Validate Model

During training, validation runs automatically on the 20% split.

It helps you monitor loss and MAE (Mean Absolute Error).

Step 5: Test on New Images

After training:

python test.py --weights best_model.pth --image data/pellets/images/sample.jpg


Loads trained model

Predicts density map for input image

Sums pixel values → pellet count

Step 6: Evaluate Full Validation/Test Set (Optional)

Modify test.py to iterate over all images in val_list.txt and compare predicted vs ground truth pellet counts.