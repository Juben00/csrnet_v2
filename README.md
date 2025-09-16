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
