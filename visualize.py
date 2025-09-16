# visualize.py
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import CSRNet
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CSRNet(load_weights=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    img = Image.open(args.img_path).convert('RGB')
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).cpu().squeeze().numpy()

    # sum equals predicted count (maybe after downsample compensation)
    pred_count = pred.sum()
    print("Predicted count:", pred_count)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(pred, cmap='jet')
    plt.title(f'Density Map (sum={pred_count:.2f})')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
