# test.py
import argparse
import os
import torch
import numpy as np
from dataset import PelletDataset
from model import CSRNet
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model, loader, device):
    model.eval()
    mae = 0.0
    mse = 0.0
    with torch.no_grad():
        for imgs, densities, fnames in tqdm(loader):
            imgs = imgs.to(device).float()
            preds = model(imgs)
            pred_counts = preds.detach().cpu().sum(dim=(1,2,3)).numpy()
            gt_counts = densities.detach().cpu().sum(dim=(1,2,3)).numpy()
            err = pred_counts - gt_counts
            mae += np.abs(err).sum()
            mse += (err**2).sum()
    mae = mae / len(loader.dataset)
    rmse = np.sqrt(mse / len(loader.dataset))
    return mae, rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--ann_file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    dataset = PelletDataset(os.path.join(args.data_root,'images'), args.ann_file, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CSRNet(load_weights=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    mae, rmse = evaluate(model, loader, device)
    print(f"Test MAE: {mae:.3f}, RMSE: {rmse:.3f}")

if __name__ == '__main__':
    main()
