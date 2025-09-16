# train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PelletDataset
from model import CSRNet
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, densities, _ in tqdm(loader, desc="Train"):
        imgs = imgs.to(device).float()
        densities = densities.to(device).float()
        preds = model(imgs)
        # preds and densities shapes should match (1xHxW)
        loss = criterion(preds, densities)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    mae = 0.0
    mse = 0.0
    with torch.no_grad():
        for imgs, densities, _ in tqdm(loader, desc="Val"):
            imgs = imgs.to(device).float()
            preds = model(imgs)
            # sum counts
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
    parser.add_argument('--data_root', default='data', help='root folder with images')
    parser.add_argument('--ann_file', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--downsample', type=int, default=1, help='if you downsample density in dataset to match model out (e.g. 8)')
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    import numpy as np

    # transforms (simple)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = PelletDataset(os.path.join(args.data_root,'images'), args.ann_file,
                            transform=None, adaptive_sigma=True, downsample=args.downsample)
    # split
    n = len(dataset)
    nval = max(1, int(n * args.val_split))
    ntrain = n - nval
    train_set, val_set = torch.utils.data.random_split(dataset, [ntrain, nval])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CSRNet(load_weights=True).to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mae = 1e9
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        mae, rmse = validate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_mae={mae:.3f} val_rmse={rmse:.3f}")

        # save
        ckpt = os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'mae': mae}, ckpt)
        if mae < best_mae:
            best_mae = mae
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'mae': mae}, os.path.join(args.checkpoint_dir, 'best.pth'))

if __name__ == '__main__':
    main()
