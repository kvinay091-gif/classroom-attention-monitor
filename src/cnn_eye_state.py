"""
Small CNN for binary eye-state classification (open vs. closed), trained on
24x24 grayscale eye crops (compatible with datasets like MRL Eye or CEW).

This is an OPTIONAL, more robust alternative to the geometric EAR heuristic
in face_analysis.py -- useful when lighting/glasses/camera angle make EAR
unreliable. It's disabled by default (config.USE_CNN_EYE_STATE = False)
because it needs a trained weights file you provide/train yourself.

Train it with `python -m src.cnn_eye_state train --data-dir <path>` where
<path> contains open/ and closed/ subfolders of eye crops.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import config


class EyeStateCNN(nn.Module):
    """Tiny CNN: 3 conv blocks -> 2 FC layers -> 2-class logits (open/closed)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 24 -> 12 -> 6 -> 3
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # index 0 = closed, index 1 = open


class EyeCropDataset(Dataset):
    """Expects data_dir/open/*.png and data_dir/closed/*.png."""

    def __init__(self, data_dir: str):
        self.samples = []
        for label, sub in enumerate(["closed", "open"]):
            folder = Path(data_dir) / sub
            if not folder.exists():
                continue
            for p in folder.glob("*"):
                self.samples.append((str(p), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (config.CNN_EYE_IMG_SIZE, config.CNN_EYE_IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        return tensor, label


class EyeStateClassifier:
    """Inference wrapper used by the live pipeline."""

    def __init__(self, model_path: str = config.CNN_EYE_MODEL_PATH, device: str = "cpu"):
        self.device = device
        self.model = EyeStateCNN().to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(
                f"No trained weights at {model_path}. Train first with "
                f"`python -m src.cnn_eye_state train --data-dir <your_dataset>`."
            )
        self.model.eval()

    @torch.no_grad()
    def predict_closed(self, eye_crop_bgr: np.ndarray) -> bool:
        """Returns True if the eye crop is classified as CLOSED."""
        gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (config.CNN_EYE_IMG_SIZE, config.CNN_EYE_IMG_SIZE))
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        logits = self.model(tensor.to(self.device))
        pred = torch.argmax(logits, dim=1).item()
        return pred == 0  # 0 = closed


def train(data_dir: str, epochs: int = 15, batch_size: int = 64, lr: float = 1e-3,
          out_path: str = config.CNN_EYE_MODEL_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EyeCropDataset(data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found under {data_dir}/open or {data_dir}/closed")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = EyeStateCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        print(f"epoch {epoch+1}/{epochs}  loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved trained weights to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--data-dir", required=True)
    train_p.add_argument("--epochs", type=int, default=15)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.cmd == "train":
        train(args.data_dir, args.epochs, args.batch_size, args.lr)
