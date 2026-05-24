import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SingleInputDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DualInputDataset(Dataset):
    def __init__(self, sat_x: np.ndarray, met_x: np.ndarray, y: np.ndarray):
        self.sat_x = torch.tensor(sat_x, dtype=torch.float32)
        self.met_x = torch.tensor(met_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.sat_x[idx], self.met_x[idx], self.y[idx]


def _cosine_warmup_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float) -> float:
    """Linear warmup → cosine decay 学习率调度。"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_single_input_torch(model, x_train, y_train, x_val, y_val, epochs, batch_size,
                             lr, wd, patience, device, warmup_epochs: int = 0):
    tr_loader = DataLoader(SingleInputDataset(x_train, y_train),
                           batch_size=min(batch_size, len(y_train)), shuffle=True)
    va_loader = DataLoader(SingleInputDataset(x_val, y_val),
                           batch_size=min(batch_size, len(y_val)), shuffle=False)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.SmoothL1Loss()
    best_state, best_val, bad = None, float("inf"), 0
    model.to(device)

    for epoch in range(epochs):
        # Warmup + cosine schedule
        if warmup_epochs > 0:
            current_lr = _cosine_warmup_lr(epoch, warmup_epochs, epochs, lr)
            for pg in optim.param_groups:
                pg["lr"] = current_lr

        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()

        model.eval()
        vloss, count = 0.0, 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                lv = criterion(model(x), y).item()
                bs = len(y)
                vloss += lv * bs
                count += bs
        vloss /= max(count, 1)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_dual_input_torch(model, sat_train, met_train, y_train, sat_val, met_val, y_val, epochs, batch_size, lr, wd, patience, device):
    tr_loader = DataLoader(DualInputDataset(sat_train, met_train, y_train), batch_size=min(batch_size, len(y_train)), shuffle=True)
    va_loader = DataLoader(DualInputDataset(sat_val, met_val, y_val), batch_size=min(batch_size, len(y_val)), shuffle=False)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.SmoothL1Loss()
    best_state, best_val, bad = None, float("inf"), 0
    model.to(device)

    for _ in range(epochs):
        model.train()
        for s, m, y in tr_loader:
            s, m, y = s.to(device), m.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(s, m), y)
            loss.backward()
            optim.step()

        model.eval()
        vloss, count = 0.0, 0
        with torch.no_grad():
            for s, m, y in va_loader:
                s, m, y = s.to(device), m.to(device), y.to(device)
                lv = criterion(model(s, m), y).item()
                bs = len(y)
                vloss += lv * bs
                count += bs
        vloss /= max(count, 1)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_single_input_torch(model, x, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    return pred


def predict_dual_input_torch(model, sat_x, met_x, device):
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(sat_x, dtype=torch.float32, device=device),
            torch.tensor(met_x, dtype=torch.float32, device=device),
        ).cpu().numpy()
    return pred
