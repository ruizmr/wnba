"""train.py
Simple training loop example that shows how to plug in CalibratedKellyLoss
and compute Sharpe / Sortino / drawdown on validation betting returns.
This is **not** the full Agent-1 training script; it is a reference
implementation that other agents can adapt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import CalibratedKellyLoss
from .metrics import sharpe_ratio, sortino_ratio, max_drawdown


def fake_dataset(n=1024):
    # toy binary classification with odds ~U[1.5,3]
    rng = np.random.default_rng(42)
    p_true = rng.uniform(0.2, 0.8, size=n)
    y = rng.binomial(1, p_true)
    odds = rng.uniform(1.5, 3.0, size=n)
    x = rng.normal(size=(n, 4))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y), torch.tensor(odds)


def build_model(input_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )


def train_epoch(model, loader, criterion, opt):
    model.train()
    total_loss = 0.0
    for xb, yb, ob in loader:
        opt.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.float(), ob.float())
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    preds_list, y_list, odds_list = [], [], []
    with torch.no_grad():
        for xb, yb, ob in loader:
            preds = model(xb).squeeze()
            preds_list.append(preds)
            y_list.append(yb)
            odds_list.append(ob)
    p = torch.cat(preds_list)
    y = torch.cat(y_list).float()
    odds = torch.cat(odds_list).float()

    # compute simulated returns with Kelly fraction (no clipping)
    k = (p * odds - 1) / (odds - 1)
    k = torch.clamp(k, 0, 1)  # long-only for demo
    rtn = y * (k * (odds - 1)) - (1 - y) * k  # per-sample percentage return

    sharpe = sharpe_ratio(rtn)
    sortino = sortino_ratio(rtn)
    cum = (rtn + 1).cumprod(0)
    mdd = max_drawdown(cum)
    loss = criterion(p, y, odds).item()
    return {
        "loss": loss,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lambda_kelly", type=float, default=0.05)
    args = parser.parse_args()

    x, y, odds = fake_dataset()
    ds = TensorDataset(x, y, odds)
    train_loader = DataLoader(ds, batch_size=128, shuffle=True)

    model = build_model(x.shape[1])
    criterion = CalibratedKellyLoss(lambda_kelly=args.lambda_kelly)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, opt)
        metrics = evaluate(model, train_loader, criterion)
        print(
            f"Epoch {epoch}: loss={loss:.4f} eval_loss={metrics['loss']:.4f} "
            f"Sharpe={metrics['sharpe']:.2f} Sortino={metrics['sortino']:.2f} "
            f"MDD={metrics['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()