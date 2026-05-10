"""
Step 4: Dataset class + Training loop
- EEGDataset: wraps numpy arrays into PyTorch Dataset
- train_loso: Leave-One-Subject-Out cross-validation
  (train on 8 subjects, test on 1 — repeat 9 times)
- Saves best model checkpoint per fold
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from src.model import EEGNet, ShallowConvNet


# ── Dataset ────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """Simple wrapper: X (n_epochs, n_ch, n_t) + y (n_epochs,)"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).unsqueeze(1)  # add channel dim → (N,1,C,T)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training utilities ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        total += len(y_batch)

    acc   = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm    = confusion_matrix(all_labels, all_preds)
    return total_loss / total, acc, kappa, cm


# ── LOSO Cross-Validation ───────────────────────────────────────────────────

def train_loso(
    data_path: str  = "data/processed/all_subjects.pkl",
    n_epochs: int   = 150,
    batch_size: int = 32,
    lr: float       = 1e-3,
    dropout: float  = 0.5,
    ckpt_dir: str   = "checkpoints",
    model_name: str = "eegnet",
):
    """Leave-One-Subject-Out cross-validation."""
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    X_all, y_all, subj_all = data["X"], data["y"], data["subjects"]
    unique_subjects = np.unique(subj_all)
    n_classes  = len(np.unique(y_all))
    n_channels = X_all.shape[1]
    n_samples  = X_all.shape[2]

    print(f"Loaded: {X_all.shape} | {n_classes} classes | {len(unique_subjects)} subjects")
    print(f"LOSO CV: training {len(unique_subjects)} folds...\n")

    fold_results = []
    criterion = nn.CrossEntropyLoss()

    for fold, test_subj in enumerate(unique_subjects):
        print(f"── Fold {fold+1}/{len(unique_subjects)} | Test subject: {test_subj:02d} ──")

        # Split
        test_mask  = subj_all == test_subj
        train_mask = ~test_mask
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]

        train_ds = EEGDataset(X_tr, y_tr)
        test_ds  = EEGDataset(X_te, y_te)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # Model
        if model_name == "eegnet":
            model = EEGNet(
                n_classes=n_classes,
                n_channels=n_channels,
                n_samples=n_samples,
                dropout_rate=dropout,
            ).to(device)
        else:
            model = ShallowConvNet(
                n_classes=n_classes,
                n_channels=n_channels,
                n_samples=n_samples,
            ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_acc, best_kappa = 0.0, -1.0
        ckpt_path = os.path.join(ckpt_dir, f"fold_{test_subj:02d}.pt")

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
            te_loss, te_acc, te_kappa, cm = evaluate(model, test_dl, criterion, device)
            scheduler.step()

            if te_acc > best_acc:
                best_acc   = te_acc
                best_kappa = te_kappa
                torch.save(model.state_dict(), ckpt_path)

            if epoch % 25 == 0 or epoch == n_epochs:
                print(f"  Epoch {epoch:3d}/{n_epochs} | "
                      f"train_acc {tr_acc:.3f} | "
                      f"test_acc {te_acc:.3f} | "
                      f"kappa {te_kappa:.3f}")

        fold_results.append({
            "subject": test_subj,
            "best_acc": best_acc,
            "best_kappa": best_kappa,
            "cm": cm,
        })
        print(f"  Best → acc {best_acc:.3f} | kappa {best_kappa:.3f}\n")

    # Summary
    accs   = [r["best_acc"]   for r in fold_results]
    kappas = [r["best_kappa"] for r in fold_results]
    print("=" * 50)
    print(f"LOSO Results ({model_name.upper()})")
    print(f"  Mean accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Mean kappa    : {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
    print("=" * 50)

    # Save summary
    summary_path = os.path.join(ckpt_dir, "results_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(fold_results, f)

    return fold_results


if __name__ == "__main__":
    results = train_loso(
        data_path  = "data/processed/all_subjects.pkl",
        n_epochs   = 150,
        batch_size = 32,
        lr         = 1e-3,
        model_name = "eegnet",
    )