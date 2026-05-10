"""
Step 3: EEGNet Architecture
Implements EEGNet-8,2 from Lawhern et al., J. Neural Engineering 2018
DOI: 10.1088/1741-2552/aace8c

Only 2,548 parameters — trains in minutes on CPU.
Designed specifically for EEG: temporal conv → depthwise spatial → separable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet-8,2: Compact CNN for EEG-based BCIs.

    Args:
        n_classes   : number of output classes (2 for binary, 4 for BCI Comp IV)
        n_channels  : number of EEG channels (64 for PhysioNet, 22 for BCI Comp IV)
        n_samples   : number of time samples per epoch (160 Hz × 4s = 640)
        F1          : number of temporal filters (default 8)
        D           : depth multiplier for depthwise conv (default 2)
        F2          : number of pointwise filters = F1 × D (default 16)
        dropout_rate: dropout probability (default 0.5)
    """

    def __init__(
        self,
        n_classes: int   = 2,
        n_channels: int  = 64,
        n_samples: int   = 641,
        F1: int          = 8,
        D: int           = 2,
        F2: int          = 16,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_classes  = n_classes
        self.n_channels = n_channels
        self.n_samples  = n_samples
        F2 = F1 * D     # always F1×D per paper

        # ── Block 1: Temporal + Depthwise Spatial ──────────────────────────
        # Temporal convolution: learns frequency filters
        # kernel=(1, 64) at 160Hz ≈ 0.4s → captures mu/beta oscillations
        self.conv1 = nn.Conv2d(
            1, F1,
            kernel_size=(1, 64),
            padding=(0, 32),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise convolution: learns spatial EEG filters (like CSP)
        # Groups=F1 means each temporal filter gets its own spatial filter
        self.conv2 = nn.Conv2d(
            F1, F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2    = nn.BatchNorm2d(F1 * D)
        self.pool1  = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1  = nn.Dropout(dropout_rate)

        # ── Block 2: Separable Convolution ─────────────────────────────────
        # Separable = depthwise + pointwise (cheaper than full conv)
        self.conv3_dw = nn.Conv2d(
            F2, F2,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F2,
            bias=False,
        )
        self.conv3_pw = nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # ── Classifier ─────────────────────────────────────────────────────
        # Compute flattened size dynamically
        self._flat_size = self._get_flat_size(n_channels, n_samples, F1, D)
        self.fc = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, n_channels, n_samples, F1, D) -> int:
        """Run a dummy forward pass to compute flattened feature size."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            x = self._forward_features(x, F1, D)
            return x.shape[1]

    def _forward_features(self, x, F1=8, D=2):
        F2 = F1 * D
        # Block 1
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = F.elu(x)
        x = self.drop1(self.pool1(x))
        # Block 2
        x = self.conv3_dw(x)
        x = self.bn3(self.conv3_pw(x))
        x = F.elu(x)
        x = self.drop2(self.pool2(x))
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG tensor of shape (batch, 1, n_channels, n_samples)
        Returns:
            logits of shape (batch, n_classes)
        """
        x = self._forward_features(x)
        return self.fc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ShallowConvNet(nn.Module):
    """
    Baseline: ShallowConvNet from Schirrmeister et al. 2017.
    Simpler than EEGNet but useful as a comparison.
    """

    def __init__(self, n_classes=2, n_channels=64, n_samples=641):
        super().__init__()
        self.conv1  = nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False)
        self.conv2  = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False)
        self.bn     = nn.BatchNorm2d(40)
        self.pool   = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop   = nn.Dropout(0.5)

        flat = self._get_flat_size(n_channels, n_samples)
        self.fc = nn.Linear(flat, n_classes)

    def _get_flat_size(self, n_channels, n_samples):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool(torch.square(x))
            return x.flatten(1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.square(x)
        x = torch.log(torch.clamp(self.pool(x), min=1e-7))
        x = self.drop(x)
        return self.fc(x.flatten(1))


if __name__ == "__main__":
    # Sanity check: verify shapes and param counts
    n_ch, n_t = 64, 641

    eegnet = EEGNet(n_classes=2, n_channels=n_ch, n_samples=n_t)
    shallow = ShallowConvNet(n_classes=2, n_channels=n_ch, n_samples=n_t)

    dummy = torch.zeros(16, 1, n_ch, n_t)  # batch of 16

    print("=" * 45)
    print("EEGNet")
    print(f"  Input:      {tuple(dummy.shape)}")
    print(f"  Output:     {tuple(eegnet(dummy).shape)}")
    print(f"  Parameters: {eegnet.count_parameters():,}")

    print("\nShallowConvNet")
    print(f"  Input:      {tuple(dummy.shape)}")
    print(f"  Output:     {tuple(shallow(dummy).shape)}")
    print(f"  Parameters: {sum(p.numel() for p in shallow.parameters()):,}")
    print("=" * 45)
    print("Both models OK")