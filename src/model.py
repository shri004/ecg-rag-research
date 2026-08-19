"""
ResNet1D-Wang for ECG classification.

Faithful reproduction of the ResNet architecture from:
  Wang, Z., Yan, W., Oates, T. (2017). "Time Series Classification from
  Scratch with Deep Neural Networks: A Strong Baseline." IJCNN 2017.
  arXiv:1611.06455

This is the architecture Strodthoff et al. (2021, IEEE JBHI) benchmark as
"resnet1d_wang" on PTB-XL, where it scores macro-AUC 0.930 (+/-0.05) on the
diagnostic-superclass task -- our target baseline
(github.com/helme/ecg_ptbxl_benchmarking, leaderboard table 4).

Verified architecture spec (cross-checked against the original paper text
and an independent reproduction in the R `sits` package, both citing Wang
et al. 2017 directly):
  - 3 residual blocks
  - Per-block filter counts: {64, 128, 128}
  - Within each block, 3 sequential conv layers with kernel sizes [8, 5, 3]
  - Each conv followed by BatchNorm + ReLU
  - No dropout, no intermediate dense layers (this is a lean baseline)
  - Residual/shortcut connection from block input to block output
    (1x1 conv projection when channel counts differ)
  - Global average pooling after the final block, then a single linear
    classifier head

Adaptation for ECG (standard, and how Strodthoff et al. also adapt it):
  - Input channels = 12 (leads) instead of the original paper's univariate
    (1-channel) time series input. Everything else is unchanged.

Input shape: (batch, channels=12, time_steps)  -- 12-lead ECG
Output: (batch, num_classes) logits, one sigmoid per class (multi-label)
"""
import torch
import torch.nn as nn


class WangResBlock(nn.Module):
    """One residual block: 3 convs with kernel sizes [8, 5, 3], BN+ReLU each,
    shortcut connection from block input to block output."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=8, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv3 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 projection on the shortcut when channel counts change
        # (standard for this architecture family, e.g. He et al. 2016)
        self.shortcut = None
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # kernel_size=8 with padding=4 adds one extra timestep -- trim to match
        if out.shape[-1] != identity.shape[-1]:
            out = out[..., : identity.shape[-1]]
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.relu(out + identity)


class ResNet1DWang(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, block_filters=(64, 128, 128)):
        super().__init__()
        blocks = []
        in_ch = in_channels
        for out_ch in block_filters:
            blocks.append(WangResBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(block_filters[-1], num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)  # logits -- apply sigmoid outside for multi-label BCE


# Kept as an alias so existing training code (train.py) doesn't need renaming.
ResNet1d = ResNet1DWang


if __name__ == "__main__":
    # Quick shape sanity check
    model = ResNet1DWang(in_channels=12, num_classes=5)
    dummy = torch.randn(4, 12, 1000)  # batch=4, 12 leads, 1000 timesteps @100Hz/10s
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (4, 5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
    print("Target baseline (Strodthoff et al. 2021, resnet1d_wang, "
          "PTB-XL diagnostic superclass task): macro-AUC 0.930 (+/-0.05)")
