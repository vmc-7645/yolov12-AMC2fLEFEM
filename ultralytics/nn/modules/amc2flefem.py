"""
AMC2fLEFEM
==========

A lazy-constructed “C2f + LEF + FEM + SimAM” block that keeps the incoming
channel width for every YOLOv12 scale (n/s/m/l/x).  The first forward pass
discovers the true `in_channels`, builds all internal layers to that size,
stores `self.out_channels`, and then behaves like a normal module.

Author: Vincent C May 2025
License: AGPL-3.0 (inherits Ultralytics license)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv  # Ultralytics helper conv

# ------------------------------------------------------------------------- #
# Helper blocks                                                             #
# ------------------------------------------------------------------------- #


class LEF(nn.Module):
    """Lightweight Explicit Feature aggregation: four global pools."""

    def __init__(self) -> None:
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,H,W]
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, 4C, H, W).
        """
        h, w = x.shape[2:]
        p1 = F.interpolate(self.pool1(x), (h, w), mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.pool2(x), (h, w), mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.pool3(x), (h, w), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.pool4(x), (h, w), mode="bilinear", align_corners=False)
        return torch.cat([p1, p2, p3, p4], dim=1)  # [B,4C,H,W]


class FEM(nn.Module):
    """Fuse the original and LEF tensors with a 1x1 conv."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c * 5, c, 1, bias=False)

    def forward(self, x: torch.Tensor, lef: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).
            lef (torch.Tensor): LEF tensor of shape (B, C2, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        return self.conv(torch.cat([x, lef], dim=1))


class SimAM(nn.Module):
    """Parameter-free 3-D attention (CVPR 2021)."""

    def __init__(self, lam: float = 1e-4) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        n = x.shape[2] * x.shape[3] - 1
        x_mu = x.mean(dim=[2, 3], keepdim=True)
        var = ((x - x_mu) ** 2).sum(dim=[2, 3], keepdim=True) / n
        e_inv = (x - x_mu) ** 2 / (4 * (var + self.lam)) + 0.5
        return x * torch.sigmoid(e_inv)


# ------------------------------------------------------------------------- #
# Lazy C2f‑LEF‑FEM                                                          #
# ------------------------------------------------------------------------- #


class C2fLEFEM(nn.Module):
    """
    Creates convolutions *after* seeing the first input.  Keeps the same
    number of channels (`out_c == in_c`) so parse_model can treat the
    module like an identity in terms of channel width.
    """

    def __init__(self) -> None:
        super().__init__()
        self.built = False  # flag for runtime construction

    # ---- helper --------------------------------------------------------- #
    def _build(self, in_c: int) -> None:
        """Allocate real layers matching the incoming channel count."""
        out_c = in_c
        self.proj = nn.Identity()
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.conv3 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.lef = LEF()
        self.fem = FEM(out_c)
        self.out_channels = out_c  # <-- important for Ultralytics export
        self.built = True

    # ---- forward -------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        if not self.built:  # first time only
            self._build(x.size(1))
        x = self.act(self.bn(self.proj(x)))
        c2f = self.conv3(x)
        return self.fem(c2f, self.lef(c2f))


# ------------------------------------------------------------------------- #
# Public wrapper with SimAM                                                #
# ------------------------------------------------------------------------- #

# ultralytics/nn/modules/amc2flefem.py
class AMC2fLEFEM(nn.Module):
    """AMC2fLEFEM module with SimAM attention."""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1)
        self.lef = LEF()
        self.fem = FEM(c_)
        self.attn = SimAM()
        self.cv3 = Conv(c_, c2, 1, 1)  # ← force compressed width
        self.add = shortcut and c1 == c2
        self.out_channels = c2

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        y = self.cv2(self.cv1(x))
        y = self.fem(y, self.lef(y))
        y = self.attn(y)
        y = self.cv3(y)
        return x + y if self.add else y
