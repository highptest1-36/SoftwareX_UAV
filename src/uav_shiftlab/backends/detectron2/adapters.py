# SPDX-License-Identifier: Apache-2.0
"""Lightweight residual FPN adapters (the only trainable parameters during test-time adaptation).

F' = F + W_up * relu(W_down * F), 1x1 convs 256 -> r -> 256, W_up = 0 at init (exact identity).
"""
import torch
from torch import nn

FPN_LEVELS = ("p2", "p3", "p4", "p5", "p6")


class ResidualAdapter(nn.Module):
    def __init__(self, channels=256, rank=32):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1)
        self.up = nn.Conv2d(rank, channels, 1)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))


class FPNAdapters(nn.Module):
    def __init__(self, levels=FPN_LEVELS, channels=256, rank=32, feat_dropout=0.0):
        super().__init__()
        self.adapters = nn.ModuleDict({k: ResidualAdapter(channels, rank) for k in levels})
        self.feat_dropout = feat_dropout

    def forward(self, feats):
        out = {}
        for k, v in feats.items():
            if self.training and self.feat_dropout > 0:
                v = nn.functional.dropout(v, self.feat_dropout)
            out[k] = self.adapters[k](v) if k in self.adapters else v
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
