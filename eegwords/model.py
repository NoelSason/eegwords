"""EEGcnn + classifier head, copied from upstream vendor/chisco/{eegcnn,EEGclassify}.py (MIT).

Changes, none of which alter the function computed: the unused
`transformers.WhisperForConditionalGeneration` import is dropped; `timestamp` is derived from the
input length; the two 1xK temporal convolutions use `SameConv1xK` (identical parameters and
init, faster forward on MPS); the all-ones padding mask upstream passes is omitted.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SameConv1xK(nn.Conv2d):
    """Drop-in for `nn.Conv2d(cin, cout, (1, k), padding="same", bias=False)` with groups in
    {1, cin==cout}. Same parameters, same init, same RNG use; forward via unfold + einsum,
    which is ~9x faster on Apple MPS than its conv kernel for these shapes. Outputs match the
    conv2d reference to float32 rounding (tests/test_model.py)."""

    def forward(self, x):  # (N, C, H, T)
        k = self.kernel_size[1]
        left = (k - 1) // 2  # PyTorch 'same' puts the extra pad on the right for even k
        p = F.pad(x, (left, k - 1 - left)).unfold(-1, k, 1)  # (N, C, H, T, k)
        w = self.weight[:, :, 0, :]  # (cout, cin/groups, k)
        if self.groups == 1:
            return torch.einsum("nchtk,ock->noht", p, w)
        assert self.groups == self.in_channels == self.out_channels
        return torch.einsum("nchtk,ck->ncht", p, w[:, 0, :])


class EEGcnn(nn.Module):
    def __init__(self, Chans=64, dropoutRate=0.5, kernLength1=100, kernLength2=50, F1=8, D=2,
                 F2=16, P1=2, P2=5, dropoutType="Dropout"):
        super().__init__()
        drop = nn.Dropout(dropoutRate) if dropoutType == "Dropout" else nn.Dropout2d(dropoutRate)
        self.block1 = nn.Sequential(
            SameConv1xK(1, F1, (1, kernLength1), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            drop,
        )
        drop2 = nn.Dropout(dropoutRate) if dropoutType == "Dropout" else nn.Dropout2d(dropoutRate)
        self.block2 = nn.Sequential(
            SameConv1xK(D * F1, D * F1, (1, kernLength2), groups=D * F1, padding="same", bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            drop2,
        )

    def forward(self, x):  # (N, Chans, Samples)
        x = torch.unsqueeze(x, dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return torch.squeeze(x, dim=2)  # (N, F2, Samples//10)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


def n_tokens(n_times: int) -> int:
    """Tokens after the two average-pools (P1=2, P2=5)."""
    return (n_times // 2) // 5


class ChiscoNet(nn.Module):
    """Upstream `EEGclassification`. Defaults below are the paper's reproduction command:
    --layer 1 --pooling mean --dropout1 0.5 --dropout2 0.5 --feel1 20 --feel2 10 --cls 39."""

    def __init__(self, chans: int, n_times: int, cls: int = 39, dropout1=0.5, dropout2=0.5,
                 layer=1, pooling="mean", size1=8, size2=8, feel1=20, feel2=10):
        super().__init__()
        self.timestamp = n_tokens(n_times)
        self.eegcnn = EEGcnn(Chans=chans, kernLength1=feel1, kernLength2=feel2, F1=size1, D=size2,
                             F2=size1 * size2, P1=2, P2=5, dropoutRate=dropout1)
        d = size1 * size2
        self.pooling = pooling
        self.layer = layer
        self.linear = nn.Linear(self.timestamp * d if pooling is None else d, cls)
        if layer > 0:
            self.poscode = PositionalEncoding(d, dropout=dropout2, max_len=self.timestamp)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d, nhead=d // 8, dim_feedforward=4 * d,
                                           batch_first=True, dropout=dropout2),
                num_layers=layer)

    def forward(self, x):
        hidden = self.eegcnn(x).permute(0, 2, 1)  # (N, tokens, d)
        if self.layer > 0:
            hidden = self.encoder(self.poscode(hidden))  # mask is all ones upstream
        if self.pooling is None:
            hidden = torch.flatten(hidden, start_dim=1)
        elif self.pooling == "mean":
            hidden = hidden.mean(dim=1)
        elif self.pooling == "sums":
            hidden = hidden.sum(dim=1)
        elif self.pooling == "tops":
            hidden = hidden[:, 0, :]
        return self.linear(hidden)
