"""
models.py
=========
AutoEncoder1D — U-Net 1D CNN for ECoG → finger flex decoding.
  Input:  (B, C, F, T)
  Output: (B, n_out, T)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → LayerNorm → GELU → Dropout → MaxPool."""
    def __init__(self, in_ch, out_ch, kernel, stride=1, dilation=1, p_drop=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, bias=False, padding='same')
        self.norm = nn.LayerNorm(out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p=p_drop)
        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act(x)
        x = self.drop(x)
        return self.pool(x)


class UpConvBlock(nn.Module):
    """ConvBlock + linear upsample."""
    def __init__(self, scale, **kw):
        super().__init__()
        self.conv = ConvBlock(**kw)
        self.up   = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)

    def forward(self, x):
        return self.up(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# AutoEncoder1D  (U-Net style 1D CNN)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHANNELS  = [32, 32, 64, 64, 128, 128]
_DEFAULT_KERNELS   = [7, 7, 5, 5, 5]
_DEFAULT_STRIDES   = [2, 2, 2, 2, 2]
_DEFAULT_DILATIONS = [1, 1, 1, 1, 1]


class AutoEncoder1D(nn.Module):
    """
    1D CNN autoencoder with selective skip connections.

    skip_levels : set of decoder-step indices (0 = deepest/first decoder block)
                  that receive a skip connection from the matching encoder level.
                  None  → all levels active (full U-Net, default).
                  set() → no skips (pure bottleneck).
                  {0,1} → only the two deepest skips active (v5 — ablation showed
                           levels 2-4 contribute <2% each to val_corr).

    Encoder architecture is identical regardless of skip_levels, so weights
    transfer cleanly between differently-configured models.
    """
    def __init__(self, n_electrodes, n_freqs=40, n_out=5, skip_levels=None,
                 channels=None, kernel_sizes=None, strides=None, dilation=None):
        super().__init__()
        channels     = channels     or _DEFAULT_CHANNELS
        kernel_sizes = kernel_sizes or _DEFAULT_KERNELS
        strides      = strides      or _DEFAULT_STRIDES
        dilation     = dilation     or _DEFAULT_DILATIONS
        depth = len(channels) - 1

        self._strides   = strides
        self.depth      = depth
        # None → all levels; freeze into a frozenset for hashing / serialisation
        self.skip_levels = frozenset(range(depth)) if skip_levels is None else frozenset(skip_levels)
        n_inp           = n_freqs * n_electrodes

        self.spatial_reduce = ConvBlock(n_inp, channels[0], kernel=3)

        self.encoder = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1],
                      kernel_sizes[i], stride=strides[i], dilation=dilation[i])
            for i in range(depth)
        ])

        # Input channels for decoder block j depend on which previous levels had skips.
        # After decoder block j-1, if (j-1) in skip_levels the output was concatenated
        # with a same-width skip → channels doubled.  The first block always receives
        # the raw bottleneck (channels[depth]).
        dec_in = []
        for j in range(depth):
            if j == 0:
                dec_in.append(channels[depth])
            else:
                prev_out = channels[depth - j]          # out_ch of decoder block j-1
                multiplier = 2 if (j - 1) in self.skip_levels else 1
                dec_in.append(multiplier * prev_out)

        # Head receives the output of the last decoder block, possibly skip-concat'd.
        final_out = channels[0]
        head_ch   = 2 * final_out if (depth - 1) in self.skip_levels else final_out

        self.decoder = nn.ModuleList([
            UpConvBlock(scale=strides[i], in_ch=dec_in[depth - 1 - i],
                        out_ch=channels[i], kernel=kernel_sizes[i])
            for i in range(depth - 1, -1, -1)
        ])

        self.head = nn.Conv1d(head_ch, n_out, kernel_size=1, padding='same')

    @property
    def stride_multiple(self):
        m = 1
        for s in self._strides:
            m *= s
        return m

    def forward(self, x):
        B, C, n_f, T = x.shape
        x = x.reshape(B, C * n_f, T)
        x = self.spatial_reduce(x)

        skips = []
        for block in self.encoder:
            skips.append(x)
            x = block(x)

        for i, block in enumerate(self.decoder):
            x = block(x)
            if i in self.skip_levels:
                skip = skips[-(i + 1)]
                t = min(x.shape[-1], skip.shape[-1])
                x = torch.cat([x[..., :t], skip[..., :t]], dim=1)

        return self.head(x)
