"""SolarTrans forecaster, reimplemented from Tables 1-2 of Siddiqa et al. (2025).

- Encoder input (B, 48, 8) and decoder input (B, 8, 4), each linearly projected to d_model = 64
- Learnable positional encodings
- Transformer core: 4 encoder + 4 decoder layers, 8 heads, FFN 2048, dropout 0.1
- Causal mask on decoder self-attention
- Head: LayerNorm -> Linear + ReLU + Dropout -> Linear(1)
"""

import torch
from torch import nn


class SolarTrans(nn.Module):
    def __init__(self, n_enc=8, n_dec=4, d_model=64, nhead=8, layers=4,
                 ff=2048, dropout=0.1, enc_len=48, dec_len=8):
        super().__init__()
        self.enc_proj = nn.Linear(n_enc, d_model)
        self.dec_proj = nn.Linear(n_dec, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, enc_len, d_model) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(1, dec_len, d_model) * 0.02)
        self.core = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=layers, num_decoder_layers=layers,
            dim_feedforward=ff, dropout=dropout, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        mask = nn.Transformer.generate_square_subsequent_mask(dec_len)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        src = self.enc_proj(x_enc) + self.enc_pos
        tgt = self.dec_proj(x_dec) + self.dec_pos
        h = self.core(src, tgt, tgt_mask=self.causal_mask)
        return self.head(h).squeeze(-1)  # (B, 8)
