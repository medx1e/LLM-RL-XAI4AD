# Copyright 2025 Valeo.

"""Sparse Autoencoder architecture (PyTorch).

Follows the Anthropic convention:
- Pre-encoder bias subtraction centers the residual stream before encoding.
- Encoder: Linear(D, F) + JumpReLU(threshold).
- Decoder: Linear(F, D) with unit-norm row constraint (each row = one feature direction).
- Loss: MSE(x, x_hat) + l1_coeff * mean(|f|).

Features are activated using standard ReLU.

Trained offline on harvested activations; weights are exported to numpy for
JAX-side causal steering in causal_steering.py.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for extracting monosemantic features from a residual stream.

    Args:
        input_dim:       Residual stream dimension D.
        latent_dim:      SAE latent dimension F (= D × expansion_factor).
        l1_coeff:        Sparsity penalty weight λ.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        l1_coeff: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.l1_coeff = l1_coeff

        # Pre-encoder bias: learned center of the residual stream distribution
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # Encoder weights: [D, F]
        self.W_enc = nn.Parameter(torch.empty(input_dim, latent_dim))
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))

        # Decoder weights: [F, D] — each row is the feature direction for feature j
        self.W_dec = nn.Parameter(torch.empty(latent_dim, input_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            # Initialize decoder as transpose of encoder, then normalize rows
            self.W_dec.data = self.W_enc.data.T.clone()
            self._normalize_decoder_inplace()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of residual stream vectors to sparse features.

        Applies ReLU activation.

        Args:
            x: Input tensor [B, D].

        Returns:
            Non-negative feature activations [B, F].
        """
        pre_act = (x - self.pre_bias) @ self.W_enc + self.b_enc
        return F.relu(pre_act)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to residual stream space.

        Args:
            f: Feature activations [B, F].

        Returns:
            Reconstructed residual stream [B, D].
        """
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, features, and combined loss.

        Args:
            x: Input tensor [B, D].

        Returns:
            Tuple of (x_hat, features, loss) where loss = MSE + l1_coeff * L1.
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        recon_loss = F.mse_loss(x_hat, x)
        # L1 norm per sample (sum over features), averaged over batch.
        # Using per-sample sum (not mean over all B×F) so that l1_coeff controls
        # the total activation budget per sample, not a diluted per-element average.
        # f >= 0 after ReLU so abs() is redundant.
        l1_loss = f.sum(dim=-1).mean()
        loss = recon_loss + self.l1_coeff * l1_loss
        return x_hat, f, loss

    @torch.no_grad()
    def _normalize_decoder_inplace(self):
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms

    def normalize_decoder(self):
        """Normalize decoder rows to unit norm. Call after each optimizer step."""
        self._normalize_decoder_inplace()

    def to_numpy_weights(self) -> Dict[str, "np.ndarray"]:
        """Export all weights as numpy arrays for JAX-side inference."""
        import numpy as np
        return {
            'W_enc':    self.W_enc.detach().cpu().numpy(),       # [D, F]
            'b_enc':    self.b_enc.detach().cpu().numpy(),       # [F]
            'W_dec':    self.W_dec.detach().cpu().numpy(),       # [F, D]
            'b_dec':    self.b_dec.detach().cpu().numpy(),       # [D]
            'pre_bias': self.pre_bias.detach().cpu().numpy(),    # [D]
        }

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str = 'cpu') -> SparseAutoencoder:
        """Load a saved checkpoint."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt['config']
        model = cls(
            input_dim=cfg['input_dim'],
            latent_dim=cfg['latent_dim'],
            l1_coeff=cfg['l1_coeff'],
        )
        model.load_state_dict(ckpt['model_state_dict'])
        return model
