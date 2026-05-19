# Copyright 2025 Valeo.

"""SAE training loop.

Loads harvested activations from HDF5, mean-centers them, trains the
SparseAutoencoder, and saves the best checkpoint.

Usage:
    python -m sae_interpretability.sae_trainer \\
        --data harvest.h5 \\
        --expansion_factor 16 \\
        --l1_coeff 1e-3 \\
        --epochs 10 \\
        --output sae_model.pt
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sae_interpretability.config import SAEConfig
from sae_interpretability.sae_model import SparseAutoencoder


def train(
    data_path: str,
    output_path: str,
    cfg: SAEConfig,
    log_dir: str | None = None,
) -> SparseAutoencoder:
    """Train SAE on harvested activations.

    Args:
        data_path: Path to harvest.h5 produced by harvester.py.
        output_path: Where to save the best checkpoint (.pt).
        cfg: SAEConfig controlling architecture and training hyperparameters.
        log_dir: Optional directory for TensorBoardX event files.
                 Defaults to ``<output_dir>/tb_<stem>``.

    Returns:
        The best SparseAutoencoder loaded from the saved checkpoint.
    """
    from tensorboardX import SummaryWriter

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Trainer] Device: {device}")

    print(f"[Trainer] Loading activations from {data_path}")
    with h5py.File(data_path, 'r') as hf:
        activations = hf['activations'][:]  # [N, D]
        hidden_dim = int(hf['metadata'].attrs.get(
            'hidden_dim', cfg.wayformer_hidden_dim))

    N, D = activations.shape
    print(f"[Trainer] Dataset: {N:,} rows × {D} dims")

    # Mean-center: stored in checkpoint so annotation/steering can reproduce it
    act_mean = activations.mean(axis=0, keepdims=True).astype(np.float32)
    act_std  = activations.std(axis=0, keepdims=True).astype(np.float32) + 1e-8
    activations_centered = ((activations - act_mean) / act_std).astype(np.float32)

    print(
        f"[Trainer] Activation mean norm: {np.linalg.norm(act_mean):.4f} (subtracted)")

    x = torch.from_numpy(activations_centered)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=cfg.sae_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    latent_dim = hidden_dim * cfg.sae_expansion_factor
    model = SparseAutoencoder(
        hidden_dim, latent_dim,
        l1_coeff=cfg.sae_l1_coeff,
    ).to(device)
    activation_name = "ReLU"
    print(
        f"[Trainer] SAE: {hidden_dim}D → {latent_dim}D (×{cfg.sae_expansion_factor})  "
        f"activation={activation_name}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.sae_learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.sae_epochs)

    # TensorBoardX writer
    if log_dir is None:
        stem = os.path.splitext(os.path.basename(output_path))[0]
        log_dir = os.path.join(os.path.dirname(output_path) or '.', f'tb_{stem}')
    writer = SummaryWriter(logdir=log_dir)
    print(f"[Trainer] TensorBoard logs → {log_dir}")

    # Sample for dead-feature tracking (up to 8192 rows, kept on CPU)
    sample_x = x[:min(8192, N)]

    best_loss = float('inf')
    best_l0 = float('inf')
    best_dead_pct = float('inf')

    for epoch in range(1, cfg.sae_epochs + 1):
        model.train()
        total_recon = total_l1 = total_combined = total_l0 = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, features, loss = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.normalize_decoder()

            with torch.no_grad():
                recon = torch.nn.functional.mse_loss(x_hat, batch).item()
                l1 = features.abs().mean().item()
                l0 = (features > 0).float().sum(dim=-1).mean().item()

            total_recon += recon
            total_l1 += l1
            total_combined += loss.item()
            total_l0 += l0
            n_batches += 1

        scheduler.step()

        # Dead feature % on sample (features that never activate)
        model.eval()
        with torch.no_grad():
            _, samp_feats, _ = model(sample_x.to(device))
            dead_pct = 100.0 * \
                (1 - (samp_feats > 0).any(dim=0).float().mean().item())

        avg_loss = total_combined / n_batches
        avg_recon = total_recon / n_batches
        avg_l1 = total_l1 / n_batches
        avg_l0 = total_l0 / n_batches

        # -- TensorBoard logging (every epoch) --
        writer.add_scalar('loss/combined', avg_loss, epoch)
        writer.add_scalar('loss/reconstruction', avg_recon, epoch)
        writer.add_scalar('loss/l1_sparsity', avg_l1, epoch)
        writer.add_scalar('sparsity/L0', avg_l0, epoch)
        writer.add_scalar('sparsity/dead_pct', dead_pct, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_l0 = avg_l0
            best_dead_pct = dead_pct
            _save_checkpoint(model, cfg, output_path,
                             act_mean, act_std, epoch, best_loss, best_l0, best_dead_pct)

        # -- Console logging (every 10 epochs + first & last) --
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.sae_epochs:
            print(
                f"Epoch {epoch:3d}/{cfg.sae_epochs}  "
                f"loss={avg_loss:.5f}  "
                f"recon={avg_recon:.5f}  "
                f"l1={avg_l1:.5f}  "
                f"L0={avg_l0:.1f}  "
                f"dead={dead_pct:.1f}%"
            )

    writer.close()
    print(
        f"\n[Trainer] Best checkpoint → {output_path}  "
        f"(loss={best_loss:.5f}, L0={best_l0:.1f})")
    return SparseAutoencoder.from_checkpoint(output_path)


def _save_checkpoint(
    model: SparseAutoencoder,
    cfg: SAEConfig,
    path: str,
    act_mean: np.ndarray,
    act_std: np.ndarray,
    epoch: int,
    loss: float,
    best_l0: float = 0.0,
    best_dead_pct: float = 0.0,
):
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': {
                'input_dim':      model.input_dim,
                'latent_dim':     model.latent_dim,
                'l1_coeff':       model.l1_coeff,
            },
            'act_mean': act_mean,
            'act_std':  act_std,
            'epoch': epoch,
            'loss': loss,
            'best_l0': best_l0,
            'best_dead_pct': best_dead_pct,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train SAE on harvested Wayformer activations.")
    parser.add_argument("--data", required=True, help="Path to harvest.h5")
    parser.add_argument("--output", default="sae_model.pt")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--l1_coeff", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    cfg = SAEConfig(
        sae_expansion_factor=args.expansion_factor,
        sae_l1_coeff=args.l1_coeff,
        sae_epochs=args.epochs,
        sae_batch_size=args.batch_size,
        sae_learning_rate=args.lr,
    )
    train(args.data, args.output, cfg)


if __name__ == "__main__":
    main()
