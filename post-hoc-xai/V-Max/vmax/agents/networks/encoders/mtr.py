# Copyright 2025 Valeo.


"""Motion Transformer Encoder.

Paper: https://arxiv.org/abs/2209.13508
"""

from functools import partial

import einops
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents import datatypes
from vmax.agents.networks import encoders


class MTRAttention(nn.Module):
    """Motion Transformer Attention module.

    Applies local attention guided by nearest neighbor indices for motion data.

    Args:
        depth: Number of attention blocks.
        num_heads: Number of attention heads.
        head_features: Feature size per attention head.
        ff_mult: Multiplier for the feedforward layer.
        attn_dropout: Dropout probability for attention.
        ff_dropout: Dropout probability for feedforward.
        k: Number of neighbors to use.

    """

    depth: int = 4
    num_heads: int = 2
    head_features: int = 64
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    k: int = 8

    @nn.compact
    def __call__(self, latent, x, mask_latent=None, mask_x=None):
        """Compute the forward pass of the MTR attention module.

        Args:
            latent: Latent tensor.
            x: Input tensor.
            mask_latent: Mask for the latent tensor.
            mask_x: Mask for the input tensor.

        Returns:
            Updated latent tensor.

        """
        x = einops.rearrange(x, "b n ... -> b n (...)")
        latent = einops.rearrange(latent, "b n ... -> b n (...)")

        attn = partial(
            encoders.LocalAttentionLayer,
            heads=self.num_heads,
            head_features=self.head_features,
            dropout=self.attn_dropout,
        )
        ff = partial(encoders.FeedForward, mult=self.ff_mult, dropout=self.ff_dropout)

        knn = jax.vmap(lambda x, y, mask_y=None: encoders.nearest_neighbors_jax(x, y, self.k, mask_y))

        index_pairs = knn(latent, x, mask_y=mask_x)
        rz = encoders.ReZero(name="rezero_0")
        latent += rz(attn(name="attn_0")(latent, x, index_pairs, mask_q=mask_latent, mask_k=mask_x))

        for i in range(1, self.depth):
            rz = encoders.ReZero(name=f"rezero_{i}")
            index_pairs = knn(latent, x, mask_y=mask_x)
            latent += rz(attn(name=f"attn_{i}")(latent, x, index_pairs, mask_k=mask_x))
            latent += rz(ff(name=f"ff_{i}")(latent))

        return latent


class MTREncoder(nn.Module):
    """Motion Transformer Encoder module.

    Builds embeddings and applies MTR attention to encode the motion input.

    Args:
        unflatten_fn: Function to unflatten the input observation.
        embedding_layer_sizes: Sizes of the MLP embedding layers.
        embedding_activation: Activation function for the embeddings.
        encoder_depth: Depth of the encoder.
        num_latents: Number of latent tokens.
        latent_num_heads: Number of attention heads for latent space.
        latent_head_features: Feature size per latent head.
        ff_mult: Feedforward multiplier.
        ff_dropout: Dropout probability for feedforward.
        attn_dropout: Dropout probability for attention.
        dk: Projection dimension.
        k: Number of neighbors in local attention.

    """

    unflatten_fn: callable = lambda x: x
    embedding_layer_sizes: tuple[int] = (256, 256)
    embedding_activation: datatypes.ActivationFn = nn.relu
    encoder_depth: int = 4
    num_latents: int = 64
    latent_num_heads: int = 2
    latent_head_features: int = 16
    ff_mult: int = 4
    ff_dropout: float = 0.0
    attn_dropout: float = 0.0
    dk: int = 64
    k: int = 8

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """Forward pass of the MTR encoder.

        Args:
            obs: Input observation tensor.

        Returns:
            Encoded output tensor.

        """
        features, masks = self.unflatten_fn(obs)

        sdc_traj_features, other_traj_features, rg_features, tl_features, gps_path_features = features
        sdc_traj_valid_mask, other_traj_valid_mask, rg_valid_mask, tl_valid_mask = masks

        num_objects = other_traj_features.shape[-3]
        num_roadgraph = rg_features.shape[-2]
        target_len = gps_path_features.shape[-2]
        num_light = tl_features.shape[-3]

        # Latent encoding
        sdc_traj_encoding = encoders.build_mlp_embedding(
            sdc_traj_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "sdc_traj_enc",
        )
        other_traj_encoding = encoders.build_mlp_embedding(
            other_traj_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "other_traj_enc",
        )
        rg_encoding = encoders.build_mlp_embedding(
            rg_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "rg_enc",
        )
        tl_encoding = encoders.build_mlp_embedding(
            tl_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "tl_enc",
        )
        gps_path_encoding = encoders.build_mlp_embedding(
            gps_path_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "gps_path_enc",
        )

        # Max pooling - https://arxiv.org/pdf/2209.13508 - P4 for input representation
        # only keep max feature over the temporal sequence
        sdc_traj_encoding = nn.max_pool(
            sdc_traj_encoding - jnp.inf * ~jnp.expand_dims(sdc_traj_valid_mask, -1),
            (1, 1, sdc_traj_encoding.shape[2]),
        ).squeeze(2)  # [B,N,D]
        sdc_traj_encoding = jnp.where(sdc_traj_encoding == -jnp.inf, 0, sdc_traj_encoding)
        sdc_traj_valid_mask = jnp.max(sdc_traj_valid_mask, axis=-1)  # [B,N]

        other_traj_encoding = nn.max_pool(
            other_traj_encoding - jnp.inf * ~jnp.expand_dims(other_traj_valid_mask, -1),
            (1, 1, other_traj_encoding.shape[2]),
        ).squeeze(2)  # [B,N,D]
        other_traj_encoding = jnp.where(other_traj_encoding == -jnp.inf, 0, other_traj_encoding)
        other_traj_valid_mask = jnp.max(other_traj_valid_mask, axis=-1)  # [B,N]

        # Similarly for traffic lights - only keep max feature over the temporal sequence
        # For roadgraph points: only keep max feature over the number of road lanes sequence - closest lanes
        # NotImplemented for RG points
        tl_encoding = nn.max_pool(
            tl_encoding - jnp.inf * ~jnp.expand_dims(tl_valid_mask, -1),
            (1, 1, tl_encoding.shape[2]),
        ).squeeze(2)  # [B,N,D]
        tl_encoding = jnp.where(tl_encoding == -jnp.inf, 0, tl_encoding)
        tl_valid_mask = jnp.max(tl_valid_mask, axis=-1)  # [B,N]

        # Positional Encoding
        sdc_traj_encoding += jnp.expand_dims(self.param("sdc_traj_pe", init.normal(), (1, self.dk)), 0)
        other_traj_encoding += jnp.expand_dims(self.param("other_traj_pe", init.normal(), (num_objects, self.dk)), 0)
        rg_encoding += jnp.expand_dims(self.param("rg_pe", init.normal(), (num_roadgraph, self.dk)), 0)
        tl_encoding += jnp.expand_dims(self.param("tl_pe", init.normal(), (num_light, self.dk)), 0)
        gps_path_encoding += jnp.expand_dims(self.param("gps_path_pe", init.normal(), (target_len, self.dk)), 0)

        # Mask for gps path target
        gps_path_valid_mask = jnp.ones(gps_path_encoding.shape[:-1]).astype(bool)

        input = jnp.concatenate(
            [sdc_traj_encoding, other_traj_encoding, rg_encoding, tl_encoding, gps_path_encoding],
            axis=1,
        )
        input_mask = jnp.concatenate(
            [
                sdc_traj_valid_mask,
                other_traj_valid_mask,
                rg_valid_mask,
                tl_valid_mask,
                gps_path_valid_mask,
            ],
            axis=1,
        )

        # MTR attention between objects history and global input
        output = MTRAttention(
            depth=self.encoder_depth,
            num_heads=self.latent_num_heads,
            head_features=self.latent_head_features,
            ff_mult=self.ff_mult,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            k=self.k,
            name="mtr_attention",
        )(input, other_traj_encoding, mask_latent=input_mask, mask_x=other_traj_valid_mask)

        output = output.mean(axis=1)

        return output
