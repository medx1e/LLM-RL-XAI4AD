# Copyright 2025 Valeo.


"""Latent-Query Encoder Module.

Paper: https://arxiv.org/abs/2103.03206
"""

from functools import partial

import einops
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents import datatypes
from vmax.agents.networks import encoders


class LQAttention(nn.Module):
    """Latent-Query attention module.

    Applies cross- and self-attention with an optional weight sharing mechanism.
    Always returns a tuple of (latent, attention_weights_dict).

    Args:
        depth: Number of attention blocks.
        num_latents: Number of latent vectors.
        latent_num_heads: Number of self-attention heads.
        latent_head_features: Feature size per latent head.
        cross_num_heads: Number of cross-attention heads.
        cross_head_features: Feature size per cross-attention head.
        ff_mult: Multiplier for the feedforward layers.
        attn_dropout: Dropout probability in attention.
        ff_dropout: Dropout probability in feedforward blocks.
        tie_layer_weights: If True, layer weights are shared across blocks.

    """

    depth: int = 4
    num_latents: int = 64
    latent_num_heads: int = 2
    latent_head_features: int = 64
    cross_num_heads: int = 2
    cross_head_features: int = 64
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    tie_layer_weights: bool = False

    @nn.compact
    def __call__(self, x, mask=None):
        """Forward pass of the LQ attention module.

        Args:
            x: Input tensor.
            mask: Input mask.

        Returns:
            Tuple of (output tensor, attention_weights_dict).

        """
        bs, dim = x.shape[0], x.shape[-1]

        # Learnable latent feature
        latents = self.param("latents", init.normal(), (self.num_latents, dim * self.ff_mult))
        latent = einops.repeat(latents, "n d -> b n d", b=bs)

        attention_weights = {}

        # Cross, self attention and feedforward layers
        cross_attn = partial(
            encoders.AttentionLayer,
            heads=self.cross_num_heads,
            head_features=self.cross_head_features,
            dropout=self.attn_dropout,
        )
        self_attn = partial(
            encoders.AttentionLayer,
            heads=self.latent_num_heads,
            head_features=self.latent_head_features,
            dropout=self.attn_dropout,
        )
        ff = partial(encoders.FeedForward, mult=self.ff_mult, dropout=self.ff_dropout)

        # weights optionnaly shared between repeats - Page 2 paper Perceiver
        if self.tie_layer_weights:
            ca = cross_attn(name="cross_attn")
            sa = self_attn(name="self_attn")
            cf = ff(name="cross_ff")
            lf = ff(name="self_ff")
            for i in range(self.depth):
                rz = encoders.ReZero(name=f"rezero_cross_{i}")
                ca_out, ca_weights = ca(latent, x, mask_k=mask)
                attention_weights[f"cross_attn_{i}"] = ca_weights
                latent += rz(ca_out)
                latent += rz(cf(latent))
                rz = encoders.ReZero(name=f"rezero_self_{i}")
                sa_out, sa_weights = sa(latent)
                attention_weights[f"self_attn_{i}"] = sa_weights
                latent += rz(sa_out)
                latent += rz(lf(latent))
        else:
            # different weights for each attn block in the lq
            for i in range(self.depth):
                rz = encoders.ReZero(name=f"rezero_cross{i}")
                ca_out, ca_weights = cross_attn(name=f"cross_attn_{i}")(latent, x, mask_k=mask)
                attention_weights[f"cross_attn_{i}"] = ca_weights
                latent += rz(ca_out)
                latent += rz(ff(name=f"cross_ff_{i}")(latent))
                rz = encoders.ReZero(name=f"rezero_self_{i}")
                sa_out, sa_weights = self_attn(name=f"latent_attn_{i}")(latent)
                attention_weights[f"self_attn_{i}"] = sa_weights
                latent += rz(sa_out)
                latent += rz(ff(name=f"latent_ff_{i}")(latent))

        return latent, attention_weights


class LQEncoder(nn.Module):
    """Latent-Query encoder module.

    Encodes the input observations by building individual embeddings followed by the LQ attention module.
    Always returns a tuple of (output, attention_weights_dict).

    Args:
        unflatten_fn: Function to unflatten the input observations.
        embedding_layer_sizes: Sizes for MLP embedding layers.
        embedding_activation: Activation function for the embeddings.
        encoder_depth: Depth of the encoder.
        dk: Projection dimension.
        num_latents: Number of latent tokens.
        latent_num_heads: Number of latent self-attention heads.
        latent_head_features: Feature size per latent head.
        cross_num_heads: Number of cross-attention heads.
        cross_head_features: Feature size per cross-attention head.
        ff_mult: Multiplier for the feedforward network.
        attn_dropout: Dropout factor in attention.
        ff_dropout: Dropout factor in feedforward network.
        tie_layer_weights: If true, reuses weights across attention blocks.

    """

    unflatten_fn: callable = lambda x: x
    embedding_layer_sizes: tuple[int] = (256, 256)
    embedding_activation: datatypes.ActivationFn = nn.relu
    encoder_depth: int = 4
    dk: int = 64
    num_latents: int = 64
    latent_num_heads: int = 2
    latent_head_features: int = 64
    cross_num_heads: int = 2
    cross_head_features: int = 64
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    tie_layer_weights: bool = False
    attention_name: str = "lq_attention"

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """Forward pass of the LQ encoder.

        Args:
            obs: Input observation tensor.

        Returns:
            Tuple of (output tensor, attention_weights_dict).

        """
        # Get features and masks
        features, masks = self.unflatten_fn(obs)
        sdc_traj_features, other_traj_features, rg_features, tl_features, gps_path_features = features
        sdc_traj_valid_mask, other_traj_valid_mask, rg_valid_mask, tl_valid_mask = masks

        # Embeddings for all sub features
        num_objects, timestep_agent = other_traj_features.shape[-3:-1]
        num_roadgraph = rg_features.shape[-2]
        target_len = gps_path_features.shape[-2]
        num_light, timestep_tl = tl_features.shape[-3:-1]

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

        # Positional Encoding
        sdc_traj_encoding += jnp.expand_dims(self.param("sdc_traj_pe", init.normal(), (1, timestep_agent, self.dk)), 0)
        other_traj_encoding += jnp.expand_dims(
            self.param("other_traj_pe", init.normal(), (num_objects, timestep_agent, self.dk)),
            0,
        )
        rg_encoding += jnp.expand_dims(self.param("rg_pe", init.normal(), (num_roadgraph, self.dk)), 0)
        tl_encoding += jnp.expand_dims(self.param("tj_pe", init.normal(), (num_light, timestep_tl, self.dk)), 0)
        gps_path_encoding += jnp.expand_dims(self.param("gps_path_pe", init.normal(), (target_len, self.dk)), 0)

        # # Flatten by NumAgent NumObsTS , Feature_dim
        sdc_traj_encoding = einops.rearrange(sdc_traj_encoding, "b n t d -> b (n t) d")
        other_traj_encoding = einops.rearrange(other_traj_encoding, "b n t d -> b (n t) d")
        tl_encoding = einops.rearrange(tl_encoding, "b n t d -> b (n t) d")

        # Masks
        sdc_traj_valid_mask = einops.rearrange(sdc_traj_valid_mask, "b n t -> b (n t)")
        other_traj_valid_mask = einops.rearrange(other_traj_valid_mask, "b n t -> b (n t)")
        tl_valid_mask = einops.rearrange(tl_valid_mask, "b n t -> b (n t)")
        gps_path_mask = jnp.ones(gps_path_encoding.shape[:-1])

        # [B, N, D]
        input = jnp.concatenate(
            [sdc_traj_encoding, other_traj_encoding, rg_encoding, tl_encoding, gps_path_encoding],
            axis=1,
        )
        mask = jnp.concatenate(
            [
                sdc_traj_valid_mask,
                other_traj_valid_mask,
                rg_valid_mask,
                tl_valid_mask,
                gps_path_mask,
            ],
            axis=1,
        )  # [B, N]

        latent, attention_weights = LQAttention(
            depth=self.encoder_depth,
            num_latents=self.num_latents,
            latent_num_heads=self.latent_num_heads,
            latent_head_features=self.latent_head_features,
            cross_num_heads=self.cross_num_heads,
            cross_head_features=self.cross_head_features,
            ff_mult=self.ff_mult,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            tie_layer_weights=self.tie_layer_weights,
            name=self.attention_name,
        )(input, mask)

        output_encoded = latent.mean(axis=1)
        return output_encoded, attention_weights


class PerceiverEncoder(LQEncoder):
    """Perceiver encoder — identical to LQEncoder but with the original Flax module name.

    Pretrained Perceiver checkpoints store parameters under 'perceiver_attention/'
    instead of 'lq_attention/', so this subclass simply overrides the default name
    to ensure parameter compatibility.
    """

    attention_name: str = "perceiver_attention"
