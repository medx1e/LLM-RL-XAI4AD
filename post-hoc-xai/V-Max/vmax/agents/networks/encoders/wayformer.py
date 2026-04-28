# Copyright 2025 Valeo.

"""Wayformer encoder module.

Paper: https://arxiv.org/abs/2207.05844
"""

from functools import partial

import einops
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents import datatypes
from vmax.agents.networks import encoders


class WayformerAttention(nn.Module):
    """Wayformer attention module.

    This module performs attention using a latent representation and cross attention.

    Args:
        depth: Number of layers.
        num_latents: Number of latent vectors.
        num_heads: Number of attention heads.
        head_features: Feature size per head.
        ff_mult: Feedforward multiplier.
        attn_dropout: Dropout factor in attention.
        ff_dropout: Dropout factor in feedforward network.

    """

    depth: int = 2
    num_latents: int = 32
    num_heads: int = 2
    head_features: int = 16
    ff_mult: int = 1
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None):
        """Forward pass of the Wayformer attention module.

        Args:
            x: Input tensor.
            mask: Input mask.

        Returns:
            Output tensor.

        """
        bs, dim = x.shape[0], x.shape[-1]
        latents = self.param("latents", init.normal(), (self.num_latents, dim * self.ff_mult))
        latent = einops.repeat(latents, "n d -> b n d", b=bs)
        x = einops.rearrange(x, "b n ... -> b n (...)")

        attn = partial(
            encoders.AttentionLayer,
            heads=self.num_heads,
            head_features=self.head_features,
            dropout=self.attn_dropout,
        )
        ff = partial(encoders.FeedForward, mult=self.ff_mult, dropout=self.ff_dropout)
        rz = encoders.ReZero(name="rezero_0")
        latent += rz(attn(name="attn_0")(latent, x, mask_k=mask))
        latent += rz(ff(name="ff_0")(latent))

        for i in range(1, self.depth):
            rz = encoders.ReZero(name=f"rezero_{i}")
            latent += rz(attn(name=f"attn_{i}")(latent))
            latent += rz(ff(name=f"ff_{i}")(latent))

        return latent


class WayformerEncoder(nn.Module):
    """Wayformer encoder module.

    This module encodes the observations using separate embeddings and applies the Wayformer attention.

    Args:
        unflatten_fn: Function to unflatten input observations.
        embedding_layer_sizes: Tuple with sizes of the MLP embedding layers.
        embedding_activation: Activation function in the MLP embedding.
        attention_depth: Depth of the attention layers.
        dk: Projection dimension.
        num_latents: Number of latent vectors.
        latent_num_heads: Number of attention heads for the latent layer.
        latent_head_features: Feature size per head in the latent layer.
        ff_mult: Multiplier used in the feedforward network.
        attn_dropout: Dropout probability in attention.
        ff_dropout: Dropout probability in the feedforward network.
        fusion_type: Fusion strategy.

    """

    unflatten_fn: callable = lambda x: x
    embedding_layer_sizes: tuple[int] = (256, 256)
    embedding_activation: datatypes.ActivationFn = nn.relu
    attention_depth: int = 2
    dk: int = 64
    num_latents: int = 64
    latent_num_heads: int = 4
    latent_head_features: int = 64
    ff_mult: int = 2
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    fusion_type: str = "late"

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """Forward pass of the Wayformer encoder.

        Args:
            obs: Input observation tensor.

        Returns:
            Output encoded tensor.

        """
        features, masks = self.unflatten_fn(obs)

        sdc_traj_features, other_traj_features, rg_features, tl_features, gps_path_features = features
        sdc_traj_valid_mask, other_traj_valid_mask, rg_valid_mask, tl_valid_mask = masks

        num_objects, timestep_agent = other_traj_features.shape[-3:-1]
        num_roadgraph = rg_features.shape[-2]
        target_len = gps_path_features.shape[-2]
        num_light, timestep_tl = tl_features.shape[-3:-1]

        # Latent encoding - Projection (Page 3 paper)
        # We found the simple transformation Projection(xi) = relu(Wxi + b)
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

        # Positional Encoding - Page 3 paper
        sdc_traj_encoding += jnp.expand_dims(self.param("sdc_traj_pe", init.normal(), (1, timestep_agent, self.dk)), 0)
        other_traj_encoding += jnp.expand_dims(
            self.param("other_traj_pe", init.normal(), (num_objects, timestep_agent, self.dk)),
            0,
        )
        rg_encoding += jnp.expand_dims(self.param("rg_pe", init.normal(), (num_roadgraph, self.dk)), 0)
        tl_encoding += jnp.expand_dims(self.param("tj_pe", init.normal(), (num_light, timestep_tl, self.dk)), 0)
        gps_path_encoding += jnp.expand_dims(self.param("gps_path_pe", init.normal(), (target_len, self.dk)), 0)

        # Temporal Encoding
        temp_pe_agents = self.param("temp_pe_agents", init.normal(), (timestep_agent,))
        temp_pe_tl = self.param("temp_pe_tl", init.normal(), (timestep_tl,))
        sdc_traj_encoding += temp_pe_agents[None, None, :, None]
        other_traj_encoding += temp_pe_agents[None, None, :, None]
        tl_encoding += temp_pe_tl[None, None, :, None]

        # Rearrange embeddings with temporal dim
        sdc_traj_encoding = einops.rearrange(sdc_traj_encoding, "b n t d -> b (n t) d")
        other_traj_encoding = einops.rearrange(other_traj_encoding, "b n t d -> b (n t) d")
        tl_encoding = einops.rearrange(tl_encoding, "b n t d -> b (n t) d")

        # Masks
        sdc_traj_valid_mask = einops.rearrange(sdc_traj_valid_mask, "b n t -> b (n t)")
        other_traj_valid_mask = einops.rearrange(other_traj_valid_mask, "b n t -> b (n t)")
        tl_valid_mask = einops.rearrange(tl_valid_mask, "b n t -> b (n t)")
        gps_path_mask = jnp.ones(gps_path_encoding.shape[:-1])

        # Only self attention mechanism in wayformer
        self_attn = partial(
            WayformerAttention,
            num_latents=self.num_latents,
            num_heads=self.latent_num_heads,
            head_features=self.latent_head_features,
            ff_mult=self.ff_mult,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
        )

        # Early Fusion - fuse after attention for all types of features
        if self.fusion_type == "early":
            # Concat the embeddings then apply self attention on the input concatenated
            concat_embeddings = jnp.concatenate(
                [sdc_traj_encoding, other_traj_encoding, rg_encoding, tl_encoding, gps_path_encoding],
                axis=1,
            )
            concat_masks = jnp.concatenate(
                [
                    sdc_traj_valid_mask,
                    other_traj_valid_mask,
                    rg_valid_mask,
                    tl_valid_mask,
                    gps_path_mask,
                ],
                axis=1,
            )
            output = self_attn(depth=self.attention_depth, name="concat_attention")(
                concat_embeddings,
                concat_masks,
            )

        elif self.fusion_type == "late":
            # Late Fusion - fuse after attention for all types of features
            # Self attention all features embeddings
            output_sdc_traj = self_attn(depth=self.attention_depth, name="sdc_traj_attention")(
                sdc_traj_encoding,
                sdc_traj_valid_mask,
            )
            output_other_traj = self_attn(depth=self.attention_depth, name="other_traj_attention")(
                other_traj_encoding,
                other_traj_valid_mask,
            )
            output_rg = self_attn(depth=self.attention_depth, name="rg_attention")(rg_encoding, rg_valid_mask)
            output_tl = self_attn(depth=self.attention_depth, name="tl_attention")(tl_encoding, tl_valid_mask)
            output_gps_path = self_attn(depth=self.attention_depth, name="gps_path_attention")(gps_path_encoding)

            # [B,M,D]
            output = jnp.concatenate(
                [output_sdc_traj, output_other_traj, output_rg, output_tl, output_gps_path],
                axis=-2,
            )

        elif self.fusion_type == "hierarchical":
            # Do both attention for each features + self attention after concatenation
            output_sdc_traj = self_attn(depth=self.attention_depth, name="sdc_traj_attention")(
                sdc_traj_encoding,
                sdc_traj_valid_mask,
            )
            output_other_traj = self_attn(depth=self.attention_depth, name="other_traj_attention")(
                other_traj_encoding,
                other_traj_valid_mask,
            )
            output_rg = self_attn(depth=self.attention_depth, name="rg_attention")(rg_encoding, rg_valid_mask)
            output_tl = self_attn(depth=self.attention_depth, name="tl_attention")(tl_encoding, tl_valid_mask)
            output_gps_path = self_attn(depth=self.attention_depth, name="gps_path_attention")(gps_path_encoding)

            # [B,M,D]
            output = jnp.concatenate(
                [output_sdc_traj, output_other_traj, output_rg, output_tl, output_gps_path],
                axis=-2,
            )

            # Self attention after concatenation of single attention per features
            output = self_attn(depth=self.attention_depth, name="concat_attention")(
                output,
                None,
            )

        # average over latent dimensions
        output = output.mean(axis=1)
        return output
