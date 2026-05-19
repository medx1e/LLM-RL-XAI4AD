# Copyright 2025 Valeo.

"""Latent-query hierarchical encoder module.

Paper: https://arxiv.org/abs/2210.09539
"""

from functools import partial

import einops
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents import datatypes
from vmax.agents.networks import encoders


class LQHAttention(nn.Module):
    """Latent-query hierarchical attention module.

    Applies cross-attention (and optionally self-attention) over different input tensors.

    Args:
        num_latents: Number of latent vectors.
        num_heads: Number of attention heads.
        head_features: Feature size per head.
        ff_mult: Feedforward multiplier.
        attn_dropout: Dropout probability in the attention layers.
        ff_dropout: Dropout probability in the feedforward layers.
        use_self_attention: Flag to include self-attention.

    """

    num_latents: int = 64
    num_heads: int = 2
    head_features: int = 64
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    use_self_attention: bool = False

    @nn.compact
    def __call__(self, xs, masks):
        """Compute the forward pass of the LQH attention module.

        Args:
            xs: List of input tensors.
            masks: List of masks corresponding to each input tensor.

        Returns:
            Updated latent tensor.

        """
        bs, dim = xs[0].shape[0], xs[0].shape[-1]
        latents = self.param("latents", init.normal(), (self.num_latents, dim * self.ff_mult))
        latent = einops.repeat(latents, "n d -> b n d", b=bs)

        attn = partial(
            encoders.AttentionLayer,
            heads=self.num_heads,
            head_features=self.head_features,
            dropout=self.attn_dropout,
        )
        ff = partial(encoders.FeedForward, mult=self.ff_mult, dropout=self.ff_dropout)

        # LQH - different weights for each cross and self attention
        for i, (x, mask) in enumerate(zip(xs, masks, strict=False)):
            _x = einops.rearrange(x, "b n ... -> b n (...)")
            rz = encoders.ReZero(name=f"rezero_cross_{i}")
            # Cross attentions
            latent += rz(attn(name=f"cross_attn_{i}")(latent, _x, mask_k=mask))
            latent += rz(ff(name=f"ff_cross_attn_{i}")(latent))

            # Self attentions
            if self.use_self_attention:
                rz = encoders.ReZero(name=f"rezero_self_{i}")
                latent += rz(attn(name=f"self_attn_{i}")(latent))
                latent += rz(ff(name=f"ff_self_attn_{i}")(latent))

        return latent


class LQHEncoder(nn.Module):
    """Latent-query hierarchical encoder module.

    Constructs embeddings for sub-features and applies LQH attention to encode observations.

    Args:
        unflatten_fn: Function to unflatten the input observations.
        embedding_layer_sizes: Sizes of the MLP embedding layers.
        embedding_activation: Activation function for the embeddings.
        dk: Projection dimension.
        num_latents: Number of latent tokens.
        latent_num_heads: Number of attention heads for the latent space.
        latent_head_features: Feature size per latent head.
        ff_mult: Multiplier for the feedforward network.
        attn_dropout: Dropout probability in attention.
        ff_dropout: Dropout probability in the feedforward network.
        use_self_attention: Flag to add self-attention blocks.

    """

    unflatten_fn: callable = lambda x: x
    embedding_layer_sizes: tuple[int] = (256, 256)
    embedding_activation: datatypes.ActivationFn = nn.relu
    dk: int = 64
    num_latents: int = 64
    latent_num_heads: int = 2
    latent_head_features: int = 64
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    use_self_attention: bool = False

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """Perform the forward pass of the latent-query hierarchical encoder.

        Args:
            obs: Input observation tensor.

        Returns:
            Encoded output tensor.

        """
        features, masks = self.unflatten_fn(obs)

        sdc_traj_features, other_traj_features, rg_features, tl_features, gps_path_features = features
        sdc_traj_valid_mask, other_traj_valid_mask, rg_valid_mask, tl_valid_mask = masks

        num_objects, timestep_agents = other_traj_features.shape[-3:-1]
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
        sdc_traj_encoding += jnp.expand_dims(self.param("sdc_traj_pe", init.normal(), (1, timestep_agents, self.dk)), 0)
        other_traj_encoding += jnp.expand_dims(
            self.param("other_traj_pe", init.normal(), (num_objects, timestep_agents, self.dk)),
            0,
        )
        rg_encoding += jnp.expand_dims(self.param("rg_pe", init.normal(), (num_roadgraph, self.dk)), 0)
        tl_encoding += jnp.expand_dims(self.param("tj_pe", init.normal(), (num_light, timestep_tl, self.dk)), 0)
        gps_path_encoding += jnp.expand_dims(self.param("gps_path_pe", init.normal(), (target_len, self.dk)), 0)

        # Flatten by NumAgent NumObsTS , Feature_dim
        sdc_traj_encoding = einops.rearrange(sdc_traj_encoding, "b n t d -> b (n t) d")
        other_traj_encoding = einops.rearrange(other_traj_encoding, "b n t d -> b (n t) d")
        tl_encoding = einops.rearrange(tl_encoding, "b n t d -> b (n t) d")

        # Mask
        sdc_traj_valid_mask = einops.rearrange(sdc_traj_valid_mask, "b n t -> b (n t)")
        other_traj_valid_mask = einops.rearrange(other_traj_valid_mask, "b n t -> b (n t)")
        tl_valid_mask = einops.rearrange(tl_valid_mask, "b n t -> b (n t)")

        inputs = [sdc_traj_encoding, other_traj_encoding, rg_encoding, tl_encoding, gps_path_encoding]
        masks = [sdc_traj_valid_mask, other_traj_valid_mask, rg_valid_mask, tl_valid_mask, None]

        output = LQHAttention(
            num_latents=self.num_latents,
            num_heads=self.latent_num_heads,
            head_features=self.latent_head_features,
            ff_mult=self.ff_mult,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            use_self_attention=self.use_self_attention,
            name="lq_attention",
        )(inputs, masks)

        output = output.mean(axis=1)

        return output
