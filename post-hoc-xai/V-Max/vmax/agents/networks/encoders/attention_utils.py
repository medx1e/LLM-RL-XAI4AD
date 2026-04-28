# Copyright 2025 Valeo.

"""Module providing attention and feedforward utilities for encoder networks."""

import einops
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import linen as nn


class FeedForward(nn.Module):
    """Feed forward network with GELU activation and dropout.

    Args:
        mult: Multiplier for the hidden layer size.
        dropout: Dropout probability.

    """

    mult: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        """Apply the feedforward network to the input tensor.

        Args:
            x: Input tensor.
            deterministic: If True, disables dropout.

        Returns:
            Output tensor after the feedforward operation.

        """
        features = x.shape[-1]

        x = nn.Dense(features * self.mult)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(features)(x)

        return x


def default(val, d):
    """Return val if val is not None, else d."""
    return val if val is not None else d


def nearest_neighbors_jax(x: jax.Array, y: jax.Array, k: int, mask_y=None) -> jax.Array:
    """Compute the nearest neighbors of x in y.

    Args:
        x: Query tensor.
        y: Reference tensor.
        k: Number of neighbors to return.
        mask_y: Optional boolean mask for y voxels.

    Returns:
        Indices of the nearest neighbors.

    """
    distance_matrix = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)  # [N,M]
    if mask_y is not None:
        distance_matrix = distance_matrix + jnp.inf * ~mask_y[None, :]

    idxs = jnp.argsort(distance_matrix, axis=-1)[:, :k]  # [N,k]

    return idxs


class ReZero(nn.Module):
    """ReZero block which scales the output.

    This block learns a scaling parameter to modulate the residual connection.

    """

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the ReZero scaling to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Scaled tensor.

        """
        scale = self.param("scale", init.zeros, (1,))

        return scale * x


class AttentionLayer(nn.Module):
    """Attention layer that computes self or cross attention.

    Args:
        heads: Number of attention heads.
        head_features: Feature size per head.
        dropout: Dropout probability.

    """

    heads: int = 8
    head_features: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, context=None, mask_k=None, mask_q=None, deterministic: bool = False) -> jax.Array:
        """Perform attention on the input.

        Args:
            x: Query tensor.
            context: Context tensor; if None, x is used.
            mask_k: Mask applied on the keys.
            mask_q: Mask applied on the queries.
            deterministic: If True, disables dropout.

        Returns:
            Output tensor after attention.

        """
        # mask is on context(k)
        h = self.heads
        dim = self.head_features * h

        q = nn.Dense(dim, use_bias=False)(x)
        k = nn.Dense(dim, use_bias=False)(default(context, x))
        v = nn.Dense(dim, use_bias=False)(default(context, x))

        q, k, v = map(lambda arr: einops.rearrange(arr, "b n (h d) -> b n h d", h=h), (q, k, v))
        sim = jnp.einsum("b i h d, b j h d -> b i j h", q, k) * self.head_features**-0.5

        if mask_k is not None:
            big_neg = jnp.finfo(jnp.float32).min
            sim = jnp.where(mask_k[:, None, :, None], sim, big_neg)
        if mask_q is not None:
            big_neg = jnp.finfo(jnp.float32).min
            sim = jnp.where(mask_q[:, :, None, None], sim, big_neg)

        attn = nn.softmax(sim, axis=-2)  # -2 we kept h dim in matrix (could we merge h with b ?)
        out = jnp.einsum("b i j h, b j h d -> b i h d", attn, v)
        out = einops.rearrange(out, "b n h d -> b n (h d)", h=h)

        out = nn.Dense(x.shape[-1])(out)
        out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        return out


class LocalAttentionLayer(nn.Module):
    """Local attention layer that performs attention over a local neighborhood.

    Args:
        heads: Number of attention heads.
        head_features: Feature size per head.
        dropout: Dropout probability.

    """

    heads: int = 8
    head_features: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        context,
        index_pairs,
        mask_q=None,
        mask_k=None,
        deterministic: bool = False,
    ) -> jax.Array:
        """Apply local attention via nearest neighbor selections.

        Args:
            x: Query tensor.
            context: Context tensor.
            index_pairs: Indices defining local neighborhoods.
            mask_q: Mask for the queries.
            mask_k: Mask for the keys.
            deterministic: If True, disables dropout.

        Returns:
            Output tensor after local attention.

        """
        # masqk_q: [B,Nq]
        # masqk_k: [B,Nk]
        # index pairs: [B, Nq, K]
        h = self.heads
        dim = self.head_features * h

        q = nn.Dense(dim)(x)
        k = nn.Dense(dim, use_bias=False)(default(context, x))
        v = nn.Dense(dim, use_bias=False)(default(context, x))

        k = jnp.take_along_axis(k[:, :, None, :], index_pairs[:, :, :, None], axis=1)  # [B,Nq,K,D]
        v = jnp.take_along_axis(v[:, :, None, :], index_pairs[:, :, :, None], axis=1)  # [B,Nq,K,D]
        # subset of k

        k = einops.rearrange(k, "b n k (h d) -> b n k h d", h=h)
        v = einops.rearrange(v, "b n k (h d) -> b n k h d", h=h)
        q = einops.rearrange(q, "b n (h d) -> b n h d", h=h)

        sim = jnp.einsum("b i h d, b i k h d -> b i k h", q, k) * self.head_features**-0.5  # [B,Nq,k,H]

        if mask_q is not None:
            big_neg = jnp.finfo(jnp.float32).min
            sim = jnp.where(mask_q[:, :, None, None], sim, big_neg)
        if mask_k is not None:
            big_neg = jnp.finfo(jnp.float32).min
            mask = jnp.take_along_axis(mask_k[:, :, None], index_pairs[:, :, :], axis=1)  # [B,Nq,K]
            sim = jnp.where(mask[:, :, :, None], sim, big_neg)

        attn = nn.softmax(sim, axis=-2)  # [B,Nq,k,H]
        out = jnp.einsum("b i k h, b i k h d -> b i h d", attn, v)
        out = einops.rearrange(out, "b n h d -> b n (h d)", h=h)  # [B,Nq,d]

        out = nn.Dense(x.shape[-1])(out)
        out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        return out
