# Copyright 2025 Valeo.


"""MLP encoder."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents import datatypes
from vmax.agents.networks import decoders, encoders


class MLPEncoder(nn.Module):
    """MLP encoder module.

    Encodes input features via MLPs after applying masking and flattening operations.

    Args:
        unflatten_fn: Function to unflatten the input observations.
        embedding_layer_sizes: Tuple of layer sizes for the MLP embedding.
        embedding_activation: Activation function for the embedding MLP.
        dk: Projection dimension for the latent space.
        concat_layer_sizes: Sizes of the MLP layers used for concatenation.
        concat_activation: Activation function for the concatenation MLP.

    """

    unflatten_fn: callable = lambda x: x
    embedding_layer_sizes: tuple[int] = (256, 256)
    embedding_activation: datatypes.ActivationFn = nn.relu
    dk: int = 64
    concat_layer_sizes: tuple[int] = (256, 256)
    concat_activation: datatypes.ActivationFn = nn.relu

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """Perform the forward pass of the MLP encoder.

        Args:
            obs: Observation tensor.

        Returns:
            Encoded output tensor.

        """
        features, masks = self.unflatten_fn(obs)

        sdc_traj_features, other_traj_features, rg_features, tl_features, gps_path_features = features
        sdc_traj_valid_mask, other_traj_valid_mask, rg_valid_mask, tl_valid_mask = masks

        # Applying masks - 0 value to all features if unvalid
        sdc_traj_features = jnp.where(sdc_traj_valid_mask[..., None], sdc_traj_features, 0)
        other_traj_features = jnp.where(other_traj_valid_mask[..., None], other_traj_features, 0)
        rg_features = jnp.where(rg_valid_mask[..., None], rg_features, 0)
        tl_features = jnp.where(tl_valid_mask[..., None], tl_features, 0)

        # Flattening
        sdc_traj_features = sdc_traj_features.reshape(sdc_traj_features.shape[0], -1)
        other_traj_features = other_traj_features.reshape(other_traj_features.shape[0], -1)
        rg_features = rg_features.reshape(rg_features.shape[0], -1)
        tl_features = tl_features.reshape(tl_features.shape[0], -1)
        gps_path_features = gps_path_features.reshape(gps_path_features.shape[0], -1)

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

        # Concatenate
        input = jnp.concatenate(
            [sdc_traj_encoding, other_traj_encoding, rg_encoding, tl_encoding, gps_path_encoding],
            axis=1,
        )

        # Apply mlp
        output = decoders.MLP(layer_sizes=self.concat_layer_sizes, activation=self.concat_activation)(input)

        return output
