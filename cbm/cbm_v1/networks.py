"""CBM-V1 Flax network modules.

Defines CBMPolicyNetwork and CBMValueNetwork that insert a concept
bottleneck between the pretrained encoder and the actor/critic heads.

Architecture:
    obs → encoder → z (128-d) → concept_head → c (11-d) → actor/critic FC → output

For HARD bottleneck: actor/critic only see the concept vector c.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents import datatypes


class ConceptHead(nn.Module):
    """MLP that maps encoder latent to concept predictions.

    Output is sigmoid-activated so all concepts are in [0, 1].
    """

    num_concepts: int = 11
    hidden_sizes: Sequence[int] = (64,)
    activation: datatypes.ActivationFn = nn.relu

    @nn.compact
    def __call__(self, z: jax.Array) -> jax.Array:
        x = z
        for i, size in enumerate(self.hidden_sizes):
            x = nn.Dense(size, name=f"concept_hidden_{i}")(x)
            x = self.activation(x)
        x = nn.Dense(self.num_concepts, name="concept_out")(x)
        return nn.sigmoid(x)  # [0, 1] for all concepts


class CBMPolicyNetwork(nn.Module):
    """Policy network with concept bottleneck.

    Flow: obs → encoder → z → concept_head → c → actor_fc → Dense(output_size)

    When frozen=True, gradients are stopped after the encoder so only
    the concept head and actor FC receive gradient updates.
    """

    encoder_layer: nn.Module | None = None
    concept_head: ConceptHead | None = None
    actor_fc: nn.Module | None = None
    final_activation: Callable | None = None
    output_size: int = 1
    frozen_encoder: bool = False

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        # Encode
        z = self.encoder_layer(obs) if self.encoder_layer is not None else obs

        # In frozen mode, stop gradients after encoder
        if self.frozen_encoder:
            z = jax.lax.stop_gradient(z)

        # Concept bottleneck
        concepts = self.concept_head(z)  # (batch, num_concepts)

        # Actor head operates on concept vector only (HARD bottleneck)
        x = self.actor_fc(concepts)
        x = nn.Dense(self.output_size, name="policy_output")(x)

        if self.final_activation:
            x = self.final_activation(x)
        return x

    def encode_and_predict_concepts(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return both latent and concepts for auxiliary use.

        The latent is returned WITHOUT stop_gradient regardless of
        frozen_encoder, so callers can decide gradient behavior.
        """
        z = self.encoder_layer(obs) if self.encoder_layer is not None else obs
        concepts = self.concept_head(z)
        return z, concepts

    @nn.compact
    def act_from_concepts(self, concepts: jax.Array) -> jax.Array:
        """Run actor head on an externally-supplied concept vector.
        
        This enables concept intervention and ablation experiments:
        callers can modify `concepts` before passing it here.
        """
        x = self.actor_fc(concepts)
        x = nn.Dense(self.output_size, name="policy_output")(x)

        if self.final_activation:
            x = self.final_activation(x)
        return x


class CBMValueNetwork(nn.Module):
    """Value network with concept bottleneck, supporting twin Q-networks.

    Flow: obs → encoder → z → concept_head → c → [c || action] → critic_fc → Q
    """

    encoder_layer: nn.Module | None = None
    concept_head: ConceptHead | None = None
    critic_fc: nn.Module | None = None
    final_activation: Callable | None = None
    output_size: int = 1
    num_networks: int = 2
    shared_encoder: bool = False
    frozen_encoder: bool = False

    @nn.compact
    def __call__(self, obs: jax.Array, actions: jax.Array | None = None) -> jax.Array:
        shared = self.shared_encoder and self.encoder_layer is not None

        if shared:
            z = self.encoder_layer(obs)
            if self.frozen_encoder:
                z = jax.lax.stop_gradient(z)
            concepts = self.concept_head(z)

        out = []
        for i in range(self.num_networks):
            if not shared:
                if self.encoder_layer is not None:
                    z = self.encoder_layer(obs)
                else:
                    z = obs
                if self.frozen_encoder:
                    z = jax.lax.stop_gradient(z)
                concepts = self.concept_head(z)

            # Critic sees concept vector + action (HARD bottleneck)
            x = jnp.concatenate([concepts, actions], axis=-1) if actions is not None else concepts
            x = self.critic_fc(x)
            x = nn.Dense(self.output_size, name=f"value_output_{i}")(x)

            if self.final_activation:
                x = self.final_activation(x)
            out.append(x)

        return jnp.concatenate(out, axis=-1)
