"""Wrapper for Perceiver/LQ encoder models.

The Perceiver uses cross-attention from learned latent queries to input
tokens.  This wrapper exposes the ExplainableModel interface and can
extract attention weights via Flax ``capture_intermediates``.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from posthoc_xai.models.base import ExplainableModel, ModelOutput
from posthoc_xai.models._obs_structure import (
    compute_observation_structures,
    extract_entity_validity,
)


class PerceiverWrapper(ExplainableModel):
    """ExplainableModel implementation for LQ/Perceiver encoder.

    Constructed via :func:`posthoc_xai.models.loader.load_vmax_model`.

    Args:
        loaded: A ``LoadedVMAXModel`` object returned by the loader.
    """

    def __init__(self, loaded: Any):
        self._loaded = loaded
        self._policy_params = loaded.policy_params
        self._policy_module = loaded.policy_module
        self._unflatten_fn = loaded.unflatten_fn
        self._action_size = loaded.action_size
        self._obs_size = loaded.obs_size
        self._config = loaded.config

        # Pre-compute observation structure (category + entity level)
        self._obs_structure, self._detailed_structure = (
            compute_observation_structures(self._unflatten_fn, self._obs_size)
        )

    # ------------------------------------------------------------------
    # ExplainableModel interface
    # ------------------------------------------------------------------

    def forward(self, observation: jnp.ndarray) -> ModelOutput:
        """Full forward pass.

        Args:
            observation: Shape ``(batch, obs_dim)`` or ``(obs_dim,)``.

        Returns:
            ``ModelOutput`` with action mean/std, embedding, and attention.
        """
        needs_batch = observation.ndim == 1
        obs = observation[None, :] if needs_batch else observation

        # Run with capture_intermediates for attention extraction
        logits, state = self._policy_module.apply(
            self._policy_params,
            obs,
            capture_intermediates=True,
            mutable=["intermediates"],
        )

        # Split logits → (mean, log_std) for NormalTanh
        action_mean = logits[..., : self._action_size]
        action_log_std = logits[..., self._action_size :]
        action_std = jnp.exp(action_log_std)

        # Extract encoder embedding by running encoder only
        embedding = self._get_embedding_from_intermediates(state)

        # Extract attention weights from intermediates
        attention = self._extract_attention(state)

        if needs_batch:
            action_mean = action_mean[0]
            action_std = action_std[0]
            if embedding is not None:
                embedding = embedding[0]

        return ModelOutput(
            action_mean=action_mean,
            action_std=action_std,
            embedding=embedding,
            attention=attention,
        )

    def get_action_value(
        self,
        observation: jnp.ndarray,
        action_idx: Optional[int] = None,
    ) -> jnp.ndarray:
        """Scalar output for ``jax.grad``.

        Args:
            observation: Shape ``(obs_dim,)`` (unbatched, float).
            action_idx: Which action dim (None → sum all means).

        Returns:
            Scalar ``jnp.ndarray``.
        """
        # Add batch dim
        obs = observation[None, :]
        logits = self._policy_module.apply(self._policy_params, obs)
        action_mean = logits[0, : self._action_size]

        if action_idx is not None:
            return action_mean[action_idx]
        return jnp.sum(action_mean)

    def get_embedding(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Get the encoder output.

        Args:
            observation: Shape ``(obs_dim,)`` or ``(batch, obs_dim)``.

        Returns:
            Embedding array.
        """
        output = self.forward(observation)
        return output.embedding

    def get_attention(
        self, observation: jnp.ndarray
    ) -> Optional[dict[str, jnp.ndarray]]:
        """Get attention weights from the LQ cross-attention layers."""
        output = self.forward(observation)
        return output.attention

    @property
    def observation_structure(self) -> dict[str, tuple[int, int]]:
        return self._obs_structure

    @property
    def observation_structure_detailed(self) -> dict:
        return self._detailed_structure

    @property
    def has_attention(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"perceiver_{self._loaded.original_encoder_type}"

    def get_entity_validity(
        self, observation: jnp.ndarray
    ) -> dict[str, dict[str, bool]]:
        return extract_entity_validity(self._unflatten_fn, observation)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embedding_from_intermediates(self, state: Any) -> Optional[jnp.ndarray]:
        """Try to pull the encoder output from captured intermediates.

        Falls back to None if the intermediates tree doesn't contain it.
        """
        intermediates = state.get("intermediates", {})
        # Walk the tree looking for the encoder_layer output
        # The PolicyNetwork calls: encoder_layer(obs) → embedding → fc → Dense
        # capture_intermediates stores __call__ returns at each module level
        try:
            enc_interms = intermediates.get("encoder_layer", {})
            if "__call__" in enc_interms:
                vals = enc_interms["__call__"]
                if isinstance(vals, (list, tuple)) and len(vals) > 0:
                    return vals[0]
        except (AttributeError, KeyError, TypeError):
            pass
        return None

    def _extract_attention(self, state: Any) -> Optional[dict[str, jnp.ndarray]]:
        """Compute softmax attention weights from captured Q/K Dense projections.

        Reconstructs both cross-attention (queries → input tokens) and
        self-attention (queries → queries), then computes attention rollout
        (Abnar & Zuidema 2020) through the self-attention layers.

        Architecture (all perceiver_*_42 models, tie_layer_weights=True):
            n_layers      = 4    encoder_depth
            n_queries     = 16   num_latents
            n_tokens      = 280  input tokens (5+40+200+25+10)
            cross_n_heads = 2    cross_num_heads
            cross_hd      = 16   cross_head_features
            self_n_heads  = 2    latent_num_heads
            self_hd       = 16   latent_head_features

        Returns dict with keys:
            cross_attn_layer_{0..3} : (B, 16, 280) per cross-attn layer
            cross_attn_avg          : (B, 16, 280) mean over 4 layers
            self_attn_layer_{0..3}  : (B, 16,  16) per self-attn layer
            cross_attn_rollout      : (B, 16, 280) rollout-corrected attention
            norm_weighted_attn      : (B, 16, 280) value-norm-weighted, row-normalised
            f_x_norms               : (B, 280)     per-token ‖f(x_j)‖₂ with W_O projection
            f_v_norms               : (B, 280)     per-token ‖v_j‖₂ without W_O (Option 1b)
        """
        intermediates = state.get("intermediates", {})
        lq = {}
        try:
            lq = (
                intermediates
                .get("encoder_layer", {})
                .get("lq_attention", {})
            )
        except (AttributeError, TypeError):
            return None

        # ── Cross-attention ───────────────────────────────────────────────
        cross    = lq.get("cross_attn", {})
        cq_list  = cross.get("Dense_0", {}).get("__call__", [])  # Q: (B, 16, 32)
        ck_list  = cross.get("Dense_1", {}).get("__call__", [])  # K: (B, 280, 32)

        if not cq_list or not ck_list or len(cq_list) != len(ck_list):
            return None

        cross_n_heads, cross_hd = 2, 16
        cross_scale = jnp.sqrt(jnp.array(cross_hd, dtype=jnp.float32))

        per_cross_layer = []
        for q_raw, k_raw in zip(cq_list, ck_list):
            B, Q = q_raw.shape[0], q_raw.shape[1]
            T    = k_raw.shape[1]
            q = q_raw.reshape(B, Q, cross_n_heads, cross_hd)   # (B, Q, H, D)
            k = k_raw.reshape(B, T, cross_n_heads, cross_hd)   # (B, T, H, D)
            scores = jnp.einsum("bqhd,bthd->bhqt", q, k) / cross_scale
            attn   = jax.nn.softmax(scores, axis=-1)            # (B, H, Q, T)
            per_cross_layer.append(attn.mean(axis=1))           # (B, Q, T)

        result: dict[str, jnp.ndarray] = {}
        for i, a in enumerate(per_cross_layer):
            result[f"cross_attn_layer_{i}"] = a
        cross_avg = jnp.stack(per_cross_layer, axis=0).mean(axis=0)  # (B, 16, 280)
        result["cross_attn_avg"] = cross_avg

        # ── Norm-weighted attention (Kobayashi et al. 2020) ───────────────
        # Weights each token's attention by its value-vector magnitude:
        # norm_weighted[i,j] = α[i,j] × ‖f(x_j)‖₂, then row-normalised.
        # Captures not just where the model looks but how much each token
        # contributes to the output representation.
        cv_list = cross.get("Dense_2", {}).get("__call__", [])
        if cv_list:
            # tied weights + same input x → all 4 entries identical; use first
            v_tokens  = cv_list[0]                                       # (B, 280, 32)
            f_v_norms = jnp.linalg.norm(v_tokens, axis=-1)              # (B, 280) Option 1b

            try:
                wo = (
                    self._policy_params['params']
                    ['encoder_layer']['lq_attention']['cross_attn']['Dense_3']
                )
                f_x       = jnp.einsum("btd,dk->btk", v_tokens, wo['kernel']) + wo['bias']
                f_x_norms = jnp.linalg.norm(f_x, axis=-1)               # (B, 280) Option 1a
            except (KeyError, TypeError):
                f_x_norms = f_v_norms                                    # fall back to 1b

            nw_layers = [a * f_x_norms[:, None, :] for a in per_cross_layer]
            nw_avg    = jnp.stack(nw_layers, axis=0).mean(axis=0)       # (B, 16, 280)
            nw_norm   = nw_avg / (nw_avg.sum(axis=-1, keepdims=True) + 1e-8)

            result["norm_weighted_attn"] = nw_norm    # (B, 16, 280) row-normalised
            result["f_x_norms"]          = f_x_norms  # (B, 280) with W_O
            result["f_v_norms"]          = f_v_norms  # (B, 280) without W_O

        # ── Self-attention ────────────────────────────────────────────────
        self_mod  = lq.get("self_attn", {})
        sq_list   = self_mod.get("Dense_0", {}).get("__call__", [])  # Q: (B, 16, 32)
        sk_list   = self_mod.get("Dense_1", {}).get("__call__", [])  # K: (B, 16, 32)

        if not sq_list or not sk_list or len(sq_list) != len(sk_list):
            # Self-attention not captured; return cross-attention only
            return result

        self_n_heads, self_hd = 2, 16
        self_scale = jnp.sqrt(jnp.array(self_hd, dtype=jnp.float32))

        per_self_layer = []
        for sq_raw, sk_raw in zip(sq_list, sk_list):
            B, Q = sq_raw.shape[0], sq_raw.shape[1]
            sq = sq_raw.reshape(B, Q, self_n_heads, self_hd)   # (B, Q, H, D)
            sk = sk_raw.reshape(B, Q, self_n_heads, self_hd)   # (B, Q, H, D)
            # Query-to-query scores: (B, H, Q, Q)
            scores = jnp.einsum("bqhd,bkhd->bhqk", sq, sk) / self_scale
            attn   = jax.nn.softmax(scores, axis=-1)            # (B, H, Q, Q)
            per_self_layer.append(attn.mean(axis=1))            # (B, Q, Q)

        for i, a in enumerate(per_self_layer):
            result[f"self_attn_layer_{i}"] = a

        # ── Attention rollout (Abnar & Zuidema 2020) ──────────────────────
        # Approximate the effective influence of input tokens on the final
        # representation by chaining residual-corrected self-attention layers
        # and applying the result to the mean cross-attention.
        #
        # For each self-attention layer l:
        #   A_eff[l] = 0.5 * I  +  0.5 * A_self[l]   (residual correction)
        # Rollout matrix R = A_eff[3] @ A_eff[2] @ A_eff[1] @ A_eff[0]  (B,16,16)
        # Effective cross-attention = R @ cross_attn_avg                  (B,16,280)
        #
        # Note: this is an approximation — it treats cross-attention as a single
        # operation (using the layer-averaged cross_attn_avg) and ignores the
        # interleaved residual additions between cross- and self-attention blocks.
        B      = cross_avg.shape[0]
        n_q    = cross_avg.shape[1]
        I_mat  = jnp.eye(n_q, dtype=cross_avg.dtype)            # (Q, Q)
        I_b    = jnp.broadcast_to(I_mat[None], (B, n_q, n_q))  # (B, Q, Q)

        rollout = I_b
        for A_self in per_self_layer:
            A_eff   = 0.5 * I_b + 0.5 * A_self                 # (B, Q, Q)
            rollout = jnp.einsum("bij,bjk->bik", A_eff, rollout)

        result["cross_attn_rollout"] = jnp.einsum(
            "bij,bjk->bik", rollout, cross_avg                  # (B, Q, Q) @ (B, Q, T)
        )                                                        # → (B, Q, T)=(B,16,280)

        return result
