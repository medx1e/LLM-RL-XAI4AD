# Copyright 2025 Valeo.

"""Factory functions for the Soft Actor-Critic (SAC) algorithm."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from vmax.agents import datatypes, networks


@flax.struct.dataclass
class SACNetworkParams:
    """Parameters for SAC network."""

    policy: datatypes.Params
    value: datatypes.Params
    target_value: datatypes.Params


@flax.struct.dataclass
class SACNetworks:
    """SAC networks."""

    policy_network: Any
    value_network: Any
    parametric_action_distribution: Any
    policy_optimizer: Any
    value_optimizer: Any


@flax.struct.dataclass
class SACTrainingState(datatypes.TrainingState):
    """Training state for SAC algorithm."""

    params: SACNetworkParams
    policy_optimizer_state: optax.OptState
    value_optimizer_state: optax.OptState
    rl_gradient_steps: int


def initialize(
    action_size: int,
    observation_size: int,
    env: Any,
    learning_rate: float,
    network_config: dict,
    num_devices: int,
    key: jax.Array,
) -> tuple[SACNetworks, SACTrainingState, datatypes.Policy]:
    """Initialize SAC components.

    Args:
        action_size: Size of the action space.
        observation_size: Size of the observation space.
        env: Environment instance with a features extractor.
        learning_rate: Learning rate for the optimizers.
        network_config: Network configuration dictionary.
        num_devices: Number of devices to use.
        key: Random key for initialization.

    Returns:
        A tuple of (networks, training state, policy function).

    """
    network = make_networks(
        observation_size=observation_size,
        action_size=action_size,
        unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
        learning_rate=learning_rate,
        network_config=network_config,
    )

    policy_function = make_inference_fn(network)

    key_policy, key_value = jax.random.split(key)

    policy_params = network.policy_network.init(key_policy)
    policy_optimizer_state = network.policy_optimizer.init(policy_params)
    value_params = network.value_network.init(key_value)
    value_optimizer_state = network.value_optimizer.init(value_params)

    init_params = SACNetworkParams(
        policy=policy_params,
        value=value_params,
        target_value=value_params,
    )

    training_state = SACTrainingState(
        params=init_params,
        policy_optimizer_state=policy_optimizer_state,
        value_optimizer_state=value_optimizer_state,
        env_steps=0,
        rl_gradient_steps=0,
    )

    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_devices])

    return network, training_state, policy_function


def make_inference_fn(sac_network: SACNetworks) -> datatypes.Policy:
    """Create the policy inference function for SAC.

    Args:
        sac_network: Instance of SACNetworks.

    Returns:
        A callable policy function.

    """

    def make_policy(params: datatypes.Params, deterministic: bool = False) -> datatypes.Policy:
        policy_network = sac_network.policy_network
        parametric_action_distribution = sac_network.parametric_action_distribution

        def policy(observations: jax.Array, key_sample: jax.Array = None) -> tuple[jax.Array, dict]:
            logits, encoder_attn_weights = policy_network.apply(params, observations)

            if deterministic:
                return parametric_action_distribution.mode(logits), {"encoder_attn_weights": encoder_attn_weights}

            return parametric_action_distribution.sample(logits, key_sample), {"encoder_attn_weights": encoder_attn_weights}

        return policy

    return make_policy


def make_networks(
    observation_size: int,
    action_size: int,
    unflatten_fn: callable,
    learning_rate: int,
    network_config: dict,
) -> SACNetworks:
    """Construct SAC networks.

    Args:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        unflatten_fn: Function to unflatten network inputs.
        learning_rate: Learning rate used for the optimizers.
        network_config: Network configuration dictionary.

    Returns:
        An instance of SACNetworks.

    """
    # Handle action distribution - be flexible with string matching
    action_dist = str(network_config.get("action_distribution", "gaussian")).lower()
    
    if "gaussian" in action_dist or "normal" in action_dist:
        parametric_action_distribution = networks.NormalTanhDistribution(event_size=action_size)
    elif "beta" in action_dist:
        parametric_action_distribution = networks.BetaDistribution(event_size=action_size)
    else:
        # Default to Gaussian if unknown
        print(f"Warning: Unknown action_distribution '{action_dist}', defaulting to Gaussian")
        parametric_action_distribution = networks.NormalTanhDistribution(event_size=action_size)

    output_size = parametric_action_distribution.param_size

    policy_network = networks.make_policy_network(network_config, observation_size, output_size, unflatten_fn)
    value_network = networks.make_value_network(network_config, observation_size, action_size, unflatten_fn)

    policy_optimizer = optax.adam(learning_rate)
    value_optimizer = optax.adam(learning_rate)

    return SACNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
    )


def make_sgd_step(
    sac_network: SACNetworks,
    alpha: float,
    discount: float,
    tau: float,
    lambda_diversity: float = 0.0,
    lambda_safety: float = 0.0,
    safety_head_index: int = 0,
    lambda_consistency: float = 0.0,
    safety_tau: float = 0.5,
    unflatten_fn: callable = None,
) -> datatypes.LearningFunction:
    """Create the SGD step function for SAC.

    Args:
        sac_network: The SAC networks.
        alpha: Entropy regularization coefficient.
        discount: Discount factor.
        tau: Coefficient for target network updates.
        lambda_diversity: Weight for the attention head diversity loss.
        lambda_safety: Weight for the safety-grounded attention loss.
        safety_head_index: Index of the attention head to enforce safety alignment.
        lambda_consistency: Weight for the within-head consistency loss.
        safety_tau: Temperature for exponential TTC target distribution.
        unflatten_fn: Function to unflatten observations into structured features.
            Required when lambda_safety > 0 for TTC computation.

    Returns:
        A function that executes an SGD step.

    """
    value_loss, policy_loss = _make_loss_fn(
        sac_network=sac_network,
        alpha=alpha,
        discount=discount,
        lambda_diversity=lambda_diversity,
        lambda_safety=lambda_safety,
        safety_head_index=safety_head_index,
        lambda_consistency=lambda_consistency,
        safety_tau=safety_tau,
        unflatten_fn=unflatten_fn,
    )

    policy_update = networks.gradient_update_fn(policy_loss, sac_network.policy_optimizer, pmap_axis_name="batch", has_aux=True)
    value_update = networks.gradient_update_fn(value_loss, sac_network.value_optimizer, pmap_axis_name="batch", has_aux=True)

    def sgd_step(
        carry: tuple[SACTrainingState, jax.Array],
        transitions: datatypes.RLTransition,
    ) -> tuple[tuple[SACTrainingState, jax.Array], datatypes.Metrics]:
        training_state, key = carry

        key, key_value, key_policy = jax.random.split(key, 3)

        (_, value_aux), value_params, value_optimizer_state = value_update(
            training_state.params.value,
            training_state.params.policy,
            training_state.params.target_value,
            transitions,
            key_value,
            optimizer_state=training_state.value_optimizer_state,
        )
        (_, policy_aux), policy_params, policy_optimizer_state = policy_update(
            training_state.params.policy,
            training_state.params.value,
            transitions,
            key_policy,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_value_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.params.target_value,
            value_params,
        )

        sgd_metrics = {**policy_aux, **value_aux}

        params = SACNetworkParams(
            policy=policy_params,
            value=value_params,
            target_value=new_target_value_params,
        )

        training_state = training_state.replace(
            params=params,
            policy_optimizer_state=policy_optimizer_state,
            value_optimizer_state=value_optimizer_state,
            rl_gradient_steps=training_state.rl_gradient_steps + 1,
        )

        return (training_state, key), sgd_metrics

    return sgd_step


def _compute_diversity_loss(attn_weights: dict) -> jax.Array:
    """Compute diversity loss via per-query cosine similarity between head pairs.

    Each query row is an independent softmax-normalized distribution over K keys.
    We compute cosine similarity along the K dimension for each query independently,
    then average over queries and batch.  For H>2 heads we loop over all unique
    pairs (i, j) where i < j.

    Args:
        attn_weights: Dictionary of attention weights from the encoder.

    Returns:
        Scalar diversity loss value.

    """
    # Try to find the other_traj cross-attention weights
    # Key format: "other_traj/cross_attn_0" for wayformer encoder
    target_key = None
    for key in attn_weights:
        if "other_traj" in key and "cross_attn" in key:
            target_key = key
            break

    if target_key is None:
        # Fallback: use the first cross_attn key available
        for key in attn_weights:
            if "cross_attn" in key:
                target_key = key
                break

    if target_key is None:
        return jnp.float32(0.0)

    # attn shape: [B, Q, K, H]
    # Q = num_latents (queries), K = num_tokens (keys), H = num_heads
    attn = attn_weights[target_key]
    H = attn.shape[-1]

   

    # Per-query cosine similarity between all unique head pairs
    pair_losses = []
    for i in range(H):
        for j in range(i + 1, H):
            a_i = attn[..., i]  # (B, Q, K)
            a_j = attn[..., j]  # (B, Q, K)

            dot = (a_i * a_j).sum(axis=-1)                       # (B, Q)
            norm_i = jnp.linalg.norm(a_i, axis=-1)               # (B, Q)
            norm_j = jnp.linalg.norm(a_j, axis=-1)               # (B, Q)
            cos_sim = dot / (norm_i * norm_j + 1e-8)              # (B, Q)

            pair_losses.append(jnp.mean(jnp.abs(cos_sim)))

    diversity_loss = sum(pair_losses) / max(len(pair_losses), 1)

    return diversity_loss


def _compute_consistency_loss(attn_weights: dict) -> jax.Array:
    """Compute within-head consistency loss via pairwise cosine similarity across queries.

    For each head, compute cosine similarity between all query pairs along the K
    dimension, then maximize it (minimize 1 - cos_sim).  This encourages all
    queries within a head to attend to similar tokens — the opposite of diversity.

    Args:
        attn_weights: Dictionary of attention weights from the encoder.

    Returns:
        Scalar consistency loss value in [0, 1].

    """
    # Find other_traj cross-attention key
    target_key = None
    for key in attn_weights:
        if "other_traj" in key and "cross_attn" in key:
            target_key = key
            break

    if target_key is None:
        for key in attn_weights:
            if "cross_attn" in key:
                target_key = key
                break

    if target_key is None:
        return jnp.float32(0.0)

    # attn shape: [B, Q, K, H]
    attn = attn_weights[target_key]
    H = attn.shape[-1]
    Q = attn.shape[1]

    head_losses = []
    for h in range(H):
        a_h = attn[..., h]  # (B, Q, K)

        pair_sims = []
        for i in range(Q):
            for j in range(i + 1, Q):
                q_i = a_h[:, i, :]  # (B, K)
                q_j = a_h[:, j, :]  # (B, K)

                dot = (q_i * q_j).sum(axis=-1)            # (B,)
                norm_i = jnp.linalg.norm(q_i, axis=-1)    # (B,)
                norm_j = jnp.linalg.norm(q_j, axis=-1)    # (B,)
                cos_sim = dot / (norm_i * norm_j + 1e-8)   # (B,)

                pair_sims.append(jnp.mean(cos_sim))

        # Consistency = 1 - mean(cos_sim): minimize to maximize agreement
        mean_sim = sum(pair_sims) / max(len(pair_sims), 1)
        head_losses.append(1.0 - mean_sim)

    consistency_loss = sum(head_losses) / max(len(head_losses), 1)
    return consistency_loss


def _compute_ttc_target(
    observation: jax.Array,
    unflatten_fn: callable,
    safety_tau: float = 0.5,
) -> tuple[jax.Array, jax.Array]:
    """Compute TTC-based target attention distribution from the observation.

    Constructs a probability distribution over agent tokens where weight is
    exponentially decaying with TTC: exp(-ttc / tau).  Low TTC (imminent threat)
    gets high weight.  All timestep tokens of the same vehicle share the
    same weight.

    Also computes scene-level collision risk R for safety loss gating.

    Feature layout (after valid is stripped to masks):
        0,1 = waypoints (x, y)   — ego-centric coordinates
        2,3 = velocity  (vx, vy)
        4   = yaw
        5,6 = size (length, width)

    Args:
        observation: Flat observation tensor, shape (B, obs_dim).
        unflatten_fn: The features extractor's unflatten_features function.
        safety_tau: Temperature for exponential weighting (default 0.5).

    Returns:
        target_dist: shape (B, n_vehicles * n_timesteps)
            Probability distribution over agent tokens.
        scene_risk: shape (B,)
            Scene-level risk R = clip(1 - min_ttc / 3, 0, 1).

    """
    FEAT_X, FEAT_Y = 0, 1
    FEAT_VX, FEAT_VY = 2, 3
    TTC_HORIZON = 5.0
    CLOSING_SPEED_THRESHOLD = 0.5

    features, masks = unflatten_fn(observation)
    other_traj = features[1]   # (B, n_vehicles, n_timesteps, n_feat)
    other_mask = masks[1]      # (B, n_vehicles, n_timesteps) boolean

    n_vehicles = other_traj.shape[-3]
    n_timesteps = other_traj.shape[-2]

    # Use latest timestep for TTC computation
    latest = other_traj[:, :, -1, :]     # (B, n_vehicles, n_feat)

    x  = latest[:, :, FEAT_X]            # (B, n_vehicles)
    y  = latest[:, :, FEAT_Y]
    vx = latest[:, :, FEAT_VX]
    vy = latest[:, :, FEAT_VY]
    valid = other_mask[:, :, -1]          # (B, n_vehicles)

    # Ego is at origin in ego-centric coordinates
    dist = jnp.sqrt(x**2 + y**2) + 1e-6  # (B, n_vehicles)

    # Closing speed: rate at which distance decreases
    closing_speed = -(x * vx + y * vy) / dist  # (B, n_vehicles)

    # TTC: time until collision (capped at horizon)
    ttc = jnp.where(
        closing_speed > CLOSING_SPEED_THRESHOLD,
        dist / (closing_speed + 1e-6),
        TTC_HORIZON,
    )
    ttc = jnp.clip(ttc, 0.0, TTC_HORIZON)      # (B, n_vehicles)

    # Exponential TTC target weight: exp(-ttc / tau)
    weights = jnp.exp(-ttc / safety_tau)         # (B, n_vehicles)

    # Zero out invalid vehicles
    weights = jnp.where(valid, weights, 0.0)

    # Scene-level risk: R = clip(1 - min_ttc / 3, 0, 1)
    # Use only valid vehicles; invalid get TTC_HORIZON
    ttc_for_risk = jnp.where(valid, ttc, TTC_HORIZON)
    min_ttc = jnp.min(ttc_for_risk, axis=-1)     # (B,)
    scene_risk = jnp.clip(1.0 - min_ttc / 3.0, 0.0, 1.0)  # (B,)

    # Expand to match token dimension: each vehicle has n_timesteps tokens
    # All timestep tokens of the same vehicle get the same weight
    weights_expanded = jnp.repeat(weights[:, :, None], n_timesteps, axis=-1)
    weights_flat = weights_expanded.reshape(weights.shape[0], n_vehicles * n_timesteps)

    # Normalize to probability distribution
    target_dist = weights_flat / (weights_flat.sum(axis=-1, keepdims=True) + 1e-8)

    return target_dist, scene_risk


def _compute_safety_loss(
    attn_weights: dict,
    observation: jax.Array,
    safety_head_index: int,
    unflatten_fn: callable,
    safety_tau: float = 0.5,
) -> tuple[jax.Array, jax.Array]:
    """Compute KL-divergence safety loss between TTC target and actual attention.

    For the designated safety head, computes D_KL(target || actual) per query,
    then averages over queries and batch.  The target distribution uses
    exponential TTC weighting: exp(-ttc / tau).

    Direction: D_KL(target || actual) penalizes the model when actual attention
    assigns low probability to tokens the target says are important (same
    direction as knowledge distillation).

    Args:
        attn_weights: Dictionary of attention weights from the encoder.
        observation: Raw observation tensor [B, obs_dim].
        safety_head_index: Index of the head designated as the safety head.
        unflatten_fn: Function to unflatten observations into structured features.
        safety_tau: Temperature for exponential TTC target.

    Returns:
        Tuple of (safety_loss, scene_risk) where scene_risk is (B,).

    """
    # Find other_traj cross-attention key
    target_key = None
    for key in attn_weights:
        if "other_traj" in key and "cross_attn" in key:
            target_key = key
            break

    if target_key is None:
        return jnp.float32(0.0), jnp.float32(0.0)

    # attn shape: [B, Q, K, H]
    attn = attn_weights[target_key]

    # Extract safety head attention: [B, Q, K]
    # Each query row is already softmax-normalized over K
    p_actual = attn[:, :, :, safety_head_index]

    # Compute TTC-based target distribution and scene risk
    p_target, scene_risk = _compute_ttc_target(observation, unflatten_fn, safety_tau=safety_tau)

    # Expand target for broadcasting: [B, 1, K]
    p_target = p_target[:, None, :]

    # KL divergence: D_KL(target || actual)
    # = sum_k  p_target_k * log(p_target_k / p_actual_k)
    eps = 1e-8
    kl = p_target * (jnp.log(p_target + eps) - jnp.log(p_actual + eps))
    kl = kl.sum(axis=-1)   # sum over tokens K -> (B, Q)

    return jnp.mean(kl), scene_risk    # mean over batch and queries


def _make_loss_fn(
    sac_network: SACNetworks,
    alpha: float,
    discount: float,
    lambda_diversity: float = 0.0,
    lambda_safety: float = 0.0,
    safety_head_index: int = 0,
    lambda_consistency: float = 0.0,
    safety_tau: float = 0.5,
    unflatten_fn: callable = None,
) -> tuple[callable, callable]:
    """Define the loss functions for SAC.

    Args:
        sac_network: The SAC networks.
        alpha: Entropy regularization coefficient.
        discount: Discount factor.
        lambda_diversity: Weight for the attention head diversity loss.
        lambda_safety: Weight for the safety-grounded attention loss.
        safety_head_index: Index of the attention head to enforce safety alignment.
        lambda_consistency: Weight for the within-head consistency loss.
        safety_tau: Temperature for exponential TTC target distribution.
        unflatten_fn: Function to unflatten observations into structured features.
            Required when lambda_safety > 0 for TTC computation.

    Returns:
        A tuple containing the value loss and policy loss functions.

    """
    policy_network = sac_network.policy_network
    value_network = sac_network.value_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def compute_value_loss(
        value_params: datatypes.Params,
        policy_params: datatypes.Params,
        target_value_params: datatypes.Params,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> jax.Array:
        value_old_action = value_network.apply(value_params, transitions.observation, transitions.action)
        next_dist_params, _ = policy_network.apply(policy_params, transitions.next_observation)

        next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
        next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
        next_action = parametric_action_distribution.postprocess(next_action)

        next_value = value_network.apply(target_value_params, transitions.next_observation, next_action)
        next_v = jnp.min(next_value, axis=-1) - alpha * next_log_prob

        target_value = jax.lax.stop_gradient(transitions.reward + transitions.flag * discount * next_v)
        value_error = value_old_action - jnp.expand_dims(target_value, -1)
        value_loss = 0.5 * jnp.mean(jnp.square(value_error))

        return value_loss, {"value_loss": value_loss}

    def compute_policy_loss(
        policy_params: datatypes.Params,
        value_params: datatypes.Params,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> jax.Array:
        dist_params, encoder_attn_weights = policy_network.apply(policy_params, transitions.observation)

        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)

        value_action = value_network.apply(value_params, transitions.observation, action)
        min_value = jnp.min(value_action, axis=-1)
        original_policy_loss = jnp.mean(alpha * log_prob - min_value)

        # --- Forced Attention Specialization Losses ---

        # Diversity loss: penalize similar attention distributions across heads
        diversity_loss = _compute_diversity_loss(encoder_attn_weights)

        # Consistency loss: encourage within-head query agreement
        consistency_loss = _compute_consistency_loss(encoder_attn_weights)

        # Safety loss: KL-divergence against exponential TTC target
        # Gated on scene-level risk R > 0.2
        if lambda_safety > 0:
            safety_loss_raw, scene_risk = _compute_safety_loss(
                encoder_attn_weights,
                transitions.observation,
                safety_head_index,
                unflatten_fn,
                safety_tau=safety_tau,
            )
            # Gate on risk: only apply when R > 0.2, stop_gradient on gate
            risk_gate = jax.lax.stop_gradient((scene_risk > 0.2).astype(jnp.float32))
            safety_loss = safety_loss_raw * jnp.mean(risk_gate)
        else:
            safety_loss_raw = jnp.float32(0.0)
            safety_loss = jnp.float32(0.0)
            scene_risk = jnp.float32(0.0)
            risk_gate = jnp.float32(0.0)

        # Combined loss
        total_policy_loss = (
            original_policy_loss
            + lambda_diversity * diversity_loss
            + lambda_consistency * consistency_loss
            + lambda_safety * safety_loss
        )

        metrics = {
            "policy_loss_base": original_policy_loss,
            "diversity_loss": diversity_loss,
            "consistency_loss": consistency_loss,
            "safety_loss": safety_loss,
            "safety_loss_ungated": safety_loss_raw,
            "risk_gate_frac": jnp.mean(risk_gate),
            "scene_risk_mean": jnp.mean(scene_risk),
            "policy_loss_total": total_policy_loss,
        }

        return total_policy_loss, metrics

    return compute_value_loss, compute_policy_loss
