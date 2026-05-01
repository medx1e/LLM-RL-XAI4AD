"""Lambda annealing schedule for CBM concept supervision loss.

Provides a single pure-JAX function that computes the current concept
loss weight λ(t) given the current environment step.

Design principles:
  - Pure function: no state, no side effects, fully JIT-safe
  - Cosine annealing: keeps λ high early (concept learning phase),
    then smoothly decays (RL driving phase)
  - λ_min > 0: never removes concept supervision entirely, preserving
    the bottleneck guarantee
  - Based on environment steps (not gradient steps) for reproducibility
    independent of grad_updates_per_step

Reference:
  Cosine annealing schedule from Loshchilov & Hutter (2017),
  "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017.
  Applied here to loss weighting rather than learning rate.
"""

import jax.numpy as jnp


def cosine_anneal_lambda(
    env_step: jnp.ndarray,
    total_env_steps: int,
    lambda_max: float,
    lambda_min: float,
) -> jnp.ndarray:
    """Compute the annealed concept loss weight at a given environment step.

    Uses a half-cosine schedule: λ starts at lambda_max, stays high for
    the first ~30% of training, then smoothly decays toward lambda_min.

    Formula:
        λ(t) = λ_min + 0.5 * (λ_max - λ_min) * (1 + cos(π * t / T))

    Args:
        env_step:        Current environment step (JAX scalar, traced).
        total_env_steps: Total number of environment steps (Python int,
                         compile-time constant for JIT).
        lambda_max:      Starting λ value (Python float).
        lambda_min:      Floor λ value — never decays below this
                         (Python float, recommended ≥ 0.01).

    Returns:
        Current λ as a JAX scalar in [lambda_min, lambda_max].

    Example:
        # λ starts at 0.5, decays to 0.01 over 15M steps
        lam = cosine_anneal_lambda(step, 15_000_000, 0.5, 0.01)
    """
    # Clip to [0, 1] so we never extrapolate past the end of training
    progress = jnp.clip(env_step / total_env_steps, 0.0, 1.0)
    return lambda_min + 0.5 * (lambda_max - lambda_min) * (1.0 + jnp.cos(jnp.pi * progress))


def constant_lambda(
    env_step: jnp.ndarray,
    total_env_steps: int,
    lambda_val: float,
    lambda_min: float = 0.0,  # unused, kept for API uniformity
) -> jnp.ndarray:
    """No-op schedule that returns a constant λ (used when annealing is off).

    Exists so the factory can always call the same interface regardless
    of whether annealing is enabled, avoiding conditional logic in the
    JIT-compiled loss.

    Args:
        env_step:        Unused (kept for uniform API).
        total_env_steps: Unused (kept for uniform API).
        lambda_val:      Constant λ to return.
        lambda_min:      Unused (kept for uniform API).

    Returns:
        lambda_val as a JAX scalar.
    """
    return jnp.asarray(lambda_val)
