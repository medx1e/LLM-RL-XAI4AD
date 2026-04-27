"""Denormalization helpers — invert V-Max observation normalization.

V-Max applies specific normalization during feature extraction.  These
functions recover real-world units so that concept values are meaningful.

All functions are pure JAX and JIT-safe.
"""

import jax
import jax.numpy as jnp

from concepts.types import ObservationConfig


# ---- Object features (SDC + agents) ----------------------------------
# Layout after unflatten (last dim): xy(0:2) vel_xy(2:4) yaw(4) length(5) width(6)

OBJ_XY = slice(0, 2)
OBJ_VEL = slice(2, 4)
OBJ_YAW = slice(4, 5)
OBJ_LENGTH = slice(5, 6)
OBJ_WIDTH = slice(6, 7)


def denorm_xy(xy_norm: jax.Array, cfg: ObservationConfig) -> jax.Array:
    """xy_norm in [-1,1] → metres."""
    return xy_norm * cfg.max_meters


def denorm_vel(vel_norm: jax.Array, cfg: ObservationConfig) -> jax.Array:
    """vel_norm in [0,1] → m/s."""
    return vel_norm * cfg.max_speed


def denorm_size(size_norm: jax.Array, cfg: ObservationConfig) -> jax.Array:
    """size_norm → metres."""
    return size_norm * cfg.max_meters


# ---- Roadgraph features ---------------------------------------------
# Layout: xy(0:2) dir_xy(2:4)

RG_XY = slice(0, 2)
RG_DIR = slice(2, 4)


# ---- Traffic light features -----------------------------------------
# Layout: xy(0:2) state_onehot(2:10)

TL_XY = slice(0, 2)
TL_STATE = slice(2, 10)

# One-hot indices for red states (ARROW_STOP=0, STOP=3, FLASHING_STOP=6)
TL_RED_INDICES = jnp.array([0, 3, 6])
# Yellow
TL_YELLOW_INDICES = jnp.array([1, 4, 7])
# Green
TL_GREEN_INDICES = jnp.array([2, 5])


# ---- Path features ---------------------------------------------------
# Layout: xy(0:2)

PATH_XY = slice(0, 2)
