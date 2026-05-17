# Copyright 2025 - Causal XRL Component for V-Max XAI
"""
CounterfactualExplainer: A module for generating counterfactual explanations 
using parallel simulator rollouts to prove agent decisions were necessary.

This acts as a "Safety Shield" that answers "Why not?" questions by 
hallucinating alternative futures in parallel.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from waymax import datatypes
from waymax import metrics as waymax_metrics
from waymax.utils import geometry

from vmax.simulator import operations


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual generation."""
    # Rollout horizon
    horizon_steps: int = 20  # 2.0s at 10Hz
    
    # Longitudinal perturbations (acceleration in m/s²)
    accel_maintain: float = 0.0
    accel_accelerate: float = 2.0  # +2 m/s²
    accel_brake: float = -4.0  # Emergency brake -4 m/s²
    
    # Lateral perturbations (steering in radians, approximating lane nudges)
    # ~0.08 rad steering ≈ 0.5m lateral displacement over 2s at 10m/s
    steer_keep: float = 0.0
    steer_left: float = 0.08  # Nudge left
    steer_right: float = -0.08  # Nudge right


# Action labels for human-readable output
ACCEL_LABELS = {
    0: "Maintain Speed",
    1: "Accelerate (+2m/s²)",
    2: "Emergency Brake (-4m/s²)"
}

STEER_LABELS = {
    0: "Lane Keep",
    1: "Nudge Left",
    2: "Nudge Right"
}


# =============================================================================
# OUTCOME CLASSIFICATION
# =============================================================================

class OutcomeClassifier:
    """Classifies the outcome of a counterfactual rollout."""
    
    @staticmethod
    def check_collision(state: datatypes.SimulatorState) -> Tuple[bool, Optional[int]]:
        """
        Check if the SDC collided with any object.
        
        Returns:
            Tuple of (collision_occurred, colliding_object_id)
        """
        # Get overlap metric from waymax
        overlap_metric = waymax_metrics.OverlapMetric().compute(state)
        overlap = overlap_metric.value  # Shape: (..., num_objects)
        
        # Get SDC index
        is_sdc = state.object_metadata.is_sdc
        if is_sdc.ndim > 1:
            is_sdc = is_sdc[0]
        sdc_idx = operations.get_index(is_sdc)
        
        # Handle batch dimension if present
        if overlap.ndim > 1:
            # We assume we are looking at the first (only) batch item in sequential rollout
            overlap = overlap[0]
        
        # Check if SDC has overlap
        sdc_overlap = overlap[sdc_idx]
        has_collision = bool(sdc_overlap > 0)
        
        if has_collision:
            # Find which object SDC collided with using pairwise overlaps
            traj_5dof = state.current_sim_trajectory
            # Get current timestep trajectory for all objects: (..., N, 5)
            # stack_fields returns a 5DOF array
            bbox_5dof = state.current_sim_trajectory.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
            
            # Compute pairwise overlaps
            pairwise_overlaps = geometry.compute_pairwise_overlaps(bbox_5dof)
            
            # Handle batch dimension
            if pairwise_overlaps.ndim > 2:
                pairwise_overlaps = pairwise_overlaps[0]
            
            # Get overlaps for SDC
            sdc_overlaps = pairwise_overlaps[sdc_idx]  # Shape: (N,)
            
            # Find first overlapping object (excluding self)
            sdc_overlaps = sdc_overlaps.at[sdc_idx].set(False)
            colliding_indices = jnp.where(sdc_overlaps, size=1, fill_value=-1)
            colliding_idx = int(colliding_indices[0][0])
            
            return True, colliding_idx if colliding_idx != -1 else None
        
        return False, None

    
    @staticmethod
    def check_offroad(state: datatypes.SimulatorState) -> bool:
        """Check if the SDC went off-road."""
        offroad_metric = waymax_metrics.OffroadMetric().compute(state)
        offroad = offroad_metric.value  # Shape: (..., num_objects)
        
        is_sdc = state.object_metadata.is_sdc
        if is_sdc.ndim > 1:
            is_sdc = is_sdc[0]
        sdc_idx = operations.get_index(is_sdc)
        
        if offroad.ndim > 1:
            offroad = offroad[0]
            
        sdc_offroad = offroad[sdc_idx]
        
        return bool(sdc_offroad > 0)


# =============================================================================
# COUNTERFACTUAL EXPLAINER
# =============================================================================

class CounterfactualExplainer:
    """
    Generates counterfactual explanations by simulating alternative futures.
    
    Uses jax.vmap to run parallel rollouts of the V-Max simulator with
    different action perturbations, then classifies outcomes.
    """
    
    def __init__(self, config: Optional[CounterfactualConfig] = None,
                 action_grid: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the explainer with a set of action perturbations.
        
        Args:
            config: Rollout configuration.
            action_grid: Optional external action grid (from AdaptiveActionGrid).
                         If None, falls back to the default fixed 3x3 grid.
        """
        self.config = config or CounterfactualConfig()
        self.action_grid = action_grid if action_grid is not None else self._build_action_grid()
        
        # Performance Cache: Pre-instantiate metrics
        self._overlap_metric = waymax_metrics.OverlapMetric()
        self._offroad_metric = waymax_metrics.OffroadMetric()
        
        # Performance Cache: JIT kernel
        # We'll initialize this on the first run of explain_vectorized
        self._jitted_rollout_kernel = None
    
    def set_action_grid(self, action_grid: List[Dict[str, Any]]):
        """Replace the action grid dynamically (e.g. from AdaptiveActionGrid).
        
        NOTE: Invalidates the JIT cache since grid size may have changed.
        """
        self.action_grid = action_grid
        self._jitted_rollout_kernel = None  # Force recompilation
    
    def _ensure_batch_dim(self, tree):
        """Ensures all leaves in the tree have at least one batch dimension."""
        return jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, 0) if jnp.ndim(x) == 0 else x,
            tree
        )
    
    def _ensure_rank(self, x, rank: int):
        """Ensures an array has at least specified rank by adding leading dims."""
        while jnp.ndim(x) < rank:
            x = jnp.expand_dims(x, 0)
        return x
    
    def _build_action_grid(self) -> List[Dict[str, Any]]:
        """
        Build the grid of action perturbations.
        
        Returns:
            List of action dicts with accel, steer, and labels.
        """
        cfg = self.config
        accels = [cfg.accel_maintain, cfg.accel_accelerate, cfg.accel_brake]
        steers = [cfg.steer_keep, cfg.steer_left, cfg.steer_right]
        
        grid = []
        for ai, accel in enumerate(accels):
            for si, steer in enumerate(steers):
                grid.append({
                    "accel": accel,
                    "steer": steer,
                    "accel_idx": ai,
                    "steer_idx": si,
                    "label": f"{ACCEL_LABELS[ai]} + {STEER_LABELS[si]}"
                })
        return grid
    
    def _create_action(self, accel: float, steer: float) -> datatypes.Action:
        """Create a waymax Action from accel/steer values."""
        data = jnp.array([accel, steer])
        valid = jnp.ones((1,), dtype=jnp.bool_)
        return datatypes.Action(data=data, valid=valid)
    
    def _rollout_single(
        self, 
        base_env,
        initial_state: datatypes.SimulatorState,
        accel: float,
        steer: float
    ) -> Dict[str, Any]:
        """
        Perform a single counterfactual rollout.
        
        Args:
            base_env: The base PlanningAgentEnvironment (unwrapped)
            initial_state: Current simulator state (scalar, rank 0)
            accel: Acceleration value
            steer: Steering value
            
        Returns:
            Dict with outcome classification
        """
        # Create scalar action (Rank 0 data)
        # Action data: (2,)  valid: (1,)
        # Note: Waymax Action.data usually (..., 2). For scalar, it's (2,).
        action_data = jnp.array([accel, steer])
        action_valid = jnp.ones((1,), dtype=jnp.bool_)
        
        action = datatypes.Action(data=action_data, valid=action_valid)
        
        # Simulate forward for horizon_steps
        current_state = initial_state
        collision_occurred = False
        collision_id = None
        offroad_occurred = False
        first_collision_step = None
        
        for step in range(self.config.horizon_steps):
            # Step the base environment
            current_state = base_env.step(current_state, action)
            
            # Check outcomes at each step
            if not collision_occurred:
                collision, coll_id = self.classifier.check_collision(current_state)
                if collision:
                    collision_occurred = True
                    collision_id = coll_id
                    first_collision_step = step
                    
            if not offroad_occurred:
                offroad_occurred = self.classifier.check_offroad(current_state)
        
        # Determine final outcome
        if collision_occurred:
            outcome = "COLLISION"
            ttc = (first_collision_step + 1) * 0.1  # Convert steps to seconds
        elif offroad_occurred:
            outcome = "OFFROAD"
            ttc = None
        else:
            outcome = "SAFE"
            ttc = None
            
        return {
            "outcome": outcome,
            "collision_id": collision_id,
            "ttc": ttc,
            "offroad": offroad_occurred
        }
    
    def explain(
        self,
        env_transition,
        env,
        step_fn,
        chosen_action_idx: int = 0  # Index in action_grid (default: maintain + lane keep)
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations for the current state.
        
        Args:
            env_transition: Current EnvTransition from the simulator
            env: The environment object
            step_fn: The step function (jitted)
            chosen_action_idx: Index of the agent's chosen action in the grid
            
        Returns:
            Dictionary with chosen action and alternatives with outcomes
        """
        # Get the initial state to branch from
        state = env_transition.state
        
        # ENSURE SCALAR STATE (Rank 0): 
        # We slice [0] to remove any batch dimensions (e.g. (1,)).
        # Scalar state prevents Waymax from triggering vmap internally, avoiding "vmap rank 0" errors.
        if state.timestep.ndim > 0:
            initial_state = jax.tree_util.tree_map(lambda x: x[0], state)
        else:
            initial_state = state
            
        # Unwrap to base PlanningAgentEnvironment once
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Run rollouts for each alternative
        alternatives = []
        chosen_action = None
        
        for idx, action_cfg in enumerate(self.action_grid):
            result = self._rollout_single(
                base_env,
                initial_state,
                action_cfg["accel"],
                action_cfg["steer"]
            )
            
            entry = {
                "label": action_cfg["label"],
                "accel": action_cfg["accel"],
                "steer": action_cfg["steer"],
                **result
            }
            
            if idx == chosen_action_idx:
                chosen_action = entry
            else:
                alternatives.append(entry)
        
        return {
            "chosen_action": chosen_action,
            "alternatives": alternatives
        }

    
    def explain_vectorized(
        self,
        env_transition,
        env,
        step_fn,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations using vectorized (vmap) rollouts.
        """
        # 1. Normalize state to scalar first
        state = env_transition.state
        if state.timestep.ndim > 0:
            state = jax.tree_util.tree_map(lambda x: x[0], state)
        
        num_alternatives = len(self.action_grid)
        sdc_idx = int(operations.get_index(state.object_metadata.is_sdc))
        
        # 2. Batch the state: (num_alternatives, ...)
        batched_state = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, 0), num_alternatives, axis=0),
            state
        )
        
        # 3. Create batched actions
        accels = jnp.array([a["accel"] for a in self.action_grid])
        steers = jnp.array([a["steer"] for a in self.action_grid])
        batched_actions = datatypes.Action(
            data=jnp.stack([accels, steers], axis=-1),
            valid=jnp.ones((num_alternatives, 1), dtype=jnp.bool_)
        )
        
        # 4. Get the base environment step function
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # 5. Execute in parallel using cached JIT kernel
        if self._jitted_rollout_kernel is None:
            self._jitted_rollout_kernel = self._create_rollout_kernel(base_env.step)

        # Run parallel rollouts — now returns 4 tensors
        coll_mask, offroad_mask, coll_steps, threat_ids = self._jitted_rollout_kernel(
            batched_state,
            batched_actions,
            sdc_idx
        )

        # 6. Post-process results
        alternatives = []
        chosen_action = None

        # Device-get once for fast per-element access
        coll_mask_np = np.array(coll_mask)
        offroad_mask_np = np.array(offroad_mask)
        coll_steps_np = np.array(coll_steps)
        threat_ids_np = np.array(threat_ids)

        for idx, action_cfg in enumerate(self.action_grid):
            if coll_mask_np[idx]:
                outcome = "COLLISION"
                min_ttc = round((int(coll_steps_np[idx]) + 1) * 0.1, 2)
                # -1 sentinel means no valid agent found (shouldn't happen)
                raw_id = int(threat_ids_np[idx])
                threat_agent_id = raw_id if raw_id >= 0 else None
            elif offroad_mask_np[idx]:
                outcome = "OFFROAD"
                min_ttc = None
                threat_agent_id = None
            else:
                outcome = "SAFE"
                min_ttc = None
                threat_agent_id = None

            entry = {
                "label": action_cfg["label"],
                "accel": action_cfg["accel"],
                "steer": action_cfg["steer"],
                "outcome": outcome,
                "min_ttc": min_ttc,
                "threat_agent_id": threat_agent_id,
            }
            
            alternatives.append(entry)
                
        return {
            "alternatives": alternatives
        }

    def _create_rollout_kernel(self, step_fn):
        """
        Creates and JIT-compiles the parallel rollout kernel.
        """
        def single_rollout(initial_state, action, sdc_idx):
            curr_state = initial_state
            collision_occurred = jnp.array(False)
            first_collision_step = jnp.array(self.config.horizon_steps, dtype=jnp.int32)
            # -1 sentinel: no threat agent yet
            threat_agent_id = jnp.array(-1, dtype=jnp.int32)
            offroad_occurred = jnp.array(False)

            for step in range(self.config.horizon_steps):
                curr_state = step_fn(curr_state, action)

                # --- Collision check ---
                # overlap_val shape: (num_objects,)
                # 1.0 where object i overlaps ANY other object
                overlap_val = self._overlap_metric.compute(curr_state).value
                sdc_overlap = overlap_val[sdc_idx] > 0

                # Identify which agent caused the collision:
                # Zero-out the SDC's own slot so argmax picks another object.
                # The highest-overlap value after excluding SDC is the threat.
                overlap_excl_sdc = overlap_val.at[sdc_idx].set(0.0)
                candidate_id = jnp.argmax(overlap_excl_sdc).astype(jnp.int32)

                # Only record on the first collision step
                is_first_coll = sdc_overlap & ~collision_occurred
                first_collision_step = jnp.where(is_first_coll, step, first_collision_step)
                threat_agent_id = jnp.where(is_first_coll, candidate_id, threat_agent_id)
                collision_occurred = collision_occurred | sdc_overlap

                # --- Offroad check ---
                offroad_val = self._offroad_metric.compute(curr_state).value
                has_offroad = offroad_val[sdc_idx] > 0
                offroad_occurred = offroad_occurred | has_offroad

            return collision_occurred, offroad_occurred, first_collision_step, threat_agent_id

        # Return a jitted, vmapped version of the single rollout
        return jax.jit(jax.vmap(single_rollout, in_axes=(0, 0, None)))



# =============================================================================
# PROMPT FORMATTING UTILITIES
# =============================================================================

def format_counterfactual_rationale(cf_result: Dict[str, Any]) -> str:
    """
    Format counterfactual results into a human-readable decision rationale.
    
    Args:
        cf_result: Output from CounterfactualExplainer.explain()
        
    Returns:
        Formatted string for LLM prompt injection
    """
    if cf_result is None or "chosen_action" not in cf_result:
        return ""
    
    chosen = cf_result["chosen_action"]
    alternatives = cf_result["alternatives"]
    
    lines = [f"\nDecision Rationale: I chose to {chosen['label']}."]
    
    # Sort alternatives by severity (collisions first, then offroad, then safe)
    severity_order = {"COLLISION": 0, "OFFROAD": 1, "SAFE": 2}
    sorted_alts = sorted(alternatives, key=lambda x: severity_order.get(x["outcome"], 3))
    
    for alt in sorted_alts:
        outcome = alt["outcome"]
        label = alt["label"]
        
        if outcome == "COLLISION":
            ttc = alt.get("ttc", "?")
            coll_id = alt.get("collision_id", "unknown")
            detail = f"CRITICAL FAIL. Collision with Vehicle {coll_id} in {ttc}s."
        elif outcome == "OFFROAD":
            detail = "FAIL. Vehicle would leave roadway."
        else:
            detail = "SAFE."
            
        lines.append(f"  - Alternative Considered: {label} -> Result: {detail}")
    
    return "\n".join(lines)
