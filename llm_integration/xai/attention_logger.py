# Copyright 2025 Valeo.

"""Attention Logger for Head Specialization Analysis.

This module handles saving attention weights and metadata to disk asynchronously
during training for offline analysis of attention head specialization.
"""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Import vmax metrics utilities for semantic feature extraction
from vmax.simulator import constants, operations
from vmax.simulator.metrics import utils as metric_utils


class AttentionLogger:
    """Asynchronously log attention weights for offline Head Specialization Analysis.
    
    This logger is designed to:
    - Extract attention weights periodically during training (every N steps)
    - Subsample batches to reduce memory overhead (only log first n_samples)
    - Save asynchronously to avoid blocking the training loop
    - Store rich semantic features from vmax metrics for correlation analysis
    
    Args:
        output_dir: Directory to save attention logs.
        config: Dictionary containing encoder configuration for token boundary computation.
        max_workers: Number of threads for async I/O.
    
    Example:
        >>> logger = AttentionLogger("./attention_logs", encoder_config)
        >>> # During training loop:
        >>> if step % 1000 == 0:
        ...     attention_weights = model.apply(..., return_attention_weights=True)
        ...     logger.log(step, attention_weights, simulator_state, n_samples=4)
        >>> logger.close()
    """
    
    # Time horizon for TTC computation (seconds)
    TTC_HORIZON = 5.0
    
    def __init__(
        self,
        output_dir: str,
        config: Dict[str, Any],
        max_workers: int = 2
    ):
        self.output_dir = output_dir
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_futures = []
        
        # Pre-compute static token boundaries
        self.token_boundaries = self._compute_token_boundaries()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[AttentionLogger] Initialized. Output: {output_dir}")
        print(f"[AttentionLogger] Token boundaries: {self.token_boundaries}")
    
    def _compute_token_boundaries(self) -> Dict[str, tuple]:
        """Compute static token boundaries from encoder config.
        
        Token boundaries tell offline analysis which tokens correspond to
        which entity types (ego, vehicles, roadgraph, traffic lights, GPS path).
        
        Returns:
            Dictionary mapping entity type to (start_idx, end_idx) tuples.
        """
        # Extract dimensions from config (with sensible defaults)
        n_sdc_timesteps = self.config.get('n_sdc_timesteps', 11)
        num_objects = self.config.get('num_objects', 64)
        timestep_agent = n_sdc_timesteps  # Usually same as SDC timesteps
        roadgraph_top_k = self.config.get('roadgraph_top_k', 1000)
        num_traffic_lights = self.config.get('num_traffic_lights', 16)
        tl_timesteps = self.config.get('tl_timesteps', 1)
        gps_path_len = self.config.get('gps_path_len', 80)
        
        # Compute boundaries (tokens are flattened: n_entities * timesteps)
        n_sdc = 1 * n_sdc_timesteps  # SDC trajectory
        n_other = num_objects * timestep_agent  # Other agent trajectories
        n_road = roadgraph_top_k  # Roadgraph points
        n_tl = num_traffic_lights * tl_timesteps  # Traffic lights
        n_gps = gps_path_len  # GPS path points
        
        # Build cumulative boundaries
        boundaries = {
            'ego': (0, n_sdc),
            'vehicles': (n_sdc, n_sdc + n_other),
            'roadgraph': (n_sdc + n_other, n_sdc + n_other + n_road),
            'traffic_lights': (n_sdc + n_other + n_road, n_sdc + n_other + n_road + n_tl),
            'gps_path': (n_sdc + n_other + n_road + n_tl, n_sdc + n_other + n_road + n_tl + n_gps),
        }
        
        return boundaries
    
    def log(
        self,
        step: int,
        attention_weights: Dict[str, Any],
        simulator_state: Any,
        n_samples: int = 4
    ):
        """Log attention weights with corresponding observations and semantic features.
        
        Args:
            step: Training step number.
            attention_weights: Dict of attention tensors from WayformerEncoder.
                Keys like 'sdc_traj/cross_attn_0', 'other_traj/self_attn_1', etc.
            simulator_state: SimulatorState containing trajectory and metadata.
            n_samples: Number of samples to log from the batch (to save space).
        """
        # Subsample attention weights (only first n_samples)
        # Convert JAX arrays to NumPy for pickling
        subsampled_attention = {}
        for key, value in attention_weights.items():
            if hasattr(value, '__getitem__'):
                arr = np.array(jax.device_get(value[:n_samples]))
                subsampled_attention[key] = arr
            else:
                subsampled_attention[key] = np.array(jax.device_get(value))
        
        # Extract semantic features using vmax metrics
        semantic_features = self._extract_semantic_features(simulator_state, n_samples)
        
        # Prepare log data
        log_data = {
            'step': step,
            'attention_weights': subsampled_attention,
            'semantic_features': semantic_features,
            'token_boundaries': self.token_boundaries,
            'config': {
                'num_heads': self.config.get('latent_num_heads', 4),
                'num_latents': self.config.get('num_latents', 64),
                'fusion_type': self.config.get('fusion_type', 'late'),
                'head_features': self.config.get('latent_head_features', 64),
            },
            # Summary for quick inspection without loading full arrays
            'attention_summary': {
                key: {
                    'shape': list(arr.shape),
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                    'max': float(arr.max()),
                }
                for key, arr in subsampled_attention.items()
            } if subsampled_attention else {}
        }
        
        # Submit async save
        future = self.executor.submit(self._save_log, step, log_data)
        self._pending_futures.append(future)
        
        # Clean up completed futures periodically
        self._cleanup_futures()
    
    def _extract_semantic_features(
        self,
        simulator_state: Any,
        n_samples: int
    ) -> Dict[str, np.ndarray]:
        """Extract semantic features from simulator state using vmax metrics.
        
        Extracts driving-relevant features for correlating with attention heads
        during offline Head Specialization Analysis (HSI).
        
        Features extracted:
        - Per-agent TTC (Time-To-Collision)
        - Per-agent distance to ego
        - Per-agent closing speed
        - Per-agent spatial relationship (ahead/behind/lateral)
        - Ego comfort metrics (acceleration, jerk, yaw rate)
        - Distance to lane centers
        
        Args:
            simulator_state: SimulatorState from vmax/waymax.
            n_samples: Number of samples to extract.
            
        Returns:
            Dictionary of semantic feature arrays.
        """
        semantic_features = {}
        
        try:
            # Handle batched vs single state
            state = simulator_state
            
            # Get current trajectory slice
            # timestep can be multi-dimensional for batched training (e.g., [num_devices, batch_size])
            timestep = jax.device_get(state.timestep)
            timestep = np.array(timestep).flatten()
            timestep = int(timestep[0])  # Use first timestep for all samples
            
            traj = state.sim_trajectory
            
            # Get SDC index
            is_sdc = jax.device_get(state.object_metadata.is_sdc)
            if is_sdc.ndim > 1:
                is_sdc = is_sdc[0]  # Take first batch
            sdc_index = int(np.argmax(is_sdc))
            
            # Extract current positions and velocities
            # Shape: (batch, num_objects, timesteps) or (num_objects, timesteps)
            x = np.array(jax.device_get(traj.x))
            y = np.array(jax.device_get(traj.y))
            vel_x = np.array(jax.device_get(traj.vel_x))
            vel_y = np.array(jax.device_get(traj.vel_y))
            yaw = np.array(jax.device_get(traj.yaw))
            valid = np.array(jax.device_get(traj.valid))
            
            # Handle batch dimension
            if x.ndim == 3:  # (batch, objects, time)
                x = x[:n_samples, :, timestep]
                y = y[:n_samples, :, timestep]
                vel_x = vel_x[:n_samples, :, timestep]
                vel_y = vel_y[:n_samples, :, timestep]
                yaw = yaw[:n_samples, :, timestep]
                valid = valid[:n_samples, :, timestep]
            elif x.ndim == 2:  # (objects, time)
                x = x[:, timestep]
                y = y[:, timestep]
                vel_x = vel_x[:, timestep]
                vel_y = vel_y[:, timestep]
                yaw = yaw[:, timestep]
                valid = valid[:, timestep]
            
            # Store basic trajectory info
            semantic_features['positions_x'] = x
            semantic_features['positions_y'] = y
            semantic_features['velocities_x'] = vel_x
            semantic_features['velocities_y'] = vel_y
            semantic_features['yaw'] = yaw
            semantic_features['valid'] = valid
            semantic_features['sdc_index'] = sdc_index
            
            # ===== Compute per-agent semantic features =====
            
            # 1. Distance to ego (per agent)
            if x.ndim == 1:  # Single sample
                ego_x, ego_y = x[sdc_index], y[sdc_index]
                distances = np.sqrt((x - ego_x)**2 + (y - ego_y)**2)
                semantic_features['distance_to_ego'] = distances
            else:  # Batched
                ego_x = x[:, sdc_index:sdc_index+1]
                ego_y = y[:, sdc_index:sdc_index+1]
                distances = np.sqrt((x - ego_x)**2 + (y - ego_y)**2)
                semantic_features['distance_to_ego'] = distances
            
            # 2. Closing speed (per agent)
            closing_speeds = self._compute_closing_speeds(
                x, y, vel_x, vel_y, sdc_index
            )
            semantic_features['closing_speed'] = closing_speeds
            
            # 3. Time-to-collision estimate (per agent)
            ttc = self._compute_simple_ttc(distances, closing_speeds)
            semantic_features['ttc'] = ttc
            
            # 4. Spatial relationship (ahead/behind/lateral)
            if x.ndim == 1:
                ego_yaw = yaw[sdc_index]
                spatial_rel = self._compute_spatial_relationship(
                    x, y, x[sdc_index], y[sdc_index], ego_yaw
                )
            else:
                ego_yaw = yaw[:, sdc_index]
                spatial_rels = []
                for i in range(min(n_samples, x.shape[0])):
                    rel = self._compute_spatial_relationship(
                        x[i], y[i], x[i, sdc_index], y[i, sdc_index], ego_yaw[i]
                    )
                    spatial_rels.append(rel)
                spatial_rel = np.stack(spatial_rels, axis=0)
            
            semantic_features['is_ahead'] = spatial_rel[..., 0] if spatial_rel.ndim > 1 else spatial_rel[0]
            semantic_features['is_behind'] = spatial_rel[..., 1] if spatial_rel.ndim > 1 else spatial_rel[1]
            semantic_features['is_left'] = spatial_rel[..., 2] if spatial_rel.ndim > 1 else spatial_rel[2]
            semantic_features['is_right'] = spatial_rel[..., 3] if spatial_rel.ndim > 1 else spatial_rel[3]
            
            # 5. Object types
            if hasattr(state.object_metadata, 'object_types'):
                obj_types = np.array(jax.device_get(state.object_metadata.object_types))
                if obj_types.ndim > 1:
                    obj_types = obj_types[:n_samples]
                semantic_features['object_types'] = obj_types
            
            # 6. Ego speed
            if x.ndim == 1:
                ego_speed = np.sqrt(vel_x[sdc_index]**2 + vel_y[sdc_index]**2)
            else:
                ego_speed = np.sqrt(vel_x[:, sdc_index]**2 + vel_y[:, sdc_index]**2)
            semantic_features['ego_speed'] = ego_speed
            
            # 7. Agent speeds
            speeds = np.sqrt(vel_x**2 + vel_y**2)
            semantic_features['agent_speeds'] = speeds
            
        except Exception as e:
            print(f"[AttentionLogger] Warning: Could not extract semantic features: {e}")
            import traceback
            traceback.print_exc()
        
        return semantic_features
    
    def _compute_closing_speeds(
        self,
        x: np.ndarray,
        y: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        sdc_index: int
    ) -> np.ndarray:
        """Compute closing speed for each agent relative to ego.
        
        Closing speed > 0 means approaching, < 0 means separating.
        
        Args:
            x, y: Positions of all agents.
            vel_x, vel_y: Velocities of all agents.
            sdc_index: Index of ego vehicle.
            
        Returns:
            Array of closing speeds for each agent.
        """
        if x.ndim == 1:
            # Single sample
            ego_x, ego_y = x[sdc_index], y[sdc_index]
            ego_vx, ego_vy = vel_x[sdc_index], vel_y[sdc_index]
            
            # Relative position vector (from ego to agent)
            rel_x = x - ego_x
            rel_y = y - ego_y
            dist = np.sqrt(rel_x**2 + rel_y**2) + 1e-6
            
            # Relative velocity
            rel_vx = vel_x - ego_vx
            rel_vy = vel_y - ego_vy
            
            # Closing speed = -dot(rel_pos, rel_vel) / dist
            closing = -(rel_x * rel_vx + rel_y * rel_vy) / dist
            return closing
        else:
            # Batched
            ego_x = x[:, sdc_index:sdc_index+1]
            ego_y = y[:, sdc_index:sdc_index+1]
            ego_vx = vel_x[:, sdc_index:sdc_index+1]
            ego_vy = vel_y[:, sdc_index:sdc_index+1]
            
            rel_x = x - ego_x
            rel_y = y - ego_y
            dist = np.sqrt(rel_x**2 + rel_y**2) + 1e-6
            
            rel_vx = vel_x - ego_vx
            rel_vy = vel_y - ego_vy
            
            closing = -(rel_x * rel_vx + rel_y * rel_vy) / dist
            return closing
    
    def _compute_simple_ttc(
        self,
        distances: np.ndarray,
        closing_speeds: np.ndarray
    ) -> np.ndarray:
        """Compute simple TTC = distance / closing_speed.
        
        Only valid when closing_speed > 0 (approaching).
        Returns TTC_HORIZON for non-approaching agents.
        """
        ttc = np.where(
            closing_speeds > 0.5,  # Only for approaching agents
            distances / (closing_speeds + 1e-6),
            self.TTC_HORIZON
        )
        return np.clip(ttc, 0, self.TTC_HORIZON)
    
    def _compute_spatial_relationship(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ego_x: float,
        ego_y: float,
        ego_yaw: float
    ) -> np.ndarray:
        """Compute spatial relationship (ahead/behind/left/right) for each agent.
        
        Returns:
            Array of shape (num_agents, 4) with [is_ahead, is_behind, is_left, is_right]
        """
        # Transform to ego-local coordinates
        dx = x - ego_x
        dy = y - ego_y
        
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        
        # Local x = forward, local y = left
        x_local = dx * cos_yaw - dy * sin_yaw
        y_local = dx * sin_yaw + dy * cos_yaw
        
        # Thresholds
        is_ahead = x_local > 2.0
        is_behind = x_local < -2.0
        is_left = y_local > 1.0
        is_right = y_local < -1.0
        
        return np.stack([is_ahead, is_behind, is_left, is_right], axis=-1)
    
    def _save_log(self, step: int, data: Dict[str, Any]):
        """Save log data to disk (runs in background thread)."""
        path = os.path.join(self.output_dir, f"attention_log_{step:08d}.pkl")
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[AttentionLogger] Error saving step {step}: {e}")
    
    def _cleanup_futures(self):
        """Remove completed futures from the list."""
        self._pending_futures = [f for f in self._pending_futures if not f.done()]
    
    def close(self):
        """Wait for pending saves and shutdown executor."""
        print(f"[AttentionLogger] Waiting for {len(self._pending_futures)} pending saves...")
        # Wait for all pending saves
        for future in self._pending_futures:
            try:
                future.result(timeout=30)
            except Exception as e:
                print(f"[AttentionLogger] Error during save: {e}")
        
        self.executor.shutdown(wait=True)
        print("[AttentionLogger] Closed.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

