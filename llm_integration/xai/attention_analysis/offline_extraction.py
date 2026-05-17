"""Offline Attention + Semantic Feature Extraction for HSI Analysis.

This script extracts attention weights and semantic features from scenarios
using saved model checkpoints, enabling offline Head Specialization Analysis.

Usage:
    python offline_extraction.py --run_dir ../../runs/PPO_VEC_WAYFORMER \\
                                  --dataset ../../training.tfrecord \\
                                  --n_scenarios 100 \\
                                  --output_dir ./extractions

    # For evolution analysis (multiple checkpoints):
    python offline_extraction.py --run_dir ../../runs/PPO_VEC_WAYFORMER \\
                                  --checkpoints model_10000.pkl model_50000.pkl model_final.pkl \\
                                  --n_scenarios 50
"""

import argparse
import glob
import os
import sys
import pickle
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path to allow importing from 'xai'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np
from waymax import dynamics
from waymax import datatypes as waymax_datatypes
from waymax.agents import expert

from vmax.simulator import operations

from vmax.simulator import make_env_for_evaluation, datasets, make_data_generator
from vmax.simulator.wrappers.interfaces.brax import EnvTransition
from vmax.agents.learning.reinforcement.ppo import ppo_factory
from vmax.agents.networks import network_utils
from vmax.scripts.evaluate.utils import load_params, load_yaml_config

# WayformerEncoder always returns (output, attn_weights) tuple
from vmax.agents.networks.encoders.wayformer import WayformerEncoder


class OfflineExtractor:
    """Extract attention weights and semantic features from scenarios.
    
    This class loads a trained model and processes scenarios to extract:
    1. Raw attention weights from the Wayformer encoder
    2. Aggregated per-vehicle attention for each head
    3. Semantic features (TTC, distance, closing speed, spatial relationships)
    4. Token boundaries for mapping attention to semantic entities
    """
    
    # Time horizon for TTC computation (seconds)
    TTC_HORIZON = 5.0
    
    # Maximum timesteps for multi-timestep rollout
    MAX_TIMESTEPS = 80
    
    # Attention module keys (per-module, late-fusion architecture)
    MODULE_KEYS = [
        'sdc_traj',
        'other_traj',
        'roadgraph',
        'traffic_lights',
        'gps_path',
    ]
    
    def __init__(self, run_dir: str, dataset_path: str, checkpoint_name: str = "model_final.pkl"):
        """Initialize the extractor.
        
        Args:
            run_dir: Path to the training run directory.
            dataset_path: Path to the dataset (tfrecord or dataset name).
            checkpoint_name: Name of the checkpoint file to load.
        """
        self.run_dir = run_dir
        self.dataset_path = dataset_path
        self.checkpoint_name = checkpoint_name
        
        # Paths
        self.config_path = os.path.join(run_dir, ".hydra/config.yaml")
        self.model_dir = os.path.join(run_dir, "model")
        
        # Will be set during setup
        self.env = None
        self.encoder = None
        self.params = None
        self.config = None
        self.encoder_params = None
        self.token_boundaries = None
        
    def setup(self):
        """Load model, config, and create environment."""
        print(f"[OfflineExtractor] Loading config from {self.config_path}")
        self.config = load_yaml_config(self.config_path)
        
        # Flatten nested config for compatibility
        if "algorithm" in self.config and "network" in self.config["algorithm"]:
            self.config["policy"] = self.config["algorithm"]["network"].get("policy", {})
            self.config["value"] = self.config["algorithm"]["network"].get("value", {})
            self.config["action_distribution"] = self.config["algorithm"]["network"].get("action_distribution", "gaussian")
        if "network" in self.config and "encoder" in self.config["network"]:
            self.config["encoder"] = self.config["network"]["encoder"]
        
        # Force vec observation type
        obs_type = self.config.get("observation_type", "vec")
        if obs_type != "vec":
            print(f"[OfflineExtractor] Warning: Forcing observation_type='vec' (was '{obs_type}')")
            obs_type = "vec"
        
        # Create environment
        print("[OfflineExtractor] Creating environment...")
        self.env = make_env_for_evaluation(
            max_num_objects=self.config.get("max_num_objects", 64),
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            sdc_paths_from_data=not self.config.get("waymo_dataset", False),
            observation_type=obs_type,
            observation_config=self.config.get("observation_config", {}),
            termination_keys=self.config.get("termination_keys", ["offroad", "overlap"]),
        )
        
        # Create encoder with attention extraction enabled
        unflatten_fn = self.env.get_wrapper_attr("features_extractor").unflatten_features
        encoder_cfg = network_utils.parse_config(self.config["encoder"], "encoder")
        encoder_cfg = network_utils.convert_to_dict_with_activation_fn(encoder_cfg)
        
        # WayformerEncoder always returns (output, attn_weights) tuple
        self.encoder = WayformerEncoder(
            unflatten_fn=unflatten_fn,
            **encoder_cfg
        )
        print(f"[OfflineExtractor] Encoder: {self.encoder}")
        
        # Compute token boundaries
        self.token_boundaries = self._compute_token_boundaries()
        print(f"[OfflineExtractor] Token boundaries: {self.token_boundaries}")
        
        # Load model parameters
        self.load_checkpoint(self.checkpoint_name)
        
        return self
    
    def load_checkpoint(self, checkpoint_name: str):
        """Load a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint file (e.g., 'model_final.pkl').
        """
        model_path = os.path.join(self.model_dir, checkpoint_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        print(f"[OfflineExtractor] Loading checkpoint: {model_path}")
        
        # Update checkpoint_name so metadata is correct when saving
        self.checkpoint_name = checkpoint_name
        
        self.params = load_params(model_path)
        
        # Debug: print full policy params structure
        print(f"\n[DEBUG] Policy params type: {type(self.params.policy)}")
        if hasattr(self.params.policy, 'keys'):
            print(f"[DEBUG] Policy top-level keys: {list(self.params.policy.keys())}")
        if 'params' in self.params.policy:
            print(f"[DEBUG] Policy['params'] keys: {list(self.params.policy['params'].keys())}")
            if 'encoder_layer' in self.params.policy['params']:
                enc_params = self.params.policy['params']['encoder_layer']
                print(f"[DEBUG] encoder_layer keys: {list(enc_params.keys())}")
                # Show nested structure of one attention module
                if 'other_traj_attention' in enc_params:
                    ota = enc_params['other_traj_attention']
                    print(f"[DEBUG] other_traj_attention keys: {list(ota.keys())}")
                    if 'attn_0' in ota:
                        print(f"[DEBUG] attn_0 keys: {list(ota['attn_0'].keys())}")
                        # Check if Dense layers have non-zero weights
                        for dense_key in list(ota['attn_0'].keys())[:2]:
                            params = ota['attn_0'][dense_key]
                            if 'kernel' in params:
                                kernel = np.array(params['kernel'])
                                print(f"[DEBUG] attn_0/{dense_key}/kernel: shape={kernel.shape}, "
                                      f"mean={kernel.mean():.6f}, std={kernel.std():.6f}")
        
        # Extract encoder parameters
        if 'params' in self.params.policy and 'encoder_layer' in self.params.policy['params']:
            self.encoder_params = self.params.policy['params']['encoder_layer']
        else:
            raise ValueError("Could not find 'encoder_layer' in policy params")
        
        return self
    
    def _compute_token_boundaries(self) -> Dict[str, Tuple[int, int]]:
        """Compute token boundaries from observation config.
        
        Token boundaries map entity types to token index ranges.
        """
        obs_cfg = self.config.get("observation_config", {})
        
        n_timesteps = obs_cfg.get("obs_past_num_steps", 5)
        n_sdc = 1 * n_timesteps  # SDC trajectory tokens
        
        obj_cfg = obs_cfg.get("objects", {})
        n_vehicles = obj_cfg.get("num_closest_objects", 8)
        n_other = n_vehicles * n_timesteps  # Other vehicle tokens
        
        rg_cfg = obs_cfg.get("roadgraphs", {})
        n_roadgraph = rg_cfg.get("roadgraph_top_k", 200)
        
        tl_cfg = obs_cfg.get("traffic_lights", {})
        n_traffic_lights = tl_cfg.get("num_closest_traffic_lights", 5)
        tl_timesteps = n_timesteps  # Assume same as agent timesteps
        n_tl = n_traffic_lights * tl_timesteps
        
        path_cfg = obs_cfg.get("path_target", {})
        n_gps = path_cfg.get("num_points", 3)
        
        # Cumulative boundaries
        boundaries = {
            'sdc_traj': (0, n_sdc),
            'other_traj': (n_sdc, n_sdc + n_other),
            'roadgraph': (n_sdc + n_other, n_sdc + n_other + n_roadgraph),
            'traffic_lights': (n_sdc + n_other + n_roadgraph, n_sdc + n_other + n_roadgraph + n_tl),
            'gps_path': (n_sdc + n_other + n_roadgraph + n_tl, n_sdc + n_other + n_roadgraph + n_tl + n_gps),
        }
        
        # Store counts for aggregation
        self._n_vehicles = n_vehicles
        self._n_timesteps = n_timesteps
        self._n_roadgraph = n_roadgraph
        self._n_traffic_lights = n_traffic_lights
        
        return boundaries
    
    def extract_attention(self, scenario, debug: bool = True) -> Dict[str, np.ndarray]:
        """Extract attention weights from a scenario.
        
        Args:
            scenario: Simulator state (single scenario, no batch dim).
            debug: If True, print debug information.
            
        Returns:
            Dictionary of attention weight arrays.
        """
        # Get observation
        obs = self.env.observe(scenario)
        if isinstance(obs, tuple):
            raise ValueError("Observation is a tuple, expected flattened array")
        
        # Add batch dimension if missing (required by WayformerEncoder)
        if obs.ndim == 1:
            obs = jnp.expand_dims(obs, axis=0)  # (obs_dim,) -> (1, obs_dim)
        
        if debug:
            obs_np = np.array(jax.device_get(obs))
            print(f"\n[DEBUG] Observation shape (with batch): {obs_np.shape}")
            print(f"[DEBUG] Obs min: {obs_np.min():.4f}, max: {obs_np.max():.4f}, mean: {obs_np.mean():.4f}")
            print(f"[DEBUG] Obs std: {obs_np.std():.4f}")
            
            # Debug: check what unflatten_fn produces
            unflatten_fn = self.env.get_wrapper_attr("features_extractor").unflatten_features
            features, masks = unflatten_fn(obs)
            print(f"\n[DEBUG] After unflatten_fn:")
            print(f"[DEBUG] other_traj_features shape: {features[1].shape}")
            feat_arr = np.array(features[1])
            print(f"[DEBUG] other_traj_features min: {feat_arr.min():.4f}, max: {feat_arr.max():.4f}")
            print(f"[DEBUG] other_traj_features mean: {feat_arr.mean():.4f}, std: {feat_arr.std():.4f}")
            print(f"[DEBUG] other_traj_features non-zero: {np.count_nonzero(feat_arr)} / {feat_arr.size}")
            print(f"[DEBUG] other_traj_valid_mask shape: {masks[1].shape}")
            mask_arr = np.array(masks[1])
            print(f"[DEBUG] other_traj_valid_mask True count: {mask_arr.sum()} / {mask_arr.size}")
            
            # Debug: check the RAW feature array BEFORE stripping valid column
            raw_feat = vectorized_obs_check = obs
            total_per_obj = 8  # 8 features based on config
            sdc_size = 1 * 5 * total_per_obj
            other_start = sdc_size
            other_size = 8 * 5 * total_per_obj
            raw_other = np.array(obs[..., other_start:other_start + other_size]).reshape(1, 8, 5, total_per_obj)
            print(f"\n[DEBUG] Raw other_traj (before unflatten strips valid):")
            print(f"[DEBUG] Shape: {raw_other.shape}")
            print(f"[DEBUG] Last column (valid?): {raw_other[0, :, 0, -1]}")  # Should be 1s for valid objects
            
            # Check encoder params structure
            print(f"\n[DEBUG] Encoder params keys: {list(self.encoder_params.keys())[:10]}...")
        
        # Forward pass with attention extraction
        @jax.jit
        def forward_with_attention(e_params, observation):
            latent, attention_weights = self.encoder.apply(
                {'params': e_params},
                observation
            )
            return latent, attention_weights
        
        _, attn_weights = forward_with_attention(self.encoder_params, obs)
        
        # Convert to numpy
        attn_weights = jax.tree_util.tree_map(
            lambda x: np.array(jax.device_get(x)),
            attn_weights
        )
        
        if debug:
            print(f"[DEBUG] Attention weight keys: {list(attn_weights.keys())}")
        
        return attn_weights
    
    def aggregate_vehicle_attention(self, attention_weights: Dict[str, np.ndarray], debug: bool = True) -> np.ndarray:
        """Aggregate attention weights per vehicle per head.
        
        Args:
            attention_weights: Raw attention weights from encoder.
            debug: If True, print debug information.
            
        Returns:
            Array of shape (n_heads, n_vehicles) with aggregated attention.
        """
        # Get the cross-attention to other vehicles
        key = "other_traj/cross_attn_0"
        if key not in attention_weights:
            print(f"[OfflineExtractor] Warning: '{key}' not found in attention weights")
            print(f"[OfflineExtractor] Available keys: {list(attention_weights.keys())}")
            return None
        
        # Shape: (batch, n_latents, n_tokens, n_heads)
        attn = attention_weights[key]
        
        if debug:
            print(f"\n[DEBUG] Raw attention shape: {attn.shape}")
            print(f"[DEBUG] Min: {attn.min():.6f}, Max: {attn.max():.6f}, Mean: {attn.mean():.6f}")
            print(f"[DEBUG] Std: {attn.std():.6f}")
        
        # Remove batch dimension if present
        if attn.ndim == 4:
            attn = attn[0]  # (n_latents, n_tokens, n_heads)
        
        n_latents, n_tokens, n_heads = attn.shape
        n_vehicles = self._n_vehicles
        n_timesteps = self._n_timesteps
        
        if debug:
            print(f"[DEBUG] After squeeze: shape={attn.shape}")
            print(f"[DEBUG] n_latents={n_latents}, n_tokens={n_tokens}, n_heads={n_heads}")
            print(f"[DEBUG] Expected: n_vehicles={n_vehicles}, n_timesteps={n_timesteps}, product={n_vehicles * n_timesteps}")
        
        # Verify token count matches
        expected_tokens = n_vehicles * n_timesteps
        if n_tokens != expected_tokens:
            print(f"[OfflineExtractor] Warning: Token count mismatch. Got {n_tokens}, expected {expected_tokens}")
            return None
        
        # Reshape: (n_latents, n_tokens, n_heads) -> (n_latents, n_vehicles, n_timesteps, n_heads)
        attn_reshaped = attn.reshape(n_latents, n_vehicles, n_timesteps, n_heads)
        
        if debug:
            print(f"[DEBUG] Reshaped: {attn_reshaped.shape}")
            # Check if attention per token varies by vehicle
            sample_latent_0_head_0 = attn_reshaped[0, :, :, 0]  # (n_vehicles, n_timesteps)
            print(f"[DEBUG] Sample attention (latent 0, head 0):")
            print(f"        Per-vehicle sums: {sample_latent_0_head_0.sum(axis=1)}")
        
        # Sum over timesteps: -> (n_latents, n_vehicles, n_heads)
        attn_per_vehicle = attn_reshaped.sum(axis=2)
        
        if debug:
            print(f"[DEBUG] After sum over timesteps: {attn_per_vehicle.shape}")
        
        # Sum over latents: -> (n_vehicles, n_heads)
        attn_final = attn_per_vehicle.sum(axis=0)
        
        if debug:
            print(f"[DEBUG] After sum over latents: {attn_final.shape}")
            print(f"[DEBUG] Final per-vehicle attention:\n{attn_final}")
        
        # Transpose to (n_heads, n_vehicles) for easier correlation
        attn_final = attn_final.T
        
        return attn_final
    
    def extract_semantic_features(self, scenario) -> Dict[str, np.ndarray]:
        """Extract semantic features from a scenario.
        
        Adapted from AttentionLogger._extract_semantic_features().
        
        Args:
            scenario: Simulator state.
            
        Returns:
            Dictionary of semantic feature arrays.
        """
        semantic_features = {}
        
        try:
            state = scenario
            
            # Get current timestep
            timestep = np.array(jax.device_get(state.timestep)).flatten()
            timestep = int(timestep[0]) if len(timestep) > 0 else 0
            
            traj = state.sim_trajectory
            
            # Get SDC index
            is_sdc = np.array(jax.device_get(state.object_metadata.is_sdc))
            if is_sdc.ndim > 1:
                is_sdc = is_sdc[0]
            sdc_index = int(np.argmax(is_sdc))
            
            # Extract trajectory data at current timestep
            x = np.array(jax.device_get(traj.x))
            y = np.array(jax.device_get(traj.y))
            vel_x = np.array(jax.device_get(traj.vel_x))
            vel_y = np.array(jax.device_get(traj.vel_y))
            yaw = np.array(jax.device_get(traj.yaw))
            valid = np.array(jax.device_get(traj.valid))
            
            # Handle dimensions
            if x.ndim == 3:  # (batch, objects, time)
                x = x[0, :, timestep]
                y = y[0, :, timestep]
                vel_x = vel_x[0, :, timestep]
                vel_y = vel_y[0, :, timestep]
                yaw = yaw[0, :, timestep]
                valid = valid[0, :, timestep]
            elif x.ndim == 2:  # (objects, time)
                x = x[:, timestep]
                y = y[:, timestep]
                vel_x = vel_x[:, timestep]
                vel_y = vel_y[:, timestep]
                yaw = yaw[:, timestep]
                valid = valid[:, timestep]
            
            # Limit to n_vehicles closest (matching observation)
            n_vehicles = self._n_vehicles
            
            # Store metadata
            semantic_features['sdc_index'] = sdc_index
            semantic_features['timestep'] = timestep
            semantic_features['valid'] = valid[:n_vehicles]
            semantic_features['positions_x'] = x[:n_vehicles]
            semantic_features['positions_y'] = y[:n_vehicles]
            
            # === Compute per-vehicle semantic features ===
            
            # 1. Distance to ego
            ego_x, ego_y = x[sdc_index], y[sdc_index]
            distances = np.sqrt((x[:n_vehicles] - ego_x)**2 + (y[:n_vehicles] - ego_y)**2)
            semantic_features['distance_to_ego'] = distances
            
            # 2. Closing speed
            ego_vx, ego_vy = vel_x[sdc_index], vel_y[sdc_index]
            rel_x = x[:n_vehicles] - ego_x
            rel_y = y[:n_vehicles] - ego_y
            dist = np.sqrt(rel_x**2 + rel_y**2) + 1e-6
            rel_vx = vel_x[:n_vehicles] - ego_vx
            rel_vy = vel_y[:n_vehicles] - ego_vy
            closing_speeds = -(rel_x * rel_vx + rel_y * rel_vy) / dist
            semantic_features['closing_speed'] = closing_speeds
            
            # 3. Time-to-collision (TTC)
            ttc = np.where(
                closing_speeds > 0.5,
                distances / (closing_speeds + 1e-6),
                self.TTC_HORIZON
            )
            ttc = np.clip(ttc, 0, self.TTC_HORIZON)
            semantic_features['ttc'] = ttc
            
            # 4. Spatial relationships
            ego_yaw = yaw[sdc_index]
            dx = x[:n_vehicles] - ego_x
            dy = y[:n_vehicles] - ego_y
            cos_yaw = np.cos(-ego_yaw)
            sin_yaw = np.sin(-ego_yaw)
            x_local = dx * cos_yaw - dy * sin_yaw
            y_local = dx * sin_yaw + dy * cos_yaw
            
            semantic_features['is_ahead'] = (x_local > 2.0).astype(float)
            semantic_features['is_behind'] = (x_local < -2.0).astype(float)
            semantic_features['is_left'] = (y_local > 1.0).astype(float)
            semantic_features['is_right'] = (y_local < -1.0).astype(float)
            
            # 5. Agent speeds
            speeds = np.sqrt(vel_x[:n_vehicles]**2 + vel_y[:n_vehicles]**2)
            semantic_features['agent_speeds'] = speeds
            
            # 6. Ego speed
            ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
            semantic_features['ego_speed'] = ego_speed
            
            # 7. Object types (if available)
            if hasattr(state.object_metadata, 'object_types'):
                obj_types = np.array(jax.device_get(state.object_metadata.object_types))
                if obj_types.ndim > 1:
                    obj_types = obj_types[0]
                semantic_features['object_types'] = obj_types[:n_vehicles]
            
        except Exception as e:
            print(f"[OfflineExtractor] Error extracting semantic features: {e}")
            import traceback
            traceback.print_exc()
        
        return semantic_features
    
    # =========================================================================
    # NEW: Multi-timestep rollout methods
    # =========================================================================
    
    def _extract_within_module_distributions(self, attn_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract within-module attention distributions for each module.
        
        Instead of summing all tokens (which trivially gives ~1.0 due to softmax),
        this computes how attention is *distributed* across entities within each module.
        
        Returns a dict with per-module distributional data:
        - 'per_vehicle_attention': shape (n_vehicles, n_heads) — agent module
        - 'roadgraph_concentration': shape (n_heads,) — entropy-based concentration
        - 'roadgraph_max_fraction': shape (n_heads,) — max attention fraction
        - 'per_light_attention': shape (n_lights, n_heads) — traffic light module  
        - 'per_waypoint_attention': shape (n_waypoints, n_heads) — GPS module
        - 'per_sdc_timestep_attention': shape (n_obs_timesteps, n_heads) — SDC module
        """
        result = {}
        
        for module_name in self.MODULE_KEYS:
            key = f"{module_name}/cross_attn_0"
            if key not in attn_weights:
                continue
            
            # Shape: (batch, n_latents, n_module_tokens, n_heads)
            attn = attn_weights[key]
            
            # Remove batch dimension
            if attn.ndim == 4:
                attn = attn[0]  # (n_latents, n_module_tokens, n_heads)
            
            n_latents, n_tokens, n_heads = attn.shape
            
            if module_name == 'other_traj':
                # Agent module: n_tokens = n_vehicles * n_obs_timesteps (e.g., 8 * 5 = 40)
                n_v = self._n_vehicles
                n_t = self._n_timesteps
                
                if n_tokens == n_v * n_t:
                    # Reshape: (n_latents, n_vehicles, n_obs_timesteps, n_heads)
                    attn_reshaped = attn.reshape(n_latents, n_v, n_t, n_heads)
                    # Sum over obs_timesteps, average over latents → (n_vehicles, n_heads)
                    per_vehicle = attn_reshaped.sum(axis=2).mean(axis=0)
                    # Normalize to fractions (each head sums to 1 across vehicles)
                    total = per_vehicle.sum(axis=0, keepdims=True)
                    total = np.where(total > 0, total, 1.0)
                    result['per_vehicle_attention'] = per_vehicle / total
                else:
                    # Fallback: can't reshape, just average over latents
                    attn_avg = attn.mean(axis=0)  # (n_tokens, n_heads)
                    total = attn_avg.sum(axis=0, keepdims=True)
                    total = np.where(total > 0, total, 1.0)
                    result['per_vehicle_attention'] = attn_avg / total
            
            elif module_name == 'roadgraph':
                # Roadgraph: 200 tokens — compute concentration metrics per head
                attn_avg = attn.mean(axis=0)  # (n_tokens, n_heads)
                # Normalize to probability distribution per head
                total = attn_avg.sum(axis=0, keepdims=True)
                total = np.where(total > 0, total, 1.0)
                probs = attn_avg / total  # (n_tokens, n_heads)
                
                # Entropy per head: H = -sum(p * log(p))
                safe_probs = np.maximum(probs, 1e-10)
                log_probs = np.log(safe_probs)
                entropy = -(probs * log_probs).sum(axis=0)  # (n_heads,)
                max_entropy = np.log(n_tokens)  # Maximum possible entropy
                
                # Concentration = 1 - normalized_entropy (high = focused, low = diffuse)
                result['roadgraph_concentration'] = 1.0 - entropy / max_entropy
                result['roadgraph_max_fraction'] = probs.max(axis=0)  # (n_heads,)
                result['roadgraph_entropy'] = entropy  # (n_heads,)
            
            elif module_name == 'traffic_lights':
                # Traffic lights: n_lights * n_obs_timesteps (e.g., 5 * 5 = 25)
                n_l = self._n_traffic_lights
                n_t = self._n_timesteps
                
                if n_tokens == n_l * n_t:
                    attn_reshaped = attn.reshape(n_latents, n_l, n_t, n_heads)
                    per_light = attn_reshaped.sum(axis=2).mean(axis=0)  # (n_lights, n_heads)
                    total = per_light.sum(axis=0, keepdims=True)
                    total = np.where(total > 0, total, 1.0)
                    result['per_light_attention'] = per_light / total
                else:
                    attn_avg = attn.mean(axis=0)
                    total = attn_avg.sum(axis=0, keepdims=True)
                    total = np.where(total > 0, total, 1.0)
                    result['per_light_attention'] = attn_avg / total
            
            elif module_name == 'gps_path':
                # GPS: n_waypoints tokens (e.g., 3 or 10)
                attn_avg = attn.mean(axis=0)  # (n_waypoints, n_heads)
                total = attn_avg.sum(axis=0, keepdims=True)
                total = np.where(total > 0, total, 1.0)
                result['per_waypoint_attention'] = attn_avg / total
            
            elif module_name == 'sdc_traj':
                # SDC: 1 * n_obs_timesteps tokens (e.g., 5)
                attn_avg = attn.mean(axis=0)  # (n_obs_timesteps, n_heads)
                total = attn_avg.sum(axis=0, keepdims=True)
                total = np.where(total > 0, total, 1.0)
                result['per_sdc_timestep_attention'] = attn_avg / total
        
        return result
    
    def _compute_collision_risk(self, semantic_features: Dict[str, np.ndarray]) -> float:
        """Compute scene-level collision risk R from per-agent TTC values.
        
        R = clip(1 - min(TTC_j for valid non-ego agents) / 3.0, 0, 1)
        
        R = 0 means safe (all agents' TTC >= 3s).
        R = 1 means imminent collision.
        
        Returns:
            Float in [0, 1].
        """
        ttc = semantic_features.get('ttc')
        if ttc is None:
            return 0.0
        
        # Filter out ego vehicle
        sdc_idx = semantic_features.get('sdc_index', 0)
        valid = semantic_features.get('valid')
        
        mask = np.ones(len(ttc), dtype=bool)
        if sdc_idx < len(mask):
            mask[sdc_idx] = False
        if valid is not None:
            mask &= np.array(valid).astype(bool)
        
        ttc_valid = ttc[mask]
        
        if len(ttc_valid) == 0:
            return 0.0
        
        min_ttc = float(np.min(ttc_valid))
        risk = np.clip(1.0 - min_ttc / 3.0, 0.0, 1.0)
        return float(risk)
    
    def _expert_step_single(self, env_transition: 'EnvTransition') -> 'EnvTransition':
        """Perform a single expert log-replay step.
        
        Extracts the logged expert action for the SDC and steps the environment.
        This does NOT require the learned policy — it replays what the human driver did.
        
        Args:
            env_transition: Current environment transition (with batch dim from vmap).
            
        Returns:
            Next environment transition.
        """
        # Get expert action from logged trajectory: (num_envs, num_agents, 2)
        actions = expert.infer_expert_action(
            env_transition.state, self.env.get_wrapper_attr("dynamics_model")
        ).data
        
        # Get SDC index
        sdc_idx = operations.get_index(env_transition.state.object_metadata.is_sdc)
        sdc_idx = jnp.expand_dims(sdc_idx, axis=(0, 1, 2))
        
        # Extract SDC action: (num_envs, 1, 2) -> (num_envs, 2)
        action_sdc = jnp.take_along_axis(actions, sdc_idx, axis=-2)
        action_sdc = jnp.squeeze(action_sdc, axis=-2)
        
        # Create valid action and step
        valid_mask = jnp.ones_like(action_sdc[..., 0:1], dtype=jnp.bool_)
        action = waymax_datatypes.Action(data=action_sdc, valid=valid_mask)
        action.validate()
        
        return self.env.step(env_transition, action)
    
    def run_episode(self, scenario_data, scenario_id: int = 0) -> Dict[str, Any]:
        """Roll out expert trajectory for T timesteps, extracting attention at each step.
        
        Uses expert log-replay (not the learned policy) for stepping. The encoder
        still processes observations using the trained weights, so attention patterns
        reflect the learned model's perception of the scene.
        
        Args:
            scenario_data: Batched scenario data from the data generator.
            scenario_id: ID for this scenario.
            
        Returns:
            Dictionary with per-timestep extraction data.
        """
        # Reset environment
        env_transition = self.env.reset(scenario_data)
        
        # JIT-compile the encoder forward pass
        @jax.jit
        def forward_with_attention(e_params, observation):
            latent, attention_weights = self.encoder.apply(
                {'params': e_params},
                observation
            )
            return latent, attention_weights
        
        timestep_records = []
        
        for t in range(self.MAX_TIMESTEPS):
            # 1. Extract attention from encoder using current observation
            obs = env_transition.observation
            _, attn_weights = forward_with_attention(self.encoder_params, obs)
            
            # Convert to numpy
            attn_weights_np = jax.tree_util.tree_map(
                lambda x: np.array(jax.device_get(x)),
                attn_weights
            )
            
            # 2. Extract within-module attention distributions
            attn_distributions = self._extract_within_module_distributions(attn_weights_np)
            
            # 3. Squeeze state for semantic feature extraction
            squeezed_state = jax.tree_util.tree_map(
                lambda x: x.squeeze(0) if hasattr(x, 'squeeze') and x.ndim > 0 else x,
                env_transition.state
            )
            
            # 4. Extract semantic features
            semantic = self.extract_semantic_features(squeezed_state)
            
            # 5. Compute collision risk R
            risk = self._compute_collision_risk(semantic)
            
            # 6. Store this timestep's data
            timestep_records.append({
                'timestep': t,
                'attention_distributions': attn_distributions,
                'semantic_features': semantic,
                'collision_risk': risk,
            })
            
            # 7. Check termination BEFORE stepping (done from previous step)
            if t > 0 and bool(jax.device_get(env_transition.done)):
                break
            
            # 8. Step using expert log-replay
            try:
                env_transition = self._expert_step_single(env_transition)
            except Exception as e:
                print(f"    [Episode] Step {t} failed: {e}")
                break
        
        return {
            'scenario_id': scenario_id,
            'n_timesteps': len(timestep_records),
            'timesteps': timestep_records,
            'token_boundaries': self.token_boundaries,
            'metadata': {
                'n_vehicles': self._n_vehicles,
                'n_timesteps_config': self._n_timesteps,
                'n_roadgraph': self._n_roadgraph,
                'n_traffic_lights': self._n_traffic_lights,
            },
        }
    
    # =========================================================================
    # Legacy single-snapshot methods (kept for backward compatibility)
    # =========================================================================
    
    def extract_scenario(self, scenario, scenario_id: int = 0) -> Dict[str, Any]:
        """Extract all data from a single scenario.
        
        Args:
            scenario: Simulator state.
            scenario_id: ID for this scenario.
            
        Returns:
            Dictionary with all extracted data.
        """
        # Extract attention weights
        attention_weights = self.extract_attention(scenario)
        
        # Aggregate per-vehicle attention
        attention_per_vehicle = self.aggregate_vehicle_attention(attention_weights)
        
        # Extract semantic features
        semantic_features = self.extract_semantic_features(scenario)
        
        return {
            'scenario_id': scenario_id,
            'attention_weights': attention_weights,
            'attention_per_vehicle': attention_per_vehicle,
            'semantic_features': semantic_features,
            'token_boundaries': self.token_boundaries,
            'metadata': {
                'n_vehicles': self._n_vehicles,
                'n_timesteps': self._n_timesteps,
                'n_roadgraph': self._n_roadgraph,
                'n_traffic_lights': self._n_traffic_lights,
            }
        }
    
    def extract_scenario_with_obs(self, state, obs, scenario_id: int = 0, debug: bool = True) -> Dict[str, Any]:
        """Extract all data using pre-computed observation.
        
        This method is used when the observation has already been computed (e.g., from env.reset).
        
        Args:
            state: Simulator state (squeezed, no batch dim).
            obs: Pre-computed observation array.
            scenario_id: ID for this scenario.
            debug: If True, print debug information.
            
        Returns:
            Dictionary with all extracted data.
        """
        # Check observation
        if debug:
            obs_np = np.array(jax.device_get(obs))
            print(f"\n[DEBUG] Pre-computed observation shape: {obs_np.shape}")
            print(f"[DEBUG] Obs min: {obs_np.min():.4f}, max: {obs_np.max():.4f}, mean: {obs_np.mean():.4f}")
            print(f"[DEBUG] Obs std: {obs_np.std():.4f}")
            
            # Check unflatten produces valid masks now
            unflatten_fn = self.env.get_wrapper_attr("features_extractor").unflatten_features
            features, masks = unflatten_fn(obs)
            mask_arr = np.array(masks[1])
            print(f"[DEBUG] other_traj_valid_mask True count: {mask_arr.sum()} / {mask_arr.size}")
        
        # Forward pass with attention extraction
        @jax.jit
        def forward_with_attention(e_params, observation):
            latent, attention_weights = self.encoder.apply(
                {'params': e_params},
                observation
            )
            return latent, attention_weights
        
        _, attn_weights = forward_with_attention(self.encoder_params, obs)
        
        # Convert to numpy
        attn_weights = jax.tree_util.tree_map(
            lambda x: np.array(jax.device_get(x)),
            attn_weights
        )
        
        if debug:
            print(f"[DEBUG] Attention weight keys: {list(attn_weights.keys())}")
        
        # Aggregate per-vehicle attention
        attention_per_vehicle = self.aggregate_vehicle_attention(attn_weights)
        
        # Extract semantic features from state
        semantic_features = self.extract_semantic_features(state)
        
        return {
            'scenario_id': scenario_id,
            'attention_weights': attn_weights,
            'attention_per_vehicle': attention_per_vehicle,
            'semantic_features': semantic_features,
            'token_boundaries': self.token_boundaries,
            'metadata': {
                'n_vehicles': self._n_vehicles,
                'n_timesteps': self._n_timesteps,
                'n_roadgraph': self._n_roadgraph,
                'n_traffic_lights': self._n_traffic_lights,
            }
        }
    
    def run(self, n_scenarios: int, output_path: str, rollout: bool = False) -> List[Dict[str, Any]]:
        """Run extraction on multiple scenarios.
        
        Args:
            n_scenarios: Number of scenarios to process.
            output_path: Path to save the results.
            rollout: If True, use multi-timestep rollout with expert log-replay.
                     If False, use legacy single-snapshot extraction.
            
        Returns:
            List of extraction results.
        """
        # Create data generator
        data_gen = make_data_generator(
            path=datasets.get_dataset(self.dataset_path),
            max_num_objects=self.config["max_num_objects"],
            include_sdc_paths=not self.config.get("waymo_dataset", False),
            batch_dims=(1,),
            seed=42,
            repeat=1,
        )
        
        results = []
        mode_str = "rollout" if rollout else "single-snapshot"
        print(f"[OfflineExtractor] Processing {n_scenarios} scenarios ({mode_str} mode)...")
        
        for i, scenario_batch in enumerate(data_gen):
            if i >= n_scenarios:
                break
            
            scenario = scenario_batch
            print(f"\n  Scenario {i+1}/{n_scenarios}", end="")
            
            try:
                if rollout:
                    # Multi-timestep rollout with expert log-replay
                    result = self.run_episode(scenario, scenario_id=i)
                    n_ts = result['n_timesteps']
                    risk_vals = [ts['collision_risk'] for ts in result['timesteps']]
                    max_risk = max(risk_vals) if risk_vals else 0.0
                    print(f" ✓ ({n_ts} timesteps, max_risk={max_risk:.2f})")
                else:
                    # Legacy single-snapshot extraction
                    env_transition = self.env.reset(scenario)
                    obs = env_transition.observation
                    reset_state = jax.tree_util.tree_map(
                        lambda x: x.squeeze(0) if hasattr(x, 'squeeze') and x.ndim > 0 else x,
                        env_transition.state
                    )
                    result = self.extract_scenario_with_obs(reset_state, obs, scenario_id=i)
                    print(" ✓")
                
                results.append(result)
            except Exception as e:
                print(f" ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        checkpoint_step = self._parse_checkpoint_step(self.checkpoint_name)
        
        output_data = {
            'checkpoint': self.checkpoint_name,
            'step': checkpoint_step,
            'run_dir': self.run_dir,
            'n_scenarios': len(results),
            'extraction_mode': 'rollout' if rollout else 'single_snapshot',
            'scenarios': results,
            'config': {
                'encoder': self.config.get('encoder', {}),
                'observation_config': self.config.get('observation_config', {}),
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n[OfflineExtractor] Saved {len(results)} scenarios to {output_path}")
        print(f"[OfflineExtractor] Total file size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return results
    
    def _parse_checkpoint_step(self, checkpoint_name: str) -> int:
        """Extract step number from checkpoint filename.
        
        Returns:
            Step number from filename, or -1 for 'model_final.pkl' (indicates final checkpoint)
        """
        import re
        
        # Special case for final model
        if 'final' in checkpoint_name.lower():
            return -1  # -1 indicates final checkpoint (step unknown)
        
        match = re.search(r'(\d+)', checkpoint_name)
        return int(match.group(1)) if match else 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract attention weights and semantic features for HSI analysis."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the training run directory (contains .hydra/config.yaml and model/)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (tfrecord file or dataset name)"
    )
    parser.add_argument(
        "--n_scenarios",
        type=int,
        default=100,
        help="Number of scenarios to process (default: 100)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./extractions",
        help="Directory to save extraction results (default: ./extractions)"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=["model_final.pkl"],
        help="Checkpoint file(s) to process (default: model_final.pkl)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scenario sampling (default: 42)"
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Use multi-timestep rollout with expert log-replay (default: single-snapshot)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each checkpoint
    extractor = None
    for checkpoint in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: {checkpoint}")
        print(f"{'='*60}")
        
        if extractor is None:
            # First checkpoint: setup everything
            extractor = OfflineExtractor(
                run_dir=args.run_dir,
                dataset_path=args.dataset,
                checkpoint_name=checkpoint
            )
            extractor.setup()
        else:
            # Subsequent checkpoints: just load new params
            extractor.load_checkpoint(checkpoint)
        
        # Generate output filename
        checkpoint_base = os.path.splitext(checkpoint)[0]
        output_path = os.path.join(args.output_dir, f"extraction_{checkpoint_base}.pkl")
        
        # Run extraction
        extractor.run(n_scenarios=args.n_scenarios, output_path=output_path, rollout=args.rollout)
    
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
