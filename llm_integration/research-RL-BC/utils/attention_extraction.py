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

from vmax.simulator import make_env_for_evaluation, datasets, make_data_generator
from vmax.agents.learning.reinforcement.ppo import ppo_factory
from vmax.agents.networks import network_utils
from vmax.scripts.evaluate.utils import load_params, load_yaml_config

# Import encoders (support both Wayformer and LQ)
from vmax.agents.networks.encoders.wayformer import WayformerEncoder
from vmax.agents.networks.encoders.lq import LQEncoder

# Import concentration metrics
import sys
analysis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if analysis_path not in sys.path:
    sys.path.insert(0, analysis_path)
from analysis.concentration_metrics import compute_per_head_concentration, compute_concentration_suite


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
    
    # DRAC (Deceleration Rate to Avoid a Crash) parameters
    # DRAC = closing_speed² / (2 * distance)
    # DRAC_MAX = 8.0 m/s² corresponds to emergency braking limit
    # Reference thresholds: 3.4 comfortable, 6.0 hard, 8.0 emergency
    DRAC_MAX = 8.0  # m/s²
    
    # Criticality threshold for identifying critical scenarios
    # Scenarios with max criticality >= this threshold are flagged
    CRITICAL_THRESHOLD = 0.5  # Moderate criticality regime
    
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
        
        # Detect encoder type from config and instantiate accordingly
        encoder_type = self.config.get("encoder", {}).get("type", self.config.get("encoder", {}).get("name", "wayformer")).lower()
        
        # Remap "perceiver" to "lq" for consistency
        if encoder_type == "perceiver":
            encoder_type = "lq"
            
        print(f"[OfflineExtractor] Detected encoder type: {encoder_type}")
        
        if "lq" in encoder_type:
            print(f"[OfflineExtractor] Using LQEncoder (Perceiver architecture)")
            self.encoder = LQEncoder(
                unflatten_fn=unflatten_fn,
                **encoder_cfg
            )
        else:
            print(f"[OfflineExtractor] Using WayformerEncoder")
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
            
            # Check for different encoder structures
            if 'encoder_layer' in self.params.policy['params']:
                enc_params = self.params.policy['params']['encoder_layer']
                print(f"[DEBUG] encoder_layer keys: {list(enc_params.keys())[:10]}...")
            elif 'lq_encoder' in self.params.policy['params']:
                enc_params = self.params.policy['params']['lq_encoder']
                print(f"[DEBUG] lq_encoder keys: {list(enc_params.keys())[:10]}...")
            else:
                print(f"[DEBUG] No standard encoder key found, checking all keys...")
        
        # Extract encoder parameters based on architecture
        # Detect encoder type from config
        encoder_type = self.config.get("encoder", {}).get("type", self.config.get("encoder", {}).get("name", "wayformer")).lower()
        if encoder_type == "perceiver":
            encoder_type = "lq"
        
        if 'params' in self.params.policy:
            policy_params = self.params.policy['params']
            
            # Try different possible encoder parameter locations
            if encoder_type == "lq":
                # LQ encoder needs the entire encoder_layer (contains embeddings + perceiver_attention)
                if 'encoder_layer' in policy_params:
                    self.encoder_params = policy_params['encoder_layer']
                    # Verify it has the expected LQ structure
                    if 'perceiver_attention' in self.encoder_params or 'lq_attention' in self.encoder_params:
                        print(f"[DEBUG] Found LQ encoder params under 'encoder_layer' (contains perceiver_attention)")
                    else:
                        print(f"[DEBUG] Using 'encoder_layer' for LQ encoder (keys: {list(self.encoder_params.keys())[:10]}...)")
                    
                    # Remap perceiver_attention -> lq_attention for compatibility
                    if 'perceiver_attention' in self.encoder_params:
                        self.encoder_params['lq_attention'] = self.encoder_params.pop('perceiver_attention')
                        print(f"[DEBUG] Remapped 'perceiver_attention' -> 'lq_attention'")
                        
                elif 'lq_encoder' in policy_params:
                    self.encoder_params = policy_params['lq_encoder']
                    print(f"[DEBUG] Found LQ encoder params under 'lq_encoder'")
                elif 'encoder' in policy_params:
                    self.encoder_params = policy_params['encoder']
                    print(f"[DEBUG] Found encoder params under 'encoder'")
                else:
                    raise ValueError(f"Could not find LQ encoder params. Available keys: {list(policy_params.keys())}")
            else:
                # Wayformer uses 'encoder_layer'
                if 'encoder_layer' in policy_params:
                    self.encoder_params = policy_params['encoder_layer']
                    print(f"[DEBUG] Found Wayformer encoder params under 'encoder_layer'")
                else:
                    raise ValueError(f"Could not find 'encoder_layer' in policy params. Available keys: {list(policy_params.keys())}")
        else:
            raise ValueError("Could not find 'params' in policy")
        
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
        
        Handles both Wayformer and LQ architectures:
        - Wayformer: Has separate attention per entity type (other_traj/cross_attn_0)
        - LQ: Has unified cross-attention to all tokens (cross_attn_0)
        
        Args:
            attention_weights: Raw attention weights from encoder.
            debug: If True, print debug information.
            
        Returns:
            Array of shape (n_heads, n_vehicles) with aggregated attention.
        """
        # Detect architecture from attention weight keys
        is_wayformer = any("other_traj" in key for key in attention_weights.keys())
        is_lq = any(key.startswith("cross_attn_") for key in attention_weights.keys())
        
        if debug:
            print(f"\n[DEBUG] Detected architecture: {'Wayformer' if is_wayformer else 'LQ' if is_lq else 'Unknown'}")
            print(f"[DEBUG] Available attention keys: {list(attention_weights.keys())}")
        
        if is_wayformer:
            return self._aggregate_wayformer_attention(attention_weights, debug)
        elif is_lq:
            return self._aggregate_lq_attention(attention_weights, debug)
        else:
            print(f"[OfflineExtractor] Error: Could not detect encoder architecture")
            return None
    
    def _aggregate_wayformer_attention(self, attention_weights: Dict[str, np.ndarray], debug: bool) -> np.ndarray:
        """Aggregate attention for Wayformer architecture."""
        # Get the cross-attention to other vehicles
        key = "other_traj/cross_attn_0"
        if key not in attention_weights:
            print(f"[OfflineExtractor] Warning: '{key}' not found in attention weights")
            return None
        
        # Shape: (batch, n_latents, n_tokens, n_heads)
        attn = attention_weights[key]
        
        if debug:
            print(f"\n[DEBUG] Wayformer attention shape: {attn.shape}")
            print(f"[DEBUG] Min: {attn.min():.6f}, Max: {attn.max():.6f}, Mean: {attn.mean():.6f}")
        
        # Remove batch dimension if present
        if attn.ndim == 4:
            attn = attn[0]  # (n_latents, n_tokens, n_heads)
        
        n_latents, n_tokens, n_heads = attn.shape
        n_vehicles = self._n_vehicles
        n_timesteps = self._n_timesteps
        
        # Verify token count matches
        expected_tokens = n_vehicles * n_timesteps
        if n_tokens != expected_tokens:
            print(f"[OfflineExtractor] Warning: Token count mismatch. Got {n_tokens}, expected {expected_tokens}")
            return None
        
        # Reshape: (n_latents, n_tokens, n_heads) -> (n_latents, n_vehicles, n_timesteps, n_heads)
        attn_reshaped = attn.reshape(n_latents, n_vehicles, n_timesteps, n_heads)
        
        # Sum over timesteps: -> (n_latents, n_vehicles, n_heads)
        attn_per_vehicle = attn_reshaped.sum(axis=2)
        
        # Sum over latents: -> (n_vehicles, n_heads)
        attn_final = attn_per_vehicle.sum(axis=0)
        
        # Transpose to (n_heads, n_vehicles)
        attn_final = attn_final.T
        
        return attn_final
    
    def _aggregate_lq_attention(self, attention_weights: Dict[str, np.ndarray], debug: bool) -> np.ndarray:
        """Aggregate attention for LQ (Perceiver) architecture.
        
        LQ has cross-attention from latents to ALL input tokens (concatenated).
        We need to extract only the tokens corresponding to other vehicles.
        """
        # Use first cross-attention layer
        key = "cross_attn_0"
        if key not in attention_weights:
            print(f"[OfflineExtractor] Warning: '{key}' not found in attention weights")
            return None
        
        # Shape: (batch, n_latents, n_all_tokens, n_heads)
        attn = attention_weights[key]
        
        if debug:
            print(f"\n[DEBUG] LQ cross-attention shape: {attn.shape}")
            print(f"[DEBUG] Min: {attn.min():.6f}, Max: {attn.max():.6f}, Mean: {attn.mean():.6f}")
        
        # Remove batch dimension if present
        if attn.ndim == 4:
            attn = attn[0]  # (n_latents, n_all_tokens, n_heads)
        
        n_latents, n_all_tokens, n_heads = attn.shape
        
        # Extract token boundaries for other vehicles
        # Token order in LQ: [sdc_traj, other_traj, roadgraph, traffic_lights, gps_path]
        other_start, other_end = self.token_boundaries['other_traj']
        
        if debug:
            print(f"[DEBUG] Total tokens: {n_all_tokens}")
            print(f"[DEBUG] Other vehicle tokens: [{other_start}:{other_end}]")
        
        # Extract attention to other vehicle tokens only
        # Shape: (n_latents, n_vehicle_tokens, n_heads)
        attn_vehicles = attn[:, other_start:other_end, :]
        
        n_vehicles = self._n_vehicles
        n_timesteps = self._n_timesteps
        n_vehicle_tokens = n_vehicles * n_timesteps
        
        if attn_vehicles.shape[1] != n_vehicle_tokens:
            print(f"[OfflineExtractor] Warning: Vehicle token mismatch. Got {attn_vehicles.shape[1]}, expected {n_vehicle_tokens}")
            return None
        
        # Reshape: (n_latents, n_vehicle_tokens, n_heads) -> (n_latents, n_vehicles, n_timesteps, n_heads)
        attn_reshaped = attn_vehicles.reshape(n_latents, n_vehicles, n_timesteps, n_heads)
        
        if debug:
            print(f"[DEBUG] Reshaped to: {attn_reshaped.shape}")
        
        # Sum over timesteps: -> (n_latents, n_vehicles, n_heads)
        attn_per_vehicle = attn_reshaped.sum(axis=2)
        
        # Sum over latents: -> (n_vehicles, n_heads)
        attn_final = attn_per_vehicle.sum(axis=0)
        
        if debug:
            print(f"[DEBUG] Final per-vehicle attention shape: {attn_final.shape}")
            print(f"[DEBUG] Per-vehicle attention (first 3 vehicles, all heads):\n{attn_final[:3, :]}")
        
        # Transpose to (n_heads, n_vehicles)
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
            
            # 8. Criticality scores
            criticality = self.compute_criticality(semantic_features)
            semantic_features['criticality'] = criticality
            
        except Exception as e:
            print(f"[OfflineExtractor] Error extracting semantic features: {e}")
            import traceback
            traceback.print_exc()
        
        return semantic_features
    
    def compute_criticality(self, semantic_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute criticality scores for each vehicle using DRAC.
        
        DRAC (Deceleration Rate to Avoid a Crash) measures the minimum
        deceleration required to avoid a collision:
        
            DRAC = closing_speed² / (2 × distance)
            Criticality = min(1, DRAC / DRAC_max)
        
        This avoids the mathematical redundancy of combining TTC, closing
        speed, and distance (since TTC ≈ distance / closing_speed).
        
        Args:
            semantic_features: Dictionary with 'closing_speed', 'distance_to_ego', 'valid'.
            
        Returns:
            Array of criticality scores in [0, 1] for each vehicle.
        """
        closing_speed = semantic_features['closing_speed']
        distance = semantic_features['distance_to_ego']
        valid = semantic_features['valid']
        
        # DRAC only meaningful when vehicle is approaching (closing_speed > 0)
        # Add epsilon to distance to prevent division by zero
        approaching = closing_speed > 0.5
        safe_distance = distance + 1e-6  # Small epsilon to prevent div-by-zero
        
        drac = np.where(
            approaching & (distance > 1e-3),
            closing_speed**2 / (2.0 * safe_distance),
            0.0
        )
        
        # Normalize to [0, 1] using emergency braking threshold
        criticality = np.minimum(1.0, drac / self.DRAC_MAX)
        
        # Mask invalid vehicles
        criticality = criticality * valid
        
        return np.clip(criticality, 0.0, 1.0)
    
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
        attention_per_vehicle = self.aggregate_vehicle_attention(attn_weights, debug=False)
        
        # Extract semantic features from state (includes criticality)
        semantic_features = self.extract_semantic_features(state)
        
        # Compute concentration scores per head (all metrics)
        concentration_scores = None
        concentration_suite = None
        if attention_per_vehicle is not None:
            # Legacy: Gini-only scores for backward compat
            concentration_scores = compute_per_head_concentration(
                attention_per_vehicle,
                metric='gini'
            )
            # Full suite: Gini, entropy, top-3 mass per head
            n_heads = attention_per_vehicle.shape[0]
            concentration_suite = {}
            for metric_name in ['gini', 'entropy', 'top3_mass']:
                concentration_suite[metric_name] = compute_per_head_concentration(
                    attention_per_vehicle,
                    metric=metric_name
                )
            if debug:
                print(f"[DEBUG] Concentration scores per head:")
                for metric_name, scores in concentration_suite.items():
                    print(f"  {metric_name}: {scores}")
        
        return {
            'scenario_id': scenario_id,
            'attention_weights': attn_weights,
            'attention_per_vehicle': attention_per_vehicle,
            'concentration_scores': concentration_scores,
            'concentration_suite': concentration_suite,
            'semantic_features': semantic_features,
            'token_boundaries': self.token_boundaries,
            'metadata': {
                'n_vehicles': self._n_vehicles,
                'n_timesteps': self._n_timesteps,
                'n_roadgraph': self._n_roadgraph,
                'n_traffic_lights': self._n_traffic_lights,
                'criticality_params': {
                    'method': 'DRAC',
                    'drac_max': self.DRAC_MAX,
                },
            }
        }
    
    def run(self, n_scenarios: int, output_path: str) -> List[Dict[str, Any]]:
        """Run extraction on multiple scenarios.
        
        Args:
            n_scenarios: Number of scenarios to process.
            output_path: Path to save the results.
            
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
        critical_scenarios = []  # Track critical scenario indices
        
        print(f"[OfflineExtractor] Processing {n_scenarios} scenarios...")
        
        for i, scenario_batch in enumerate(data_gen):
            if i >= n_scenarios:
                break
            
            # Keep the batch dimension for env.reset (it uses vmap)
            scenario = scenario_batch
            
            print(f"\n  Scenario {i+1}/{n_scenarios}", end="")
            
            try:
                # Reset env to properly initialize the state
                env_transition = self.env.reset(scenario)
                
                # Use observation from EnvTransition (already computed during reset)
                # This avoids the shape issues with calling env.observe again
                obs = env_transition.observation
                
                # Squeeze state for semantic feature extraction
                reset_state = jax.tree_util.tree_map(
                    lambda x: x.squeeze(0) if hasattr(x, 'squeeze') and x.ndim > 0 else x,
                    env_transition.state
                )
                
                # Use pre-computed observation and squeezed state for extraction
                result = self.extract_scenario_with_obs(reset_state, obs, scenario_id=i)
                results.append(result)
                
                # Check if scenario is critical
                if 'semantic_features' in result and 'criticality' in result['semantic_features']:
                    criticality = result['semantic_features']['criticality']
                    valid = result['semantic_features'].get('valid', np.ones_like(criticality))
                    max_criticality = float(np.max(criticality[valid > 0.5])) if np.any(valid > 0.5) else 0.0
                    
                    if max_criticality >= self.CRITICAL_THRESHOLD:
                        critical_scenarios.append({
                            'scenario_id': i,
                            'max_criticality': max_criticality
                        })
                        print(f" ✓ [CRITICAL: {max_criticality:.3f}]")
                    else:
                        print(" ✓")
                else:
                    print(" ✓")
            except Exception as e:
                print(f" ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Extract checkpoint step from filename
        checkpoint_step = self._parse_checkpoint_step(self.checkpoint_name)
        
        output_data = {
            'checkpoint': self.checkpoint_name,
            'step': checkpoint_step,
            'run_dir': self.run_dir,
            'n_scenarios': len(results),
            'scenarios': results,
            'critical_scenarios': critical_scenarios,
            'critical_threshold': self.CRITICAL_THRESHOLD,
            'config': {
                'encoder': self.config.get('encoder', {}),
                'observation_config': self.config.get('observation_config', {}),
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n[OfflineExtractor] Saved {len(results)} scenarios to {output_path}")
        print(f"[OfflineExtractor] Critical scenarios: {len(critical_scenarios)}/{len(results)} (threshold: {self.CRITICAL_THRESHOLD})")
        if critical_scenarios:
            print(f"[OfflineExtractor] Critical scenario IDs: {[s['scenario_id'] for s in critical_scenarios[:10]]}{'...' if len(critical_scenarios) > 10 else ''}")
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
        extractor.run(n_scenarios=args.n_scenarios, output_path=output_path)
    
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
