# Reward-Conditioned Attention Analysis in Autonomous Driving

## Implementation Plan for Publishable Research

---

# 1. Research Overview

## 1.1 Paper Title (Working)

**"Reward-Conditioned Attention: How Multi-Objective RL Agents Learn to Focus on What Matters"**

Alternative titles:
- "Do Driving Agents Look Where They Should? Correlating Attention with Reward Objectives"
- "From Rewards to Attention: Understanding Learned Representations in Autonomous Driving"

## 1.2 Core Research Questions

| ID | Question | Analysis Type |
|----|----------|---------------|
| **RQ1** | Does attention correlate with safety-critical features (TTC, distance) during dangerous moments? | All models |
| **RQ2** | Does attention shift from road to vehicles as danger increases? | Temporal analysis |
| **RQ3** | Do different encoder architectures show different attention-reward correlations? | Cross-architecture |
| **RQ4** | Does reward shaping during training change learned attention patterns? | Config comparison |
| **RQ5** | Can we predict agent actions (brake, steer) from attention patterns? | Action prediction |

## 1.3 Key Contributions

1. **Methodological:** Framework for correlating attention with reward components in multi-objective RL
2. **Empirical:** Large-scale analysis across 5 architectures, 3 reward configs, 36 models
3. **Insights:** Evidence that attention encodes implicit reward weighting
4. **Practical:** Guidelines for validating reward design through attention analysis

## 1.4 Available Resources

### Models (from V-Max authors)

| Architecture | Reward Configs | Seeds | Total |
|--------------|----------------|-------|-------|
| Perceiver/LQ | minimal, basic, complete | 42, 69, 99 | 9+ |
| MTR | minimal | 42, 69, 99 | 3 |
| Wayformer | minimal | 42, 69, 99 | 3 |
| MGAIL | minimal | 42, 69, 99 | 3 |
| None (MLP) | minimal | 42, 69, 99 | 3 |

### Data

- Event catalog: 152 critical events from 5 scenarios (expandable)
- WOMD validation: 44,000 scenarios
- Full trajectory data: ~80 timesteps per scenario

---

# 2. Theoretical Framework

## 2.1 Reward Decomposition in V-Max

V-Max uses hierarchical reward:

```
r_total = r_safety + r_navigation + r_behavior

where:
  r_safety = r_collision + r_offroad + r_red_light
  r_navigation = r_progress + r_off_route  
  r_behavior = r_comfort + r_speed (complete config only)
```

### Reward Components (from V-Max paper)

| Component | Formula | Weight | Config |
|-----------|---------|--------|--------|
| `collision` | -1 if overlap | 1.0 | all |
| `offroad` | -1 if off drivable area | 1.0 | all |
| `red_light` | -1 if violation | 1.0 | all |
| `off_route` | -1 if off route | 0.6 | minimal+ |
| `progression` | +1 if progress | 0.2 | minimal+ |
| `comfort` | penalty for jerk | 0.2 | complete |
| `overspeed` | -1 if over limit | 0.1 | complete |

## 2.2 Continuous Risk Metrics

Instead of binary rewards, we compute continuous "risk" that a reward will be triggered:

```python
# Safety risk: How close to collision?
safety_risk = clip(1 - min_TTC / τ_safety, 0, 1)  # τ_safety = 3.0s

# Navigation risk: How far off route?
navigation_risk = clip(route_deviation / τ_nav, 0, 1)  # τ_nav = 5.0m

# Behavior risk: How harsh is the driving?
behavior_risk = clip(|acceleration| / τ_accel, 0, 1)  # τ_accel = 5.0 m/s²
```

## 2.3 Attention-Reward Hypothesis

**Core hypothesis:** If an agent has learned to optimize a reward component, its attention should correlate with features relevant to that reward.

| Reward Component | Relevant Features | Expected Attention |
|------------------|-------------------|-------------------|
| Safety (collision) | Nearby vehicles, TTC | High attn to trajectory tokens |
| Safety (offroad) | Road boundaries | High attn to roadgraph tokens |
| Navigation | Route, path | High attn to path tokens |
| Behavior | Current dynamics | Diffuse / road-focused |

---

# 3. Project Structure

```
reward_attention/
├── __init__.py
├── config.py                    # Configuration dataclasses
│
├── data/
│   ├── __init__.py
│   ├── scenario_loader.py       # Load V-Max scenarios
│   ├── event_loader.py          # Load event catalog
│   └── trajectory_sampler.py    # Sample timesteps for analysis
│
├── rewards/
│   ├── __init__.py
│   ├── components.py            # Individual reward computations
│   ├── risk_metrics.py          # Continuous risk computations
│   └── aggregator.py            # Combine into analysis-ready format
│
├── attention/
│   ├── __init__.py
│   ├── extractor.py             # Extract attention from models
│   ├── aggregator.py            # Aggregate by token category
│   └── per_agent.py             # Per-agent attention tracking
│
├── analysis/
│   ├── __init__.py
│   ├── correlation.py           # Attention-reward correlations
│   ├── temporal.py              # Temporal evolution analysis
│   ├── action_conditioned.py    # Analysis by action type
│   ├── cross_architecture.py    # Compare architectures
│   └── cross_config.py          # Compare reward configs
│
├── visualization/
│   ├── __init__.py
│   ├── scatter.py               # Scatter plots
│   ├── heatmaps.py              # Correlation heatmaps
│   ├── temporal_plots.py        # Time series plots
│   └── paper_figures.py         # Publication-ready figures
│
├── experiments/
│   ├── __init__.py
│   ├── run_full_analysis.py     # Main experiment script
│   ├── run_event_analysis.py    # Event-based analysis
│   └── run_comparison.py        # Cross-model comparison
│
└── utils/
    ├── __init__.py
    ├── stats.py                 # Statistical tests
    └── io.py                    # Save/load results
```

---

# 4. Data Structures

## 4.1 Core Dataclasses

```python
# reward_attention/config.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class RewardComponent(Enum):
    """All reward components."""
    # Safety
    COLLISION = "collision"
    OFFROAD = "offroad"
    RED_LIGHT = "red_light"
    
    # Navigation
    PROGRESS = "progress"
    OFF_ROUTE = "off_route"
    
    # Behavior
    COMFORT = "comfort"
    OVERSPEED = "overspeed"


class AttentionCategory(Enum):
    """Token categories for attention aggregation."""
    TRAJECTORY = "trajectory"      # Other vehicles
    ROADGRAPH = "roadgraph"        # Road geometry
    TRAFFIC_LIGHT = "traffic_light"
    PATH_TARGET = "path_target"    # Navigation goal
    SDC = "sdc"                    # Ego history


@dataclass
class TimestepData:
    """All data for a single timestep analysis."""
    # Identification
    scenario_id: str
    timestep: int
    
    # State
    ego_position: Tuple[float, float]
    ego_velocity: Tuple[float, float]
    ego_speed: float
    ego_heading: float
    
    # Action
    acceleration: float
    steering: float
    
    # Other agents
    num_valid_agents: int
    agent_distances: List[float]       # Distance to each agent
    agent_ttcs: List[float]            # TTC to each agent
    nearest_agent_id: int
    
    # Risk metrics (continuous 0-1)
    safety_risk: float
    navigation_risk: float
    behavior_risk: float
    
    # Component-level risk
    collision_risk: float              # Based on min TTC
    offroad_risk: float                # Based on distance to road edge
    off_route_risk: float              # Based on route deviation
    
    # Binary rewards (what would be received)
    reward_collision: float            # -1 or 0
    reward_offroad: float
    reward_red_light: float
    reward_progress: float
    reward_off_route: float
    
    # Attention (aggregated by category)
    attn_trajectory: float             # Sum of attention to vehicle tokens
    attn_roadgraph: float
    attn_traffic_light: float
    attn_path_target: float
    attn_sdc: float
    
    # Attention (per agent)
    attn_per_agent: List[float]        # Attention to each of 8 agents
    attn_to_nearest: float             # Attention to nearest agent
    attn_to_threat: Optional[float]    # Attention to lowest-TTC agent
    
    # Attention (per query) - for query specialization analysis
    query_attn_trajectory: List[float]  # (16,) attention to trajectory per query
    query_attn_roadgraph: List[float]
    query_attn_path: List[float]
    
    # Metadata
    is_critical_event: bool
    event_type: Optional[str]
    event_severity: Optional[float]


@dataclass
class AnalysisConfig:
    """Configuration for analysis run."""
    # Models to analyze
    model_paths: List[str]
    
    # Data selection
    num_scenarios: int = 100
    use_event_catalog: bool = True
    event_catalog_path: Optional[str] = None
    sample_full_trajectories: bool = True
    
    # Risk thresholds
    ttc_threshold: float = 3.0         # seconds
    distance_threshold: float = 5.0    # meters
    route_deviation_threshold: float = 5.0  # meters
    accel_threshold: float = 5.0       # m/s²
    
    # Analysis settings
    compute_per_agent_attention: bool = True
    compute_per_query_attention: bool = True
    
    # Output
    output_dir: str = "results/reward_attention"
    save_raw_data: bool = True
```

## 4.2 Results Structures

```python
# reward_attention/analysis/correlation.py

@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""
    variable_x: str
    variable_y: str
    
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    
    n_samples: int
    
    # For subgroup analysis
    subgroup: Optional[str] = None  # e.g., "critical_events", "braking_moments"
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.pearson_p < alpha


@dataclass 
class AnalysisSummary:
    """Summary of full analysis run."""
    model_name: str
    architecture: str
    reward_config: str
    
    n_timesteps_analyzed: int
    n_critical_events: int
    
    # Main correlations
    correlations: Dict[str, CorrelationResult]
    
    # Summary statistics
    mean_safety_risk: float
    mean_attn_trajectory: float
    mean_attn_roadgraph: float
    
    # Key findings
    safety_attention_corr: float       # corr(safety_risk, attn_trajectory)
    navigation_attention_corr: float   # corr(nav_risk, attn_path)
```

---

# 5. Core Implementations

## 5.1 Risk Metric Computation

```python
# reward_attention/rewards/risk_metrics.py

import numpy as np
from typing import Dict, Tuple
import jax.numpy as jnp


class RiskComputer:
    """
    Compute continuous risk metrics from scenario state.
    
    Risk metrics are continuous [0, 1] values indicating how close
    the agent is to triggering each reward component.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.ttc_threshold = config.ttc_threshold
        self.distance_threshold = config.distance_threshold
        self.route_threshold = config.route_deviation_threshold
        self.accel_threshold = config.accel_threshold
    
    def compute_all_risks(
        self,
        ego_state: Dict,
        other_agents: Dict,
        road_info: Dict,
        action: Tuple[float, float],
    ) -> Dict[str, float]:
        """
        Compute all risk metrics for a timestep.
        
        Returns:
            Dict with keys: safety_risk, navigation_risk, behavior_risk,
                           collision_risk, offroad_risk, off_route_risk
        """
        # Collision risk (based on TTC)
        ttcs = self._compute_ttcs(ego_state, other_agents)
        min_ttc = np.min(ttcs) if len(ttcs) > 0 else 10.0
        collision_risk = np.clip(1 - min_ttc / self.ttc_threshold, 0, 1)
        
        # Offroad risk (based on distance to road edge)
        dist_to_edge = road_info.get('distance_to_edge', 10.0)
        offroad_risk = np.clip(1 - dist_to_edge / 2.0, 0, 1)  # Risk increases within 2m of edge
        
        # Off-route risk (based on route deviation)
        route_deviation = road_info.get('route_deviation', 0.0)
        off_route_risk = np.clip(route_deviation / self.route_threshold, 0, 1)
        
        # Behavior risk (based on acceleration magnitude)
        accel = action[0]
        behavior_risk = np.clip(np.abs(accel) / self.accel_threshold, 0, 1)
        
        # Aggregate risks
        safety_risk = max(collision_risk, offroad_risk)
        navigation_risk = off_route_risk
        
        return {
            'safety_risk': float(safety_risk),
            'collision_risk': float(collision_risk),
            'offroad_risk': float(offroad_risk),
            'navigation_risk': float(navigation_risk),
            'off_route_risk': float(off_route_risk),
            'behavior_risk': float(behavior_risk),
            'min_ttc': float(min_ttc),
            'route_deviation': float(route_deviation),
        }
    
    def _compute_ttcs(
        self, 
        ego_state: Dict, 
        other_agents: Dict
    ) -> np.ndarray:
        """Compute TTC to each other agent."""
        ttcs = []
        
        ego_x, ego_y = ego_state['x'], ego_state['y']
        ego_vx, ego_vy = ego_state['vx'], ego_state['vy']
        
        for i in range(other_agents['num_valid']):
            if not other_agents['valid'][i]:
                continue
                
            other_x = other_agents['x'][i]
            other_y = other_agents['y'][i]
            other_vx = other_agents['vx'][i]
            other_vy = other_agents['vy'][i]
            
            # Relative position and velocity
            rel_x = other_x - ego_x
            rel_y = other_y - ego_y
            rel_vx = ego_vx - other_vx
            rel_vy = ego_vy - other_vy
            
            distance = np.sqrt(rel_x**2 + rel_y**2)
            
            # Closing speed
            if distance > 0.01:
                closing_speed = (rel_x * rel_vx + rel_y * rel_vy) / distance
            else:
                closing_speed = 0
            
            # TTC
            if closing_speed > 0.1:
                ttc = distance / closing_speed
            else:
                ttc = 10.0  # Not approaching
            
            ttcs.append(min(ttc, 10.0))
        
        return np.array(ttcs) if ttcs else np.array([10.0])


def compute_binary_rewards(
    state: Dict,
    next_state: Dict,
    action: Tuple[float, float],
) -> Dict[str, float]:
    """
    Compute actual binary reward values.
    
    These are the rewards the agent would receive at this timestep.
    """
    rewards = {}
    
    # Safety rewards
    rewards['collision'] = -1.0 if next_state.get('collision', False) else 0.0
    rewards['offroad'] = -1.0 if next_state.get('offroad', False) else 0.0
    rewards['red_light'] = -1.0 if next_state.get('red_light_violation', False) else 0.0
    
    # Navigation rewards
    progress = next_state.get('progress', 0) - state.get('progress', 0)
    rewards['progress'] = 0.2 if progress > 0 else 0.0
    rewards['off_route'] = -0.6 if next_state.get('off_route', False) else 0.0
    
    # Behavior rewards (for complete config analysis)
    accel = action[0]
    jerk = next_state.get('jerk', 0)
    rewards['comfort'] = -0.2 * min(1.0, abs(jerk) / 5.0)  # Continuous comfort penalty
    rewards['overspeed'] = -0.1 if next_state.get('over_speed_limit', False) else 0.0
    
    return rewards
```

## 5.2 Attention Extraction and Aggregation

```python
# reward_attention/attention/extractor.py

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
import numpy as np


class AttentionExtractor:
    """
    Extract and aggregate attention weights from LQ/Perceiver models.
    """
    
    # Token ranges in V-Max observation (adjust based on actual config)
    TOKEN_RANGES = {
        'sdc': (0, 40),              # 8 features × 5 timesteps
        'trajectory': (40, 360),      # 8 agents × 8 features × 5 timesteps
        'roadgraph': (360, 1360),     # 200 points × 5 features
        'traffic_light': (1360, 1635),# 5 lights × 11 features × 5 timesteps
        'path_target': (1635, 1655),  # 10 points × 2 features
    }
    
    # Per-agent token ranges within trajectory
    AGENT_TOKEN_SIZE = 40  # 8 features × 5 timesteps per agent
    NUM_AGENTS = 8
    
    def __init__(self, model, observation_config: Optional[Dict] = None):
        """
        Args:
            model: Loaded V-Max model with attention extraction capability
            observation_config: Override token ranges if different from default
        """
        self.model = model
        
        if observation_config:
            self._update_token_ranges(observation_config)
    
    def _update_token_ranges(self, config: Dict):
        """Update token ranges based on observation config."""
        # Calculate ranges from config
        num_objects = config.get('num_objects', 8)
        num_steps = config.get('steps', 5)
        max_roadgraph = config.get('max_roadgraph', 200)
        max_tl = config.get('max_traffic_lights', 5)
        path_points = config.get('path_points', 10)
        
        sdc_size = num_steps * 8  # SDC trajectory
        traj_size = num_objects * num_steps * 8  # Other agents
        road_size = max_roadgraph * 5
        tl_size = max_tl * num_steps * 11
        path_size = path_points * 2
        
        idx = 0
        self.TOKEN_RANGES = {}
        
        self.TOKEN_RANGES['sdc'] = (idx, idx + sdc_size)
        idx += sdc_size
        
        self.TOKEN_RANGES['trajectory'] = (idx, idx + traj_size)
        idx += traj_size
        
        self.TOKEN_RANGES['roadgraph'] = (idx, idx + road_size)
        idx += road_size
        
        self.TOKEN_RANGES['traffic_light'] = (idx, idx + tl_size)
        idx += tl_size
        
        self.TOKEN_RANGES['path_target'] = (idx, idx + path_size)
        
        self.AGENT_TOKEN_SIZE = num_steps * 8
        self.NUM_AGENTS = num_objects
    
    def extract_attention(
        self, 
        observation: jnp.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract raw attention weights from model.
        
        Returns:
            Dict with:
                'cross_attention': (num_queries, num_tokens) - main attention
                'self_attention': (num_queries, num_queries) - if available
        """
        # This depends on model implementation
        # Assuming model has get_attention method
        attention = self.model.get_attention(observation)
        
        return {
            'cross_attention': np.array(attention['cross_attention']),
            'self_attention': np.array(attention.get('self_attention', None)),
        }
    
    def aggregate_by_category(
        self, 
        attention: np.ndarray
    ) -> Dict[str, float]:
        """
        Aggregate attention weights by token category.
        
        Args:
            attention: (num_queries, num_tokens) attention weights
            
        Returns:
            Dict mapping category name to total attention
        """
        # Average across queries first
        attn_avg = attention.mean(axis=0)  # (num_tokens,)
        
        aggregated = {}
        for category, (start, end) in self.TOKEN_RANGES.items():
            if end <= len(attn_avg):
                aggregated[category] = float(attn_avg[start:end].sum())
            else:
                aggregated[category] = 0.0
        
        # Normalize to sum to 1
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}
        
        return aggregated
    
    def get_per_agent_attention(
        self, 
        attention: np.ndarray
    ) -> np.ndarray:
        """
        Get attention to each individual agent.
        
        Args:
            attention: (num_queries, num_tokens) attention weights
            
        Returns:
            (num_agents,) attention per agent
        """
        attn_avg = attention.mean(axis=0)
        
        traj_start, traj_end = self.TOKEN_RANGES['trajectory']
        traj_attention = attn_avg[traj_start:traj_end]
        
        per_agent = []
        for i in range(self.NUM_AGENTS):
            agent_start = i * self.AGENT_TOKEN_SIZE
            agent_end = (i + 1) * self.AGENT_TOKEN_SIZE
            if agent_end <= len(traj_attention):
                per_agent.append(float(traj_attention[agent_start:agent_end].sum()))
            else:
                per_agent.append(0.0)
        
        return np.array(per_agent)
    
    def get_per_query_category_attention(
        self, 
        attention: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get attention per query per category.
        
        Useful for query specialization analysis.
        
        Args:
            attention: (num_queries, num_tokens)
            
        Returns:
            Dict mapping category to (num_queries,) attention array
        """
        num_queries = attention.shape[0]
        
        per_query = {}
        for category, (start, end) in self.TOKEN_RANGES.items():
            if end <= attention.shape[1]:
                per_query[category] = attention[:, start:end].sum(axis=1)
            else:
                per_query[category] = np.zeros(num_queries)
        
        return per_query
    
    def extract_full_timestep_data(
        self,
        observation: jnp.ndarray,
        scenario_id: str,
        timestep: int,
        ego_state: Dict,
        other_agents: Dict,
        road_info: Dict,
        action: Tuple[float, float],
        risk_computer: 'RiskComputer',
        event_info: Optional[Dict] = None,
    ) -> TimestepData:
        """
        Extract all data needed for analysis at a single timestep.
        
        This is the main extraction function that produces TimestepData.
        """
        # Extract attention
        raw_attention = self.extract_attention(observation)
        cross_attn = raw_attention['cross_attention']
        
        # Aggregate attention
        category_attn = self.aggregate_by_category(cross_attn)
        per_agent_attn = self.get_per_agent_attention(cross_attn)
        per_query_attn = self.get_per_query_category_attention(cross_attn)
        
        # Compute risks
        risks = risk_computer.compute_all_risks(
            ego_state, other_agents, road_info, action
        )
        
        # Find threat agent (lowest TTC)
        ttcs = risk_computer._compute_ttcs(ego_state, other_agents)
        threat_agent_id = int(np.argmin(ttcs)) if len(ttcs) > 0 else 0
        
        # Compute distances
        distances = []
        for i in range(other_agents['num_valid']):
            if other_agents['valid'][i]:
                dx = other_agents['x'][i] - ego_state['x']
                dy = other_agents['y'][i] - ego_state['y']
                distances.append(float(np.sqrt(dx**2 + dy**2)))
            else:
                distances.append(100.0)
        
        nearest_agent_id = int(np.argmin(distances)) if distances else 0
        
        return TimestepData(
            scenario_id=scenario_id,
            timestep=timestep,
            
            ego_position=(ego_state['x'], ego_state['y']),
            ego_velocity=(ego_state['vx'], ego_state['vy']),
            ego_speed=ego_state.get('speed', np.sqrt(ego_state['vx']**2 + ego_state['vy']**2)),
            ego_heading=ego_state.get('heading', 0.0),
            
            acceleration=action[0],
            steering=action[1],
            
            num_valid_agents=other_agents['num_valid'],
            agent_distances=distances,
            agent_ttcs=list(ttcs),
            nearest_agent_id=nearest_agent_id,
            
            safety_risk=risks['safety_risk'],
            navigation_risk=risks['navigation_risk'],
            behavior_risk=risks['behavior_risk'],
            collision_risk=risks['collision_risk'],
            offroad_risk=risks['offroad_risk'],
            off_route_risk=risks['off_route_risk'],
            
            reward_collision=0.0,  # Computed separately if needed
            reward_offroad=0.0,
            reward_red_light=0.0,
            reward_progress=0.0,
            reward_off_route=0.0,
            
            attn_trajectory=category_attn.get('trajectory', 0.0),
            attn_roadgraph=category_attn.get('roadgraph', 0.0),
            attn_traffic_light=category_attn.get('traffic_light', 0.0),
            attn_path_target=category_attn.get('path_target', 0.0),
            attn_sdc=category_attn.get('sdc', 0.0),
            
            attn_per_agent=list(per_agent_attn),
            attn_to_nearest=per_agent_attn[nearest_agent_id] if nearest_agent_id < len(per_agent_attn) else 0.0,
            attn_to_threat=per_agent_attn[threat_agent_id] if threat_agent_id < len(per_agent_attn) else 0.0,
            
            query_attn_trajectory=list(per_query_attn.get('trajectory', [])),
            query_attn_roadgraph=list(per_query_attn.get('roadgraph', [])),
            query_attn_path=list(per_query_attn.get('path_target', [])),
            
            is_critical_event=event_info is not None,
            event_type=event_info.get('event_type') if event_info else None,
            event_severity=event_info.get('severity_score') if event_info else None,
        )
```

## 5.3 Correlation Analysis

```python
# reward_attention/analysis/correlation.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CorrelationResult:
    """Single correlation result."""
    variable_x: str
    variable_y: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n_samples: int
    subgroup: Optional[str] = None
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.pearson_p < alpha and self.spearman_p < alpha


class CorrelationAnalyzer:
    """
    Compute and analyze correlations between attention and reward risks.
    """
    
    # Define expected correlation pairs
    HYPOTHESIZED_CORRELATIONS = [
        # (risk_metric, attention_category, expected_direction)
        ('safety_risk', 'attn_trajectory', 'positive'),
        ('collision_risk', 'attn_trajectory', 'positive'),
        ('collision_risk', 'attn_to_threat', 'positive'),
        ('navigation_risk', 'attn_path_target', 'positive'),
        ('off_route_risk', 'attn_path_target', 'positive'),
        ('behavior_risk', 'attn_roadgraph', 'unclear'),
    ]
    
    def __init__(self, data: List[TimestepData]):
        """
        Args:
            data: List of TimestepData from extraction
        """
        self.data = data
        self.df = self._to_dataframe()
    
    def _to_dataframe(self) -> pd.DataFrame:
        """Convert TimestepData list to DataFrame."""
        records = []
        for d in self.data:
            records.append({
                'scenario_id': d.scenario_id,
                'timestep': d.timestep,
                
                # Risks
                'safety_risk': d.safety_risk,
                'collision_risk': d.collision_risk,
                'offroad_risk': d.offroad_risk,
                'navigation_risk': d.navigation_risk,
                'off_route_risk': d.off_route_risk,
                'behavior_risk': d.behavior_risk,
                
                # Attention
                'attn_trajectory': d.attn_trajectory,
                'attn_roadgraph': d.attn_roadgraph,
                'attn_traffic_light': d.attn_traffic_light,
                'attn_path_target': d.attn_path_target,
                'attn_sdc': d.attn_sdc,
                'attn_to_threat': d.attn_to_threat,
                'attn_to_nearest': d.attn_to_nearest,
                
                # Action
                'acceleration': d.acceleration,
                'steering': d.steering,
                
                # State
                'ego_speed': d.ego_speed,
                'num_valid_agents': d.num_valid_agents,
                
                # Event info
                'is_critical': d.is_critical_event,
                'event_type': d.event_type,
                
                # Derived
                'is_braking': d.acceleration < -1.0,
                'is_accelerating': d.acceleration > 1.0,
                'is_steering': abs(d.steering) > 0.1,
            })
        
        return pd.DataFrame(records)
    
    def compute_correlation(
        self,
        var_x: str,
        var_y: str,
        subgroup: Optional[str] = None,
    ) -> CorrelationResult:
        """
        Compute correlation between two variables.
        
        Args:
            var_x: Column name for X variable
            var_y: Column name for Y variable
            subgroup: Optional filter ('critical', 'braking', 'all')
        """
        df = self.df.copy()
        
        # Apply subgroup filter
        if subgroup == 'critical':
            df = df[df['is_critical'] == True]
        elif subgroup == 'braking':
            df = df[df['is_braking'] == True]
        elif subgroup == 'high_risk':
            df = df[df['safety_risk'] > 0.5]
        
        # Drop NaN
        df = df[[var_x, var_y]].dropna()
        
        if len(df) < 10:
            return CorrelationResult(
                variable_x=var_x,
                variable_y=var_y,
                pearson_r=0.0,
                pearson_p=1.0,
                spearman_rho=0.0,
                spearman_p=1.0,
                n_samples=len(df),
                subgroup=subgroup,
            )
        
        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(df[var_x], df[var_y])
        spearman_rho, spearman_p = stats.spearmanr(df[var_x], df[var_y])
        
        return CorrelationResult(
            variable_x=var_x,
            variable_y=var_y,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_rho=spearman_rho,
            spearman_p=spearman_p,
            n_samples=len(df),
            subgroup=subgroup,
        )
    
    def compute_all_hypothesized(
        self,
        subgroups: List[str] = ['all', 'critical', 'high_risk']
    ) -> Dict[str, List[CorrelationResult]]:
        """
        Compute all hypothesized correlations.
        
        Returns:
            Dict mapping subgroup to list of CorrelationResult
        """
        results = {}
        
        for subgroup in subgroups:
            sg_results = []
            for risk, attn, expected in self.HYPOTHESIZED_CORRELATIONS:
                result = self.compute_correlation(risk, attn, subgroup)
                sg_results.append(result)
            results[subgroup] = sg_results
        
        return results
    
    def compute_full_correlation_matrix(
        self,
        subgroup: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute full correlation matrix between all risks and attentions.
        """
        df = self.df.copy()
        
        if subgroup == 'critical':
            df = df[df['is_critical'] == True]
        
        risk_cols = ['safety_risk', 'collision_risk', 'navigation_risk', 'behavior_risk']
        attn_cols = ['attn_trajectory', 'attn_roadgraph', 'attn_traffic_light', 
                     'attn_path_target', 'attn_to_threat']
        
        corr_matrix = df[risk_cols + attn_cols].corr()
        
        # Extract risk-attention submatrix
        return corr_matrix.loc[risk_cols, attn_cols]
    
    def compute_action_conditioned_attention(self) -> pd.DataFrame:
        """
        Compute mean attention by action type.
        
        Answers: "When braking, what does the model attend to?"
        """
        df = self.df.copy()
        
        # Define action categories
        df['action_type'] = 'neutral'
        df.loc[df['is_braking'], 'action_type'] = 'braking'
        df.loc[df['is_accelerating'], 'action_type'] = 'accelerating'
        df.loc[df['is_steering'], 'action_type'] = 'steering'
        
        attn_cols = ['attn_trajectory', 'attn_roadgraph', 'attn_traffic_light', 
                     'attn_path_target', 'attn_to_threat']
        
        return df.groupby('action_type')[attn_cols].mean()
    
    def summary(self) -> Dict:
        """Generate summary statistics."""
        hypothesized = self.compute_all_hypothesized()
        action_attention = self.compute_action_conditioned_attention()
        
        # Find strongest correlations
        all_corrs = hypothesized.get('all', [])
        significant_corrs = [c for c in all_corrs if c.is_significant()]
        
        return {
            'n_timesteps': len(self.df),
            'n_critical': self.df['is_critical'].sum(),
            'n_significant_correlations': len(significant_corrs),
            'strongest_correlation': max(all_corrs, key=lambda c: abs(c.pearson_r)) if all_corrs else None,
            'action_conditioned_attention': action_attention.to_dict(),
            'mean_risks': {
                'safety': self.df['safety_risk'].mean(),
                'navigation': self.df['navigation_risk'].mean(),
                'behavior': self.df['behavior_risk'].mean(),
            },
            'mean_attention': {
                'trajectory': self.df['attn_trajectory'].mean(),
                'roadgraph': self.df['attn_roadgraph'].mean(),
                'path_target': self.df['attn_path_target'].mean(),
            },
        }
```

## 5.4 Temporal Analysis

```python
# reward_attention/analysis/temporal.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TemporalPattern:
    """Pattern of attention change over event window."""
    event_id: str
    scenario_id: str
    
    # Timesteps
    onset: int
    peak: int
    offset: int
    
    # Attention at key moments
    attn_trajectory_onset: float
    attn_trajectory_peak: float
    attn_trajectory_offset: float
    
    attn_path_onset: float
    attn_path_peak: float
    attn_path_offset: float
    
    # Change metrics
    trajectory_attention_increase: float  # peak - onset
    path_attention_decrease: float        # onset - peak (expected to decrease)
    
    # Risk at key moments
    safety_risk_onset: float
    safety_risk_peak: float


class TemporalAnalyzer:
    """
    Analyze how attention evolves during critical events.
    """
    
    def __init__(self, timestep_data: List[TimestepData], events: List[Dict]):
        """
        Args:
            timestep_data: Full timestep data for all analyzed timesteps
            events: Event catalog entries
        """
        self.timestep_data = timestep_data
        self.events = events
        
        # Index timestep data by (scenario, timestep)
        self.data_index = {
            (d.scenario_id, d.timestep): d 
            for d in timestep_data
        }
    
    def analyze_event(self, event: Dict) -> TemporalPattern:
        """
        Analyze attention evolution for a single event.
        """
        scenario_id = event['scenario_id']
        onset = event['onset']
        peak = event['peak']
        offset = event['offset']
        
        # Get data at key moments
        data_onset = self.data_index.get((scenario_id, onset))
        data_peak = self.data_index.get((scenario_id, peak))
        data_offset = self.data_index.get((scenario_id, offset))
        
        if not all([data_onset, data_peak, data_offset]):
            return None
        
        return TemporalPattern(
            event_id=f"{scenario_id}_{peak}",
            scenario_id=scenario_id,
            onset=onset,
            peak=peak,
            offset=offset,
            
            attn_trajectory_onset=data_onset.attn_trajectory,
            attn_trajectory_peak=data_peak.attn_trajectory,
            attn_trajectory_offset=data_offset.attn_trajectory,
            
            attn_path_onset=data_onset.attn_path_target,
            attn_path_peak=data_peak.attn_path_target,
            attn_path_offset=data_offset.attn_path_target,
            
            trajectory_attention_increase=data_peak.attn_trajectory - data_onset.attn_trajectory,
            path_attention_decrease=data_onset.attn_path_target - data_peak.attn_path_target,
            
            safety_risk_onset=data_onset.safety_risk,
            safety_risk_peak=data_peak.safety_risk,
        )
    
    def analyze_all_events(self) -> List[TemporalPattern]:
        """Analyze all events."""
        patterns = []
        for event in self.events:
            pattern = self.analyze_event(event)
            if pattern:
                patterns.append(pattern)
        return patterns
    
    def compute_attention_trajectory(
        self,
        scenario_id: str,
        window: Tuple[int, int],
    ) -> pd.DataFrame:
        """
        Get attention values across a time window.
        
        Returns DataFrame with timestep as index and attention columns.
        """
        start, end = window
        records = []
        
        for t in range(start, end + 1):
            data = self.data_index.get((scenario_id, t))
            if data:
                records.append({
                    'timestep': t,
                    'attn_trajectory': data.attn_trajectory,
                    'attn_roadgraph': data.attn_roadgraph,
                    'attn_path_target': data.attn_path_target,
                    'attn_to_threat': data.attn_to_threat,
                    'safety_risk': data.safety_risk,
                    'collision_risk': data.collision_risk,
                })
        
        return pd.DataFrame(records).set_index('timestep')
    
    def summary(self) -> Dict:
        """Summarize temporal patterns."""
        patterns = self.analyze_all_events()
        
        if not patterns:
            return {'n_events': 0}
        
        # Compute statistics
        traj_increases = [p.trajectory_attention_increase for p in patterns]
        path_decreases = [p.path_attention_decrease for p in patterns]
        
        return {
            'n_events_analyzed': len(patterns),
            'mean_trajectory_attention_increase': np.mean(traj_increases),
            'std_trajectory_attention_increase': np.std(traj_increases),
            'pct_events_with_traj_increase': np.mean([x > 0 for x in traj_increases]),
            'mean_path_attention_decrease': np.mean(path_decreases),
            'pct_events_with_path_decrease': np.mean([x > 0 for x in path_decreases]),
        }
```

## 5.5 Cross-Architecture Comparison

```python
# reward_attention/analysis/cross_architecture.py

import numpy as np
import pandas as pd
from typing import List, Dict
from scipy import stats


class CrossArchitectureAnalyzer:
    """
    Compare attention-reward correlations across architectures.
    """
    
    def __init__(self, results_by_model: Dict[str, 'AnalysisSummary']):
        """
        Args:
            results_by_model: Dict mapping model name to AnalysisSummary
        """
        self.results = results_by_model
        self._parse_model_names()
    
    def _parse_model_names(self):
        """Extract architecture and config from model names."""
        self.model_info = {}
        for name in self.results.keys():
            parts = name.split('_')
            
            # Parse: womd_sac_road_perceiver_minimal_42
            if len(parts) >= 6:
                self.model_info[name] = {
                    'dataset': parts[0],
                    'encoder': parts[3],
                    'reward_config': parts[4],
                    'seed': parts[5],
                }
            else:
                # Handle special cases like sac_seed0
                self.model_info[name] = {
                    'dataset': 'womd',
                    'encoder': 'lq',
                    'reward_config': 'extended',
                    'seed': name.split('seed')[-1] if 'seed' in name else '0',
                }
    
    def compare_architectures(
        self,
        metric: str = 'safety_attention_corr'
    ) -> pd.DataFrame:
        """
        Compare a metric across architectures.
        
        Args:
            metric: Which correlation/metric to compare
            
        Returns:
            DataFrame with architecture comparison
        """
        records = []
        for name, summary in self.results.items():
            info = self.model_info[name]
            records.append({
                'model': name,
                'encoder': info['encoder'],
                'reward_config': info['reward_config'],
                'seed': info['seed'],
                metric: getattr(summary, metric, None),
            })
        
        df = pd.DataFrame(records)
        
        # Aggregate by encoder
        by_encoder = df.groupby('encoder')[metric].agg(['mean', 'std', 'count'])
        
        return by_encoder
    
    def compare_reward_configs(
        self,
        encoder: str = 'perceiver'
    ) -> pd.DataFrame:
        """
        Compare metrics across reward configs for a single encoder.
        """
        records = []
        for name, summary in self.results.items():
            info = self.model_info[name]
            if info['encoder'] == encoder:
                records.append({
                    'model': name,
                    'reward_config': info['reward_config'],
                    'seed': info['seed'],
                    'safety_attn_corr': summary.safety_attention_corr,
                    'nav_attn_corr': summary.navigation_attention_corr,
                    'mean_attn_trajectory': summary.mean_attn_trajectory,
                    'mean_attn_path': summary.mean_attn_path,
                })
        
        df = pd.DataFrame(records)
        return df.groupby('reward_config').agg(['mean', 'std'])
    
    def statistical_comparison(
        self,
        group1_models: List[str],
        group2_models: List[str],
        metric: str,
    ) -> Dict:
        """
        Statistical test comparing two groups of models.
        
        Returns t-test and effect size.
        """
        values1 = [getattr(self.results[m], metric) for m in group1_models if m in self.results]
        values2 = [getattr(self.results[m], metric) for m in group2_models if m in self.results]
        
        if len(values1) < 2 or len(values2) < 2:
            return {'error': 'Not enough samples'}
        
        t_stat, p_value = stats.ttest_ind(values1, values2)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
        cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'group1_mean': np.mean(values1),
            'group2_mean': np.mean(values2),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
        }
```

---

# 6. Visualization

## 6.1 Publication-Ready Figures

```python
# reward_attention/visualization/paper_figures.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def set_paper_style():
    """Set matplotlib style for publication."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_risk_attention_scatter(
    df: pd.DataFrame,
    risk_col: str,
    attn_col: str,
    title: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of risk vs attention with regression line.
    
    Figure 1 in paper: Main result showing correlation.
    """
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Scatter
    ax.scatter(df[risk_col], df[attn_col], alpha=0.3, s=10, c='steelblue')
    
    # Regression line
    z = np.polyfit(df[risk_col], df[attn_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[risk_col].min(), df[risk_col].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear fit')
    
    # Correlation annotation
    from scipy.stats import pearsonr
    r, p_val = pearsonr(df[risk_col], df[attn_col])
    ax.annotate(f'r = {r:.3f}\np < {p_val:.0e}' if p_val < 0.001 else f'r = {r:.3f}\np = {p_val:.3f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(risk_col.replace('_', ' ').title())
    ax.set_ylabel(attn_col.replace('_', ' ').title())
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Risk-Attention Correlation Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of correlations between risks and attention.
    
    Figure 2 in paper: Overview of all correlations.
    """
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.5,
        vmax=0.5,
        ax=ax,
        cbar_kws={'label': 'Pearson r'}
    )
    
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_temporal_evolution(
    df: pd.DataFrame,  # With timestep as index
    event_peak: int,
    title: str = "Attention Evolution During Hazard Event",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Time series of attention and risk during an event.
    
    Figure 3 in paper: Temporal dynamics.
    """
    set_paper_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    
    # Top: Attention
    ax1.plot(df.index, df['attn_trajectory'], 'b-', label='Vehicles', linewidth=2)
    ax1.plot(df.index, df['attn_roadgraph'], 'g-', label='Road', linewidth=2)
    ax1.plot(df.index, df['attn_path_target'], 'orange', label='Path', linewidth=2)
    ax1.axvline(event_peak, color='red', linestyle='--', alpha=0.7, label='Event peak')
    ax1.set_ylabel('Attention')
    ax1.legend(loc='upper right')
    ax1.set_title(title)
    
    # Bottom: Risk
    ax2.plot(df.index, df['safety_risk'], 'r-', label='Safety Risk', linewidth=2)
    ax2.fill_between(df.index, 0, df['safety_risk'], alpha=0.3, color='red')
    ax2.axvline(event_peak, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Safety Risk')
    ax2.set_xlabel('Timestep')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_action_conditioned_attention(
    df: pd.DataFrame,  # From compute_action_conditioned_attention
    title: str = "Attention Distribution by Action Type",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart showing attention patterns for different actions.
    
    Figure 4 in paper: What does the model attend to when braking?
    """
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    df_plot = df.reset_index()
    df_melted = df_plot.melt(
        id_vars='action_type',
        var_name='Attention Category',
        value_name='Attention'
    )
    
    sns.barplot(
        data=df_melted,
        x='action_type',
        y='Attention',
        hue='Attention Category',
        ax=ax
    )
    
    ax.set_xlabel('Action Type')
    ax.set_ylabel('Mean Attention')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_architecture_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'safety_attention_corr',
    title: str = "Safety-Attention Correlation by Architecture",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing architectures.
    
    Figure 5 in paper: Cross-architecture comparison.
    """
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    encoders = comparison_df.index
    means = comparison_df['mean']
    stds = comparison_df['std']
    
    bars = ax.bar(encoders, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Encoder Architecture')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax.set_title(title)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add significance markers (placeholder)
    # Would need actual statistical test results
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_config_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Effect of Reward Configuration on Attention",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing reward configs.
    
    Figure 6 in paper: Does reward shaping change attention?
    """
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    configs = comparison_df.index.get_level_values(0).unique()
    metrics = ['safety_attn_corr', 'nav_attn_corr']
    
    x = np.arange(len(configs))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        means = [comparison_df.loc[c, (metric, 'mean')] for c in configs]
        stds = [comparison_df.loc[c, (metric, 'std')] for c in configs]
        ax.bar(x + i*width, means, width, yerr=stds, label=metric.replace('_', ' ').title(), capsize=3)
    
    ax.set_xlabel('Reward Configuration')
    ax.set_ylabel('Correlation')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.set_title(title)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig
```

---

# 7. Main Experiment Script

```python
# reward_attention/experiments/run_full_analysis.py

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd

from ..config import AnalysisConfig, TimestepData
from ..data.scenario_loader import ScenarioLoader
from ..data.event_loader import load_event_catalog
from ..rewards.risk_metrics import RiskComputer
from ..attention.extractor import AttentionExtractor
from ..analysis.correlation import CorrelationAnalyzer
from ..analysis.temporal import TemporalAnalyzer
from ..visualization import paper_figures


def run_single_model_analysis(
    model_path: str,
    config: AnalysisConfig,
) -> Dict:
    """
    Run full analysis for a single model.
    
    This is the main analysis pipeline.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_path}")
    print(f"{'='*60}")
    
    # 1. Load model
    print("Loading model...")
    from posthoc_xai.models.loader import load_model
    model = load_model(model_path)
    
    # 2. Setup components
    risk_computer = RiskComputer(config)
    attention_extractor = AttentionExtractor(model)
    
    # 3. Load events if using event-based analysis
    events = []
    if config.use_event_catalog and config.event_catalog_path:
        events = load_event_catalog(config.event_catalog_path)
        print(f"Loaded {len(events)} events")
    
    # 4. Extract data for all timesteps
    print("Extracting timestep data...")
    timestep_data: List[TimestepData] = []
    
    # Load scenarios
    scenario_loader = ScenarioLoader(config)
    
    for scenario in tqdm(scenario_loader.iter_scenarios(config.num_scenarios)):
        scenario_events = [e for e in events if e['scenario_id'] == scenario.id]
        event_timesteps = {e['peak'] for e in scenario_events}
        
        for t in range(scenario.num_timesteps):
            # Get state at timestep
            obs = scenario.get_observation(t)
            ego_state = scenario.get_ego_state(t)
            other_agents = scenario.get_other_agents(t)
            road_info = scenario.get_road_info(t)
            action = scenario.get_action(t)
            
            # Check if this is an event timestep
            event_info = None
            if t in event_timesteps:
                event_info = next(e for e in scenario_events if e['peak'] == t)
            
            # Extract full timestep data
            data = attention_extractor.extract_full_timestep_data(
                observation=obs,
                scenario_id=scenario.id,
                timestep=t,
                ego_state=ego_state,
                other_agents=other_agents,
                road_info=road_info,
                action=action,
                risk_computer=risk_computer,
                event_info=event_info,
            )
            
            timestep_data.append(data)
    
    print(f"Extracted {len(timestep_data)} timesteps")
    
    # 5. Run correlation analysis
    print("Running correlation analysis...")
    correlation_analyzer = CorrelationAnalyzer(timestep_data)
    
    correlations = correlation_analyzer.compute_all_hypothesized()
    correlation_matrix = correlation_analyzer.compute_full_correlation_matrix()
    action_attention = correlation_analyzer.compute_action_conditioned_attention()
    correlation_summary = correlation_analyzer.summary()
    
    # 6. Run temporal analysis (if events available)
    temporal_summary = {}
    if events:
        print("Running temporal analysis...")
        temporal_analyzer = TemporalAnalyzer(timestep_data, events)
        temporal_summary = temporal_analyzer.summary()
    
    # 7. Generate visualizations
    print("Generating figures...")
    output_dir = Path(config.output_dir) / Path(model_path).name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Main scatter plot
    df = correlation_analyzer.df
    paper_figures.plot_risk_attention_scatter(
        df, 'safety_risk', 'attn_trajectory',
        title='Safety Risk vs Vehicle Attention',
        save_path=output_dir / 'fig1_scatter_safety.pdf'
    )
    
    # Figure 2: Correlation heatmap
    paper_figures.plot_correlation_heatmap(
        correlation_matrix,
        save_path=output_dir / 'fig2_correlation_heatmap.pdf'
    )
    
    # Figure 3: Temporal evolution (example event)
    if events:
        example_event = events[0]
        temporal_df = temporal_analyzer.compute_attention_trajectory(
            example_event['scenario_id'],
            example_event['window']
        )
        paper_figures.plot_temporal_evolution(
            temporal_df,
            event_peak=example_event['peak'],
            save_path=output_dir / 'fig3_temporal.pdf'
        )
    
    # Figure 4: Action-conditioned attention
    paper_figures.plot_action_conditioned_attention(
        action_attention,
        save_path=output_dir / 'fig4_action_attention.pdf'
    )
    
    # 8. Save results
    print("Saving results...")
    results = {
        'model_path': model_path,
        'model_name': Path(model_path).name,
        'config': config.__dict__,
        'n_timesteps': len(timestep_data),
        'correlation_summary': correlation_summary,
        'temporal_summary': temporal_summary,
        'correlations_by_subgroup': {
            subgroup: [c.__dict__ for c in corrs]
            for subgroup, corrs in correlations.items()
        },
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    if config.save_raw_data:
        with open(output_dir / 'timestep_data.pkl', 'wb') as f:
            pickle.dump(timestep_data, f)
    
    print(f"Results saved to {output_dir}")
    
    return results


def run_comparison(
    model_paths: List[str],
    config: AnalysisConfig,
):
    """
    Run analysis on multiple models and compare.
    """
    all_results = {}
    
    for model_path in model_paths:
        try:
            results = run_single_model_analysis(model_path, config)
            all_results[results['model_name']] = results
        except Exception as e:
            print(f"Error analyzing {model_path}: {e}")
            continue
    
    # Cross-model comparison
    print("\n" + "="*60)
    print("Cross-Model Comparison")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    for name, results in all_results.items():
        summary = results['correlation_summary']
        comparison_data.append({
            'model': name,
            'n_timesteps': results['n_timesteps'],
            'safety_attn_corr': summary.get('strongest_correlation', {}).get('pearson_r', 0),
            'mean_attn_trajectory': summary.get('mean_attention', {}).get('trajectory', 0),
            'mean_safety_risk': summary.get('mean_risks', {}).get('safety', 0),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string())
    
    # Save comparison
    output_dir = Path(config.output_dir)
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    return all_results


# CLI entry point
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--events', type=str, default=None)
    parser.add_argument('--scenarios', type=int, default=100)
    parser.add_argument('--output', type=str, default='results/reward_attention')
    
    args = parser.parse_args()
    
    config = AnalysisConfig(
        model_paths=args.models,
        num_scenarios=args.scenarios,
        use_event_catalog=args.events is not None,
        event_catalog_path=args.events,
        output_dir=args.output,
    )
    
    run_comparison(args.models, config)
```

---

# 8. Expected Results and Paper Outline

## 8.1 Hypothesized Findings

| Finding | Evidence | Significance |
|---------|----------|--------------|
| **F1:** Safety risk correlates with vehicle attention | r > 0.3, p < 0.001 | Models attend to threats |
| **F2:** Correlation stronger during critical events | r_critical > r_all | Situational awareness |
| **F3:** Attention shifts from road to vehicles as danger increases | Temporal analysis | Dynamic adaptation |
| **F4:** Braking moments show highest vehicle attention | Action-conditioned analysis | Action-attention link |
| **F5:** Architectures differ in correlation strength | Cross-architecture comparison | Design implications |
| **F6:** Reward config affects attention patterns (if complete available) | Config comparison | Reward shaping insight |

## 8.2 Paper Outline

```
Abstract (150 words)
- Problem: Black-box RL for AD
- Approach: Correlate attention with reward objectives
- Findings: Attention encodes reward-relevant features
- Implication: Attention as implicit reward weighting

1. Introduction (1 page)
- Motivation: Trust in AD systems
- Gap: Understanding what RL agents learn
- Contribution: Reward-conditioned attention analysis

2. Related Work (0.5 page)
- XAI for RL
- Attention analysis in driving
- Multi-objective RL

3. Background (0.5 page)
- V-Max framework
- SAC algorithm
- Attention mechanisms in Perceiver/LQ

4. Methodology (1.5 pages)
- Risk metrics (safety, navigation, behavior)
- Attention extraction and aggregation
- Correlation analysis
- Temporal analysis

5. Experiments (2 pages)
- Setup: 36 models, 5 architectures, 3 configs
- RQ1: Risk-attention correlation
- RQ2: Temporal evolution
- RQ3: Action-conditioned attention
- RQ4: Architecture comparison
- RQ5: Reward config comparison

6. Results (1.5 pages)
- Main finding: Strong safety-attention correlation
- Temporal: Attention shifts during danger
- Cross-architecture: Perceiver shows strongest correlation
- Cross-config: Complete reward increases path attention

7. Discussion (0.5 page)
- Implications for reward design
- Limitations
- Future work

8. Conclusion (0.25 page)

References
```

## 8.3 Target Venues

| Venue | Type | Deadline | Fit |
|-------|------|----------|-----|
| **ICRA** | Conference | Sep | Robotics + AD |
| **IV (Intelligent Vehicles)** | Conference | Feb | AD-specific |
| **CoRL** | Conference | Jun | Robot learning |
| **IROS** | Conference | Mar | Robotics |
| **NeurIPS Workshop** | Workshop | Sep | XAI focus |

---

# 9. Implementation Timeline

| Phase | Tasks | Days |
|-------|-------|------|
| **Phase 1** | Data structures, risk computation | 2 |
| **Phase 2** | Attention extraction, aggregation | 2 |
| **Phase 3** | Correlation analysis | 2 |
| **Phase 4** | Temporal analysis | 1 |
| **Phase 5** | Visualization | 2 |
| **Phase 6** | Run experiments (all models) | 2-3 |
| **Phase 7** | Analysis, figures | 2 |
| **Phase 8** | Paper writing | 5 |

**Total: ~18-20 days**

---

# 10. Checklist for Claude Code

- [ ] Create project structure
- [ ] Implement `TimestepData` and `AnalysisConfig` dataclasses
- [ ] Implement `RiskComputer` with TTC and risk metrics
- [ ] Implement `AttentionExtractor` with category aggregation
- [ ] Implement `CorrelationAnalyzer` with all correlation methods
- [ ] Implement `TemporalAnalyzer` for event analysis
- [ ] Implement visualization functions (6 figure types)
- [ ] Implement main experiment script
- [ ] Test on single model
- [ ] Run on all models
- [ ] Generate comparison tables and figures

---

*This document provides complete specifications for implementing the Reward-Conditioned Attention Analysis. The analysis leverages existing pre-trained models and event mining to produce publishable research on how RL agents learn to focus on reward-relevant features.*
