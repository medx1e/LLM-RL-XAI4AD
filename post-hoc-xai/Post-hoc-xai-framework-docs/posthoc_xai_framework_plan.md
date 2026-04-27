# Post-Hoc XAI Framework for Autonomous Driving

## Implementation Plan for Claude Code

---

# 1. Overview

## 1.1 Purpose

Build a modular, JAX-based framework for applying post-hoc explainability methods to autonomous driving policies trained in V-Max. The framework should:

1. Load any V-Max model (Perceiver, MTR, Wayformer, MGAIL, MLP)
2. Apply multiple XAI methods with a unified API
3. Compare explanations across methods and models
4. Visualize attributions on driving scenes

## 1.2 Design Principles

- **JAX-native:** Use `jax.grad`, `jax.vmap`, `jax.jit` throughout
- **Modular:** Each XAI method is a separate module with the same interface
- **Model-agnostic:** Works on any model that implements the interface
- **Research-ready:** Easy to add new methods, run experiments, generate figures

## 1.3 Target Models

From V-Max pre-trained weights:
```
Attention-based:
- Perceiver/LQ (global attention, 16 latent queries)
- MTR (local attention, k=8 neighbors)
- Wayformer (late fusion, per-modality attention)

Non-attention:
- MGAIL (hierarchical, no self-attention)
- MLP/None (flattened input, fully connected)
```

---

# 2. Project Structure

```
posthoc_xai/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── base.py                 # Abstract base class / Protocol
│   ├── loader.py               # Load V-Max checkpoints
│   ├── perceiver_wrapper.py    # Perceiver-specific wrapper
│   ├── mtr_wrapper.py          # MTR-specific wrapper
│   ├── wayformer_wrapper.py    # Wayformer-specific wrapper
│   ├── mgail_wrapper.py        # MGAIL-specific wrapper
│   └── mlp_wrapper.py          # MLP baseline wrapper
│
├── methods/
│   ├── __init__.py
│   ├── base.py                 # Attribution base class
│   ├── vanilla_gradient.py     # ∂output/∂input
│   ├── integrated_gradients.py # Path-integrated gradients
│   ├── smooth_grad.py          # Averaged noisy gradients
│   ├── gradient_x_input.py     # Gradient × input
│   ├── perturbation.py         # Occlusion / masking
│   ├── feature_ablation.py     # Per-category ablation
│   └── sarfa.py                # RL-specific saliency
│
├── metrics/
│   ├── __init__.py
│   ├── faithfulness.py         # Correlation with attention, deletion curves
│   ├── sparsity.py             # Gini, top-k concentration
│   ├── consistency.py          # Agreement across similar inputs
│   └── sanity_checks.py        # Randomization tests
│
├── visualization/
│   ├── __init__.py
│   ├── heatmaps.py             # Attribution heatmaps
│   ├── bev_overlay.py          # Overlay on bird's eye view
│   ├── comparison_plots.py     # Side-by-side method comparison
│   └── temporal.py             # Attribution over time
│
├── utils/
│   ├── __init__.py
│   ├── observation.py          # Parse V-Max observation structure
│   ├── normalization.py        # Normalize attributions
│   └── io.py                   # Save/load results
│
├── experiments/
│   ├── __init__.py
│   ├── run_all_methods.py      # Apply all methods to scenarios
│   ├── compare_architectures.py # Cross-architecture comparison
│   ├── faithfulness_study.py   # Attention vs gradient faithfulness
│   └── generate_figures.py     # Paper figures
│
└── configs/
    ├── default.yaml            # Default parameters
    ├── methods.yaml            # Method-specific configs
    └── models.yaml             # Model paths and configs
```

---

# 3. Model Interface

## 3.1 Abstract Base Class

```python
# posthoc_xai/models/base.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, Any
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class ModelOutput:
    """Standardized output from model forward pass."""
    action_logits: jnp.ndarray      # (action_dim,) or (batch, action_dim)
    action_mean: jnp.ndarray        # For continuous: mean of action distribution
    action_std: jnp.ndarray         # For continuous: std of action distribution
    value: Optional[jnp.ndarray]    # V(s) if available
    embedding: jnp.ndarray          # Encoder output (for probing)
    attention: Optional[Dict[str, jnp.ndarray]]  # Attention weights if available


class ExplainableModel(ABC):
    """
    Abstract base class for explainable V-Max models.
    
    All model wrappers must implement this interface to work with XAI methods.
    """
    
    @abstractmethod
    def forward(self, observation: jnp.ndarray) -> ModelOutput:
        """
        Full forward pass returning all outputs.
        
        Args:
            observation: V-Max observation array
            
        Returns:
            ModelOutput with action, value, embedding, attention
        """
        pass
    
    @abstractmethod
    def get_action_value(
        self, 
        observation: jnp.ndarray, 
        action_idx: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Get scalar value for gradient computation.
        
        For continuous actions: returns action_mean[action_idx] or sum
        For discrete actions: returns logit[action_idx] or max
        
        This is what we differentiate w.r.t. input for saliency.
        
        Args:
            observation: Input observation
            action_idx: Which action dimension (None = sum all)
            
        Returns:
            Scalar value
        """
        pass
    
    @abstractmethod
    def get_embedding(self, observation: jnp.ndarray) -> jnp.ndarray:
        """
        Get encoder output (latent representation).
        
        Args:
            observation: Input observation
            
        Returns:
            Embedding array (typically 256-dim for V-Max)
        """
        pass
    
    def get_attention(self, observation: jnp.ndarray) -> Optional[Dict[str, jnp.ndarray]]:
        """
        Get attention weights if available.
        
        Returns None for models without attention (MGAIL, MLP).
        
        Args:
            observation: Input observation
            
        Returns:
            Dict with attention arrays, or None
            Expected keys: 'cross_attention', 'self_attention'
            Shapes depend on architecture
        """
        return None
    
    @property
    @abstractmethod
    def observation_structure(self) -> Dict[str, Tuple[int, int]]:
        """
        Return observation structure for attribution grouping.
        
        Returns:
            Dict mapping category name to (start_idx, end_idx) in flattened obs
            Example: {'trajectory': (0, 640), 'roadgraph': (640, 1040), ...}
        """
        pass
    
    @property
    def has_attention(self) -> bool:
        """Whether model has extractable attention weights."""
        return False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier string."""
        pass
```

## 3.2 Model Loader

```python
# posthoc_xai/models/loader.py

import pickle
from pathlib import Path
from typing import Union
import yaml

from .perceiver_wrapper import PerceiverWrapper
from .mtr_wrapper import MTRWrapper
from .wayformer_wrapper import WayformerWrapper
from .mgail_wrapper import MGAILWrapper
from .mlp_wrapper import MLPWrapper


ENCODER_TO_WRAPPER = {
    'perceiver': PerceiverWrapper,
    'lq': PerceiverWrapper,  # LQ uses same wrapper as Perceiver
    'mtr': MTRWrapper,
    'wayformer': WayformerWrapper,
    'mgail': MGAILWrapper,
    'none': MLPWrapper,
}


def load_model(model_dir: Union[str, Path]) -> ExplainableModel:
    """
    Load a V-Max model and wrap it for XAI analysis.
    
    Args:
        model_dir: Path to model directory containing:
            - .hydra/config.yaml
            - model/model_final.pkl
    
    Returns:
        ExplainableModel wrapper
    """
    model_dir = Path(model_dir)
    
    # Load config
    config_path = model_dir / '.hydra' / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load weights
    weights_path = model_dir / 'model' / 'model_final.pkl'
    with open(weights_path, 'rb') as f:
        params = pickle.load(f)
    
    # Determine encoder type
    encoder_type = config['network']['encoder']['type']
    
    # Get appropriate wrapper
    if encoder_type not in ENCODER_TO_WRAPPER:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    wrapper_class = ENCODER_TO_WRAPPER[encoder_type]
    
    return wrapper_class(params=params, config=config)


def load_multiple_models(model_dirs: list) -> Dict[str, ExplainableModel]:
    """
    Load multiple models for comparison.
    
    Args:
        model_dirs: List of model directory paths
        
    Returns:
        Dict mapping model name to wrapper
    """
    models = {}
    for model_dir in model_dirs:
        model = load_model(model_dir)
        models[model.name] = model
    return models
```

## 3.3 Example Wrapper (Perceiver)

```python
# posthoc_xai/models/perceiver_wrapper.py

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional
from functools import partial

from .base import ExplainableModel, ModelOutput


class PerceiverWrapper(ExplainableModel):
    """
    Wrapper for Perceiver/LQ encoder models.
    
    Perceiver uses cross-attention from learned latent queries to input tokens.
    We can extract these attention weights for comparison with post-hoc methods.
    """
    
    def __init__(self, params: dict, config: dict):
        self.params = params
        self.config = config
        self.encoder_config = config['network']['encoder']
        
        # Build the model using V-Max's network builders
        # This requires importing from vmax
        self._build_model()
        
        # Parse observation structure from config
        self._parse_observation_structure()
    
    def _build_model(self):
        """
        Reconstruct the model architecture from config.
        
        This uses V-Max's network building utilities.
        """
        # Import V-Max components
        from vmax.networks import build_encoder, build_policy_head, build_value_head
        
        self.encoder = build_encoder(self.encoder_config)
        self.policy_head = build_policy_head(self.config['network']['policy'])
        self.value_head = build_value_head(self.config['network']['value'])
    
    def _parse_observation_structure(self):
        """
        Determine observation boundaries from config.
        
        V-Max observations are structured as:
        [trajectory_features | roadgraph_features | traffic_light_features | path_features]
        """
        obs_config = self.config['observation']
        
        # Calculate sizes based on config
        # These formulas match V-Max's observation function
        
        num_objects = obs_config.get('num_objects', 8)
        num_steps = obs_config.get('steps', 5)
        traj_features_per_step = 7  # x, y, vx, vy, yaw, length, width
        
        trajectory_size = num_objects * num_steps * traj_features_per_step
        
        max_roadgraph = obs_config.get('max_roadgraph', 200)
        roadgraph_features = 5  # x, y, dx, dy, valid
        roadgraph_size = max_roadgraph * roadgraph_features
        
        max_tl = obs_config.get('max_traffic_lights', 5)
        tl_features = 4  # x, y, state, valid
        traffic_light_size = max_tl * tl_features
        
        path_points = obs_config.get('path_points', 10)
        path_features = 2  # x, y
        path_size = path_points * path_features
        
        # Build structure dict
        idx = 0
        self._obs_structure = {}
        
        self._obs_structure['trajectory'] = (idx, idx + trajectory_size)
        idx += trajectory_size
        
        self._obs_structure['roadgraph'] = (idx, idx + roadgraph_size)
        idx += roadgraph_size
        
        self._obs_structure['traffic_light'] = (idx, idx + traffic_light_size)
        idx += traffic_light_size
        
        self._obs_structure['path_target'] = (idx, idx + path_size)
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, observation: jnp.ndarray) -> ModelOutput:
        """Forward pass with attention extraction."""
        
        # Encode observation
        embedding, attention_weights = self._encode_with_attention(observation)
        
        # Policy head
        action_mean, action_std = self.policy_head.apply(
            self.params['policy'], embedding
        )
        
        # Value head
        value = self.value_head.apply(
            self.params['value'], embedding
        )
        
        return ModelOutput(
            action_logits=action_mean,  # For continuous, logits = mean
            action_mean=action_mean,
            action_std=action_std,
            value=value,
            embedding=embedding,
            attention=attention_weights
        )
    
    def _encode_with_attention(
        self, 
        observation: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Run encoder and extract attention weights.
        
        This requires modifying the encoder forward pass to return attention.
        """
        # The actual implementation depends on V-Max's encoder structure
        # Typically we need to:
        # 1. Run embedding layers
        # 2. Run cross-attention with return_attention=True
        # 3. Collect attention weights from each layer
        
        # Placeholder - actual implementation needs V-Max internals
        embedding = self.encoder.apply(
            self.params['encoder'], 
            observation,
            return_attention=True  # Modified encoder
        )
        
        # Extract attention from encoder state
        # Structure depends on Perceiver implementation
        attention_weights = {
            'cross_attention': self._extract_cross_attention(),
            'self_attention': self._extract_self_attention(),
        }
        
        return embedding, attention_weights
    
    @partial(jax.jit, static_argnums=(0, 2))
    def get_action_value(
        self, 
        observation: jnp.ndarray, 
        action_idx: Optional[int] = None
    ) -> jnp.ndarray:
        """Get scalar for gradient computation."""
        output = self.forward(observation)
        
        if action_idx is not None:
            return output.action_mean[action_idx]
        else:
            # Default: sum of action means (for overall saliency)
            return jnp.sum(output.action_mean)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_embedding(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Get encoder output."""
        output = self.forward(observation)
        return output.embedding
    
    def get_attention(self, observation: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Get attention weights."""
        output = self.forward(observation)
        return output.attention
    
    @property
    def observation_structure(self) -> Dict[str, Tuple[int, int]]:
        return self._obs_structure
    
    @property
    def has_attention(self) -> bool:
        return True
    
    @property
    def name(self) -> str:
        encoder_type = self.encoder_config['type']
        return f"perceiver_{encoder_type}"
```

---

# 4. XAI Methods

## 4.1 Attribution Base Class

```python
# posthoc_xai/methods/base.py

from abc import ABC, abstractmethod
from typing import Optional, Dict
import jax.numpy as jnp
from flax import struct

from ..models.base import ExplainableModel


@struct.dataclass
class Attribution:
    """
    Standardized attribution output.
    
    All XAI methods return this format for easy comparison.
    """
    # Raw attribution values (same shape as observation)
    raw: jnp.ndarray
    
    # Normalized attribution (sum to 1, absolute values)
    normalized: jnp.ndarray
    
    # Per-category aggregated importance
    category_importance: Dict[str, float]
    
    # Metadata
    method_name: str
    target_action: Optional[int]
    computation_time_ms: float
    
    # Optional: intermediate results for debugging
    extras: Optional[Dict] = None


class AttributionMethod(ABC):
    """
    Base class for all attribution methods.
    
    Subclasses implement `compute_raw_attribution` and can override
    normalization and aggregation if needed.
    """
    
    def __init__(self, model: ExplainableModel, **kwargs):
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute raw attribution values.
        
        Args:
            observation: Input observation
            target_action: Which action dimension to explain (None = all)
            
        Returns:
            Attribution array with same shape as observation
        """
        pass
    
    def normalize(self, raw_attribution: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize attribution to sum to 1 (absolute values).
        
        Override for method-specific normalization.
        """
        abs_attr = jnp.abs(raw_attribution)
        total = jnp.sum(abs_attr) + 1e-10
        return abs_attr / total
    
    def aggregate_by_category(
        self,
        normalized_attribution: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Sum attribution by observation category.
        """
        structure = self.model.observation_structure
        flat_attr = normalized_attribution.flatten()
        
        category_importance = {}
        for category, (start, end) in structure.items():
            category_importance[category] = float(jnp.sum(flat_attr[start:end]))
        
        return category_importance
    
    def __call__(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> Attribution:
        """
        Compute full attribution with normalization and aggregation.
        """
        import time
        
        start_time = time.time()
        
        # Compute raw attribution
        raw = self.compute_raw_attribution(observation, target_action)
        
        # Normalize
        normalized = self.normalize(raw)
        
        # Aggregate by category
        category_importance = self.aggregate_by_category(normalized)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return Attribution(
            raw=raw,
            normalized=normalized,
            category_importance=category_importance,
            method_name=self.name,
            target_action=target_action,
            computation_time_ms=elapsed_ms,
        )
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Method identifier."""
        pass
```

## 4.2 Vanilla Gradients

```python
# posthoc_xai/methods/vanilla_gradient.py

import jax
import jax.numpy as jnp
from typing import Optional
from functools import partial

from .base import AttributionMethod


class VanillaGradient(AttributionMethod):
    """
    Vanilla gradient saliency.
    
    Computes ∂f(x)/∂x where f is the action value function.
    
    Simple and fast, but can be noisy.
    
    Reference: Simonyan et al., "Deep Inside Convolutional Networks" (2014)
    """
    
    @partial(jax.jit, static_argnums=(0, 2))
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute gradient of action value w.r.t. input.
        """
        def forward_fn(obs):
            return self.model.get_action_value(obs, target_action)
        
        grad_fn = jax.grad(forward_fn)
        gradient = grad_fn(observation)
        
        return gradient
    
    @property
    def name(self) -> str:
        return "vanilla_gradient"
```

## 4.3 Integrated Gradients

```python
# posthoc_xai/methods/integrated_gradients.py

import jax
import jax.numpy as jnp
from typing import Optional
from functools import partial

from .base import AttributionMethod


class IntegratedGradients(AttributionMethod):
    """
    Integrated Gradients attribution.
    
    Computes attribution by integrating gradients along a path from baseline to input:
    
    IG(x) = (x - x') × ∫₀¹ ∂f(x' + α(x-x'))/∂x dα
    
    Satisfies completeness and sensitivity axioms.
    
    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017)
    """
    
    def __init__(self, model, n_steps: int = 50, baseline: str = 'zero', **kwargs):
        """
        Args:
            model: ExplainableModel instance
            n_steps: Number of interpolation steps (more = more accurate)
            baseline: Baseline type - 'zero', 'mean', 'noise', or array
        """
        super().__init__(model, **kwargs)
        self.n_steps = n_steps
        self.baseline_type = baseline
    
    def get_baseline(self, observation: jnp.ndarray) -> jnp.ndarray:
        """
        Get baseline observation for IG computation.
        
        The choice of baseline affects interpretation:
        - Zero: "absence of features"
        - Mean: "average observation"
        - Noise: "uninformative input"
        """
        if self.baseline_type == 'zero':
            return jnp.zeros_like(observation)
        elif self.baseline_type == 'mean':
            # Use per-feature mean (would need to be precomputed)
            return jnp.zeros_like(observation)  # Placeholder
        elif self.baseline_type == 'noise':
            key = jax.random.PRNGKey(0)
            return jax.random.normal(key, observation.shape) * 0.01
        elif isinstance(self.baseline_type, jnp.ndarray):
            return self.baseline_type
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")
    
    @partial(jax.jit, static_argnums=(0, 2))
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute Integrated Gradients.
        """
        baseline = self.get_baseline(observation)
        
        # Create interpolation path
        alphas = jnp.linspace(0, 1, self.n_steps)
        
        def compute_gradient_at_alpha(alpha):
            """Gradient at interpolated point."""
            interpolated = baseline + alpha * (observation - baseline)
            
            def forward_fn(x):
                return self.model.get_action_value(x, target_action)
            
            return jax.grad(forward_fn)(interpolated)
        
        # Compute gradients at all alphas in parallel
        path_gradients = jax.vmap(compute_gradient_at_alpha)(alphas)
        
        # Average gradients (Riemann sum approximation of integral)
        avg_gradients = jnp.mean(path_gradients, axis=0)
        
        # Multiply by input difference
        integrated_gradients = (observation - baseline) * avg_gradients
        
        return integrated_gradients
    
    @property
    def name(self) -> str:
        return f"integrated_gradients_{self.n_steps}steps"
```

## 4.4 SmoothGrad

```python
# posthoc_xai/methods/smooth_grad.py

import jax
import jax.numpy as jnp
from typing import Optional
from functools import partial

from .base import AttributionMethod


class SmoothGrad(AttributionMethod):
    """
    SmoothGrad: Reducing noise in gradient-based saliency.
    
    Averages gradients over noisy copies of the input:
    
    SG(x) = (1/n) Σᵢ ∂f(x + εᵢ)/∂x,  εᵢ ~ N(0, σ²)
    
    Reference: Smilkov et al., "SmoothGrad" (2017)
    """
    
    def __init__(
        self, 
        model, 
        n_samples: int = 50, 
        noise_std: float = 0.1,
        **kwargs
    ):
        """
        Args:
            model: ExplainableModel instance
            n_samples: Number of noisy samples
            noise_std: Standard deviation of Gaussian noise
        """
        super().__init__(model, **kwargs)
        self.n_samples = n_samples
        self.noise_std = noise_std
    
    @partial(jax.jit, static_argnums=(0, 2))
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute SmoothGrad attribution.
        """
        key = jax.random.PRNGKey(42)
        
        # Generate noisy samples
        noise = jax.random.normal(key, (self.n_samples,) + observation.shape)
        noisy_inputs = observation + self.noise_std * noise
        
        def compute_gradient(noisy_obs):
            def forward_fn(x):
                return self.model.get_action_value(x, target_action)
            return jax.grad(forward_fn)(noisy_obs)
        
        # Compute gradients for all noisy samples
        all_gradients = jax.vmap(compute_gradient)(noisy_inputs)
        
        # Average
        smooth_gradient = jnp.mean(all_gradients, axis=0)
        
        return smooth_gradient
    
    @property
    def name(self) -> str:
        return f"smooth_grad_{self.n_samples}samples"
```

## 4.5 Gradient × Input

```python
# posthoc_xai/methods/gradient_x_input.py

import jax
import jax.numpy as jnp
from typing import Optional
from functools import partial

from .base import AttributionMethod


class GradientXInput(AttributionMethod):
    """
    Gradient × Input attribution.
    
    Simple modification of vanilla gradients:
    
    GxI(x) = x ⊙ ∂f(x)/∂x
    
    Often more meaningful than raw gradients for non-zero inputs.
    
    Reference: Shrikumar et al., "Learning Important Features Through Propagating Activation Differences" (2017)
    """
    
    @partial(jax.jit, static_argnums=(0, 2))
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute Gradient × Input.
        """
        def forward_fn(obs):
            return self.model.get_action_value(obs, target_action)
        
        gradient = jax.grad(forward_fn)(observation)
        
        return observation * gradient
    
    @property
    def name(self) -> str:
        return "gradient_x_input"
```

## 4.6 Perturbation-Based Attribution

```python
# posthoc_xai/methods/perturbation.py

import jax
import jax.numpy as jnp
from typing import Optional, Literal
from functools import partial

from .base import AttributionMethod


class PerturbationAttribution(AttributionMethod):
    """
    Perturbation-based (occlusion) attribution.
    
    Measures importance by observing output change when input features are masked/perturbed.
    
    For feature i:
    Importance(i) = |f(x) - f(x with feature i masked)|
    
    Model-agnostic but can be slow for high-dimensional inputs.
    
    Reference: Zeiler & Fergus, "Visualizing and Understanding CNNs" (2014)
    """
    
    def __init__(
        self, 
        model, 
        perturbation_type: Literal['zero', 'mean', 'noise'] = 'zero',
        aggregate_features: bool = True,
        **kwargs
    ):
        """
        Args:
            model: ExplainableModel instance
            perturbation_type: How to perturb features
            aggregate_features: If True, perturb by category instead of per-feature
        """
        super().__init__(model, **kwargs)
        self.perturbation_type = perturbation_type
        self.aggregate_features = aggregate_features
    
    def get_perturbation_value(self, original_value: jnp.ndarray) -> jnp.ndarray:
        """Get value to replace masked features."""
        if self.perturbation_type == 'zero':
            return jnp.zeros_like(original_value)
        elif self.perturbation_type == 'mean':
            return jnp.ones_like(original_value) * jnp.mean(original_value)
        elif self.perturbation_type == 'noise':
            key = jax.random.PRNGKey(0)
            return jax.random.normal(key, original_value.shape) * 0.01
        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation_type}")
    
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute perturbation-based attribution.
        """
        # Get baseline output
        baseline_output = self.model.get_action_value(observation, target_action)
        
        flat_obs = observation.flatten()
        attribution = jnp.zeros_like(flat_obs)
        
        if self.aggregate_features:
            # Perturb by category (much faster)
            for category, (start, end) in self.model.observation_structure.items():
                perturbed = flat_obs.at[start:end].set(
                    self.get_perturbation_value(flat_obs[start:end])
                )
                perturbed_obs = perturbed.reshape(observation.shape)
                perturbed_output = self.model.get_action_value(perturbed_obs, target_action)
                
                importance = jnp.abs(baseline_output - perturbed_output)
                # Distribute importance equally across features in category
                per_feature_importance = importance / (end - start)
                attribution = attribution.at[start:end].set(per_feature_importance)
        else:
            # Perturb each feature individually (slow but precise)
            def compute_single_importance(idx):
                perturbed = flat_obs.at[idx].set(
                    self.get_perturbation_value(flat_obs[idx])
                )
                perturbed_obs = perturbed.reshape(observation.shape)
                perturbed_output = self.model.get_action_value(perturbed_obs, target_action)
                return jnp.abs(baseline_output - perturbed_output)
            
            # Use vmap for parallelization
            indices = jnp.arange(len(flat_obs))
            attribution = jax.vmap(compute_single_importance)(indices)
        
        return attribution.reshape(observation.shape)
    
    @property
    def name(self) -> str:
        agg = "category" if self.aggregate_features else "feature"
        return f"perturbation_{self.perturbation_type}_{agg}"
```

## 4.7 Feature Ablation

```python
# posthoc_xai/methods/feature_ablation.py

import jax
import jax.numpy as jnp
from typing import Optional, Dict
from functools import partial

from .base import AttributionMethod, Attribution


class FeatureAblation(AttributionMethod):
    """
    Feature ablation by category.
    
    Measures importance of entire feature categories by removing them:
    
    Importance(category) = |f(x) - f(x without category)|
    
    Useful for high-level "what type of information matters" analysis.
    """
    
    def __init__(self, model, replacement: str = 'zero', **kwargs):
        """
        Args:
            model: ExplainableModel instance
            replacement: How to replace ablated features ('zero', 'mean')
        """
        super().__init__(model, **kwargs)
        self.replacement = replacement
    
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute per-category ablation importance.
        
        Returns array with constant values per category (for compatibility).
        """
        baseline_output = self.model.get_action_value(observation, target_action)
        flat_obs = observation.flatten()
        attribution = jnp.zeros_like(flat_obs)
        
        for category, (start, end) in self.model.observation_structure.items():
            # Ablate this category
            if self.replacement == 'zero':
                replacement_val = jnp.zeros(end - start)
            else:
                replacement_val = jnp.ones(end - start) * jnp.mean(flat_obs[start:end])
            
            ablated = flat_obs.at[start:end].set(replacement_val)
            ablated_obs = ablated.reshape(observation.shape)
            ablated_output = self.model.get_action_value(ablated_obs, target_action)
            
            # Importance = change when category is removed
            importance = jnp.abs(baseline_output - ablated_output)
            
            # Assign to all features in category
            attribution = attribution.at[start:end].set(importance)
        
        return attribution.reshape(observation.shape)
    
    def compute_category_importance(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute importance directly per category (more interpretable output).
        """
        baseline_output = self.model.get_action_value(observation, target_action)
        flat_obs = observation.flatten()
        
        importance = {}
        for category, (start, end) in self.model.observation_structure.items():
            if self.replacement == 'zero':
                replacement_val = jnp.zeros(end - start)
            else:
                replacement_val = jnp.ones(end - start) * jnp.mean(flat_obs[start:end])
            
            ablated = flat_obs.at[start:end].set(replacement_val)
            ablated_obs = ablated.reshape(observation.shape)
            ablated_output = self.model.get_action_value(ablated_obs, target_action)
            
            importance[category] = float(jnp.abs(baseline_output - ablated_output))
        
        return importance
    
    @property
    def name(self) -> str:
        return f"feature_ablation_{self.replacement}"
```

## 4.8 SARFA (RL-Specific)

```python
# posthoc_xai/methods/sarfa.py

import jax
import jax.numpy as jnp
from typing import Optional
from functools import partial

from .base import AttributionMethod


class SARFA(AttributionMethod):
    """
    Specific and Relevant Feature Attribution (SARFA).
    
    RL-specific saliency method that considers both:
    - Relevance: Does the feature affect Q-value of chosen action?
    - Specificity: Does the feature specifically affect THIS action vs others?
    
    SARFA(x, a) = Relevance(x, a) × Specificity(x, a)
    
    where:
    - Relevance = |Q(x, a) - Q(x', a)| (how much feature affects chosen action)
    - Specificity = |Q(x, a) - Q(x', a)| / Σ_a' |Q(x, a') - Q(x', a')| 
                    (how specific is the effect to this action)
    
    Reference: Puri et al., "Explain Your Move: Understanding Agent Actions Using Focused Feature Saliency" (2020)
    """
    
    def __init__(
        self, 
        model, 
        perturbation_type: str = 'zero',
        aggregate_features: bool = True,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.perturbation_type = perturbation_type
        self.aggregate_features = aggregate_features
    
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Compute SARFA attribution.
        
        For continuous actions, we treat each action dimension as a separate "action"
        and compute specificity across dimensions.
        """
        # Get baseline outputs for all action dimensions
        baseline_output = self.model.forward(observation)
        baseline_actions = baseline_output.action_mean  # (action_dim,)
        
        flat_obs = observation.flatten()
        attribution = jnp.zeros_like(flat_obs)
        
        # Get target action index
        if target_action is None:
            target_action = 0  # Default to first action dimension
        
        if self.aggregate_features:
            for category, (start, end) in self.model.observation_structure.items():
                # Perturb category
                if self.perturbation_type == 'zero':
                    perturbed_val = jnp.zeros(end - start)
                else:
                    perturbed_val = jnp.mean(flat_obs[start:end]) * jnp.ones(end - start)
                
                perturbed = flat_obs.at[start:end].set(perturbed_val)
                perturbed_obs = perturbed.reshape(observation.shape)
                perturbed_output = self.model.forward(perturbed_obs)
                perturbed_actions = perturbed_output.action_mean
                
                # Relevance: change in target action
                relevance = jnp.abs(baseline_actions[target_action] - perturbed_actions[target_action])
                
                # Specificity: change in target relative to all actions
                all_changes = jnp.abs(baseline_actions - perturbed_actions)
                total_change = jnp.sum(all_changes) + 1e-10
                specificity = relevance / total_change
                
                # SARFA score
                sarfa_score = relevance * specificity
                
                # Distribute to features
                per_feature = sarfa_score / (end - start)
                attribution = attribution.at[start:end].set(per_feature)
        else:
            # Per-feature SARFA (slow)
            def compute_single_sarfa(idx):
                if self.perturbation_type == 'zero':
                    perturbed = flat_obs.at[idx].set(0.0)
                else:
                    perturbed = flat_obs.at[idx].set(jnp.mean(flat_obs))
                
                perturbed_obs = perturbed.reshape(observation.shape)
                perturbed_output = self.model.forward(perturbed_obs)
                perturbed_actions = perturbed_output.action_mean
                
                relevance = jnp.abs(baseline_actions[target_action] - perturbed_actions[target_action])
                all_changes = jnp.abs(baseline_actions - perturbed_actions)
                specificity = relevance / (jnp.sum(all_changes) + 1e-10)
                
                return relevance * specificity
            
            indices = jnp.arange(len(flat_obs))
            attribution = jax.vmap(compute_single_sarfa)(indices)
        
        return attribution.reshape(observation.shape)
    
    @property
    def name(self) -> str:
        return "sarfa"
```

---

# 5. Metrics

## 5.1 Faithfulness Metrics

```python
# posthoc_xai/metrics/faithfulness.py

import jax.numpy as jnp
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple
import numpy as np

from ..methods.base import Attribution


def attention_gradient_correlation(
    attention_attr: Attribution,
    gradient_attr: Attribution,
) -> Dict[str, float]:
    """
    Compute correlation between attention-based and gradient-based attributions.
    
    High correlation suggests attention is "faithful" to actual feature importance.
    
    Returns:
        Dict with pearson_r, spearman_rho, and p-values
    """
    attn_flat = np.array(attention_attr.normalized.flatten())
    grad_flat = np.array(gradient_attr.normalized.flatten())
    
    pearson_r, pearson_p = pearsonr(attn_flat, grad_flat)
    spearman_rho, spearman_p = spearmanr(attn_flat, grad_flat)
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
    }


def deletion_curve(
    model,
    observation: jnp.ndarray,
    attribution: Attribution,
    n_steps: int = 20,
    target_action: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deletion curve: progressively remove most important features.
    
    A good attribution should cause rapid output decrease when removing
    top-attributed features.
    
    Args:
        model: ExplainableModel
        observation: Input
        attribution: Attribution to evaluate
        n_steps: Number of deletion steps
        target_action: Which action to track
        
    Returns:
        (percentages, outputs): Arrays for plotting
    """
    flat_obs = observation.flatten()
    flat_attr = np.array(attribution.normalized.flatten())
    
    # Sort features by attribution (descending)
    sorted_indices = np.argsort(flat_attr)[::-1]
    
    baseline_output = float(model.get_action_value(observation, target_action))
    
    percentages = np.linspace(0, 1, n_steps)
    outputs = [baseline_output]
    
    for pct in percentages[1:]:
        n_remove = int(pct * len(flat_obs))
        indices_to_remove = sorted_indices[:n_remove]
        
        # Zero out top features
        modified_obs = flat_obs.copy()
        modified_obs[indices_to_remove] = 0
        modified_obs = jnp.array(modified_obs.reshape(observation.shape))
        
        output = float(model.get_action_value(modified_obs, target_action))
        outputs.append(output)
    
    return percentages, np.array(outputs)


def insertion_curve(
    model,
    observation: jnp.ndarray,
    attribution: Attribution,
    n_steps: int = 20,
    target_action: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute insertion curve: progressively add most important features to blank input.
    
    A good attribution should cause rapid output increase when adding
    top-attributed features.
    """
    flat_obs = observation.flatten()
    flat_attr = np.array(attribution.normalized.flatten())
    
    sorted_indices = np.argsort(flat_attr)[::-1]
    
    # Start from zero baseline
    baseline_obs = jnp.zeros_like(observation)
    baseline_output = float(model.get_action_value(baseline_obs, target_action))
    
    percentages = np.linspace(0, 1, n_steps)
    outputs = [baseline_output]
    
    for pct in percentages[1:]:
        n_add = int(pct * len(flat_obs))
        indices_to_add = sorted_indices[:n_add]
        
        modified_obs = np.zeros_like(flat_obs)
        modified_obs[indices_to_add] = flat_obs[indices_to_add]
        modified_obs = jnp.array(modified_obs.reshape(observation.shape))
        
        output = float(model.get_action_value(modified_obs, target_action))
        outputs.append(output)
    
    return percentages, np.array(outputs)


def area_under_deletion_curve(deletion_outputs: np.ndarray) -> float:
    """
    Compute AUC for deletion curve.
    
    Lower is better (faster decrease = better attribution).
    """
    return np.trapz(deletion_outputs) / len(deletion_outputs)


def area_under_insertion_curve(insertion_outputs: np.ndarray) -> float:
    """
    Compute AUC for insertion curve.
    
    Higher is better (faster increase = better attribution).
    """
    return np.trapz(insertion_outputs) / len(insertion_outputs)
```

## 5.2 Sparsity Metrics

```python
# posthoc_xai/metrics/sparsity.py

import jax.numpy as jnp
import numpy as np
from typing import Dict

from ..methods.base import Attribution


def gini_coefficient(attribution: Attribution) -> float:
    """
    Gini coefficient of attribution distribution.
    
    0 = perfectly uniform (all features equally important)
    1 = perfectly sparse (one feature has all importance)
    """
    values = np.sort(np.array(attribution.normalized.flatten()))
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum(np.arange(1, n+1) * values) / (n * np.sum(values) + 1e-10)) - (n + 1) / n


def top_k_concentration(attribution: Attribution, k: int = 10) -> float:
    """
    Fraction of total attribution in top-k features.
    
    Higher = more concentrated attribution.
    """
    values = np.array(attribution.normalized.flatten())
    sorted_values = np.sort(values)[::-1]
    return float(np.sum(sorted_values[:k]))


def entropy(attribution: Attribution) -> float:
    """
    Shannon entropy of attribution distribution.
    
    Lower = more concentrated, Higher = more uniform.
    Normalized by max entropy.
    """
    values = np.array(attribution.normalized.flatten())
    values = values + 1e-10  # Avoid log(0)
    values = values / values.sum()
    
    ent = -np.sum(values * np.log(values))
    max_ent = np.log(len(values))
    
    return ent / max_ent  # Normalized to [0, 1]


def compute_all_sparsity_metrics(attribution: Attribution) -> Dict[str, float]:
    """Compute all sparsity metrics for an attribution."""
    return {
        'gini': gini_coefficient(attribution),
        'top_10_concentration': top_k_concentration(attribution, k=10),
        'top_50_concentration': top_k_concentration(attribution, k=50),
        'entropy': entropy(attribution),
    }
```

## 5.3 Consistency Metrics

```python
# posthoc_xai/metrics/consistency.py

import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr
from typing import List

from ..methods.base import Attribution


def attribution_consistency(attributions: List[Attribution]) -> float:
    """
    Measure consistency of attributions across similar inputs.
    
    Computes average pairwise correlation of attribution vectors.
    
    High consistency = method produces stable explanations.
    """
    if len(attributions) < 2:
        return 1.0
    
    correlations = []
    for i in range(len(attributions)):
        for j in range(i + 1, len(attributions)):
            a1 = np.array(attributions[i].normalized.flatten())
            a2 = np.array(attributions[j].normalized.flatten())
            corr, _ = pearsonr(a1, a2)
            correlations.append(corr)
    
    return float(np.mean(correlations))


def category_consistency(attributions: List[Attribution]) -> float:
    """
    Measure consistency at category level (trajectory, road, etc.).
    
    More robust than feature-level consistency.
    """
    if len(attributions) < 2:
        return 1.0
    
    # Extract category importance vectors
    categories = list(attributions[0].category_importance.keys())
    
    vectors = []
    for attr in attributions:
        vec = [attr.category_importance[c] for c in categories]
        vectors.append(vec)
    
    correlations = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            corr, _ = pearsonr(vectors[i], vectors[j])
            correlations.append(corr)
    
    return float(np.mean(correlations))
```

---

# 6. Visualization

## 6.1 Attribution Heatmaps

```python
# posthoc_xai/visualization/heatmaps.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

from ..methods.base import Attribution


def plot_category_importance(
    attribution: Attribution,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Bar plot of per-category importance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    categories = list(attribution.category_importance.keys())
    values = [attribution.category_importance[c] for c in categories]
    
    bars = ax.bar(categories, values, color='steelblue')
    ax.set_ylabel('Importance')
    ax.set_title(title or f'{attribution.method_name} - Category Importance')
    ax.set_ylim(0, max(values) * 1.1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    return fig


def plot_method_comparison(
    attributions: List[Attribution],
    title: str = "Method Comparison",
) -> plt.Figure:
    """
    Side-by-side comparison of multiple methods' category importance.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = list(attributions[0].category_importance.keys())
    n_methods = len(attributions)
    n_categories = len(categories)
    
    x = np.arange(n_categories)
    width = 0.8 / n_methods
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    
    for i, attr in enumerate(attributions):
        values = [attr.category_importance[c] for c in categories]
        offset = (i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=attr.method_name, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Importance')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_deletion_insertion_curves(
    deletion_data: List[tuple],  # [(method_name, percentages, outputs), ...]
    insertion_data: List[tuple],
) -> plt.Figure:
    """
    Plot deletion and insertion curves for multiple methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(deletion_data)))
    
    # Deletion curve
    ax = axes[0]
    for i, (name, pcts, outputs) in enumerate(deletion_data):
        ax.plot(pcts, outputs, label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Fraction of Features Removed')
    ax.set_ylabel('Model Output')
    ax.set_title('Deletion Curve (↓ is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Insertion curve
    ax = axes[1]
    for i, (name, pcts, outputs) in enumerate(insertion_data):
        ax.plot(pcts, outputs, label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Fraction of Features Added')
    ax.set_ylabel('Model Output')
    ax.set_title('Insertion Curve (↑ is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## 6.2 BEV Overlay Visualization

```python
# posthoc_xai/visualization/bev_overlay.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, Optional

from ..methods.base import Attribution


def plot_bev_attribution(
    attribution: Attribution,
    scenario_data: Dict,  # Contains positions, road geometry, etc.
    ax: Optional[plt.Axes] = None,
    colormap: str = 'hot',
) -> plt.Figure:
    """
    Overlay attribution on bird's eye view of driving scene.
    
    scenario_data should contain:
    - ego_position: (2,)
    - other_positions: (N, 2)
    - road_points: (M, 2)
    - traffic_light_positions: (K, 2)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # Extract category attributions
    cat_importance = attribution.category_importance
    
    # Normalize for coloring
    max_importance = max(cat_importance.values())
    
    # Plot road
    road_pts = scenario_data.get('road_points', [])
    road_importance = cat_importance.get('roadgraph', 0) / max_importance
    for pt in road_pts:
        ax.plot(pt[0], pt[1], 'o', color=plt.cm.Blues(road_importance), 
                markersize=2, alpha=0.5)
    
    # Plot vehicles
    other_pos = scenario_data.get('other_positions', [])
    traj_importance = cat_importance.get('trajectory', 0) / max_importance
    for pos in other_pos:
        rect = patches.Rectangle(
            (pos[0] - 2.5, pos[1] - 1), 5, 2,
            linewidth=2,
            edgecolor='black',
            facecolor=plt.cm.Reds(traj_importance),
            alpha=0.7
        )
        ax.add_patch(rect)
    
    # Plot ego vehicle
    ego_pos = scenario_data.get('ego_position', [0, 0])
    ego_rect = patches.Rectangle(
        (ego_pos[0] - 2.5, ego_pos[1] - 1), 5, 2,
        linewidth=3,
        edgecolor='green',
        facecolor='lightgreen',
    )
    ax.add_patch(ego_rect)
    
    # Plot traffic lights
    tl_pos = scenario_data.get('traffic_light_positions', [])
    tl_importance = cat_importance.get('traffic_light', 0) / max_importance
    for pos in tl_pos:
        ax.plot(pos[0], pos[1], 's', color=plt.cm.Oranges(tl_importance),
                markersize=15, markeredgecolor='black')
    
    # Plot path
    path_pts = scenario_data.get('path_points', [])
    path_importance = cat_importance.get('path_target', 0) / max_importance
    if len(path_pts) > 0:
        path_pts = np.array(path_pts)
        ax.plot(path_pts[:, 0], path_pts[:, 1], '--', 
                color=plt.cm.Greens(path_importance), linewidth=3)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=plt.cm.Reds(traj_importance), label=f'Vehicles ({cat_importance.get("trajectory", 0):.3f})'),
        patches.Patch(facecolor=plt.cm.Blues(road_importance), label=f'Road ({cat_importance.get("roadgraph", 0):.3f})'),
        patches.Patch(facecolor=plt.cm.Oranges(tl_importance), label=f'Traffic Light ({cat_importance.get("traffic_light", 0):.3f})'),
        patches.Patch(facecolor=plt.cm.Greens(path_importance), label=f'Path ({cat_importance.get("path_target", 0):.3f})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'{attribution.method_name} Attribution')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig
```

---

# 7. Main API

## 7.1 High-Level Interface

```python
# posthoc_xai/__init__.py

from .models.loader import load_model, load_multiple_models
from .methods import (
    VanillaGradient,
    IntegratedGradients,
    SmoothGrad,
    GradientXInput,
    PerturbationAttribution,
    FeatureAblation,
    SARFA,
)
from .metrics import faithfulness, sparsity, consistency
from .visualization import heatmaps, bev_overlay


# Convenience function for running all methods
def explain(
    model,
    observation,
    methods=['vanilla_gradient', 'integrated_gradients', 'perturbation', 'sarfa'],
    target_action=None,
):
    """
    Run multiple XAI methods on a single observation.
    
    Args:
        model: ExplainableModel or path to model directory
        observation: Input observation array
        methods: List of method names to run
        target_action: Which action dimension to explain
        
    Returns:
        Dict mapping method name to Attribution
    """
    if isinstance(model, str):
        model = load_model(model)
    
    METHOD_CLASSES = {
        'vanilla_gradient': VanillaGradient,
        'integrated_gradients': IntegratedGradients,
        'smooth_grad': SmoothGrad,
        'gradient_x_input': GradientXInput,
        'perturbation': PerturbationAttribution,
        'feature_ablation': FeatureAblation,
        'sarfa': SARFA,
    }
    
    results = {}
    for method_name in methods:
        if method_name not in METHOD_CLASSES:
            raise ValueError(f"Unknown method: {method_name}")
        
        method = METHOD_CLASSES[method_name](model)
        results[method_name] = method(observation, target_action)
    
    return results


def compare_methods(
    model,
    observation,
    methods=['vanilla_gradient', 'integrated_gradients', 'perturbation'],
    target_action=None,
):
    """
    Run methods and compute comparison metrics.
    """
    attributions = explain(model, observation, methods, target_action)
    
    # Compute pairwise correlations
    from itertools import combinations
    correlations = {}
    for m1, m2 in combinations(methods, 2):
        corr = faithfulness.attention_gradient_correlation(
            attributions[m1], attributions[m2]
        )
        correlations[f'{m1}_vs_{m2}'] = corr['pearson_r']
    
    # Compute sparsity for each
    sparsity_metrics = {}
    for name, attr in attributions.items():
        sparsity_metrics[name] = sparsity.compute_all_sparsity_metrics(attr)
    
    return {
        'attributions': attributions,
        'correlations': correlations,
        'sparsity': sparsity_metrics,
    }
```

## 7.2 Example Usage

```python
# Example: Analyze a single scenario

import posthoc_xai as xai

# Load model
model = xai.load_model('runs_rlc/womd_sac_road_perceiver_minimal_42')

# Get observation from V-Max environment
observation = env.get_observation()

# Run all XAI methods
attributions = xai.explain(model, observation)

# Visualize comparison
fig = xai.heatmaps.plot_method_comparison(list(attributions.values()))
fig.savefig('method_comparison.pdf')

# Compare with attention (if available)
if model.has_attention:
    attention = model.get_attention(observation)
    
    # Compute faithfulness
    ig_attr = attributions['integrated_gradients']
    faith = xai.faithfulness.attention_gradient_correlation(
        attention_attr,  # Need to convert attention to Attribution format
        ig_attr
    )
    print(f"Attention-IG correlation: {faith['pearson_r']:.3f}")

# Compute deletion curves
for name, attr in attributions.items():
    pcts, outputs = xai.faithfulness.deletion_curve(model, observation, attr)
    auc = xai.faithfulness.area_under_deletion_curve(outputs)
    print(f"{name} deletion AUC: {auc:.3f}")
```

---

# 8. Experiments to Run

## 8.1 Cross-Architecture Comparison

```python
# experiments/compare_architectures.py

"""
Compare XAI explanations across different architectures.

Research questions:
- Do different architectures produce different explanations?
- Which architecture is most "explainable" (sparsest, most consistent)?
"""

import posthoc_xai as xai
from pathlib import Path
import pandas as pd

MODELS = [
    'runs_rlc/womd_sac_road_perceiver_minimal_42',
    'runs_rlc/womd_sac_road_mtr_minimal_42',
    'runs_rlc/womd_sac_road_wayformer_minimal_42',
    'runs_rlc/womd_sac_road_mgail_minimal_42',
    'runs_rlc/womd_sac_road_none_minimal_42',
]

METHODS = ['integrated_gradients', 'perturbation', 'sarfa']

def run_experiment(scenarios, output_dir):
    results = []
    
    for model_path in MODELS:
        model = xai.load_model(model_path)
        
        for scenario in scenarios:
            obs = scenario['observation']
            
            comparison = xai.compare_methods(model, obs, METHODS)
            
            for method_name, attr in comparison['attributions'].items():
                results.append({
                    'model': model.name,
                    'scenario_id': scenario['id'],
                    'method': method_name,
                    **attr.category_importance,
                    **comparison['sparsity'][method_name],
                })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'architecture_comparison.csv', index=False)
    
    return df
```

## 8.2 Attention Faithfulness Study

```python
# experiments/faithfulness_study.py

"""
Study faithfulness of attention weights as explanations.

For attention-based models (Perceiver, MTR, Wayformer):
- Extract attention weights
- Compare with gradient-based attributions
- Compute correlation and deletion/insertion curves
"""

import posthoc_xai as xai

def attention_faithfulness_experiment(model_path, scenarios):
    model = xai.load_model(model_path)
    
    if not model.has_attention:
        print(f"Model {model.name} has no attention")
        return None
    
    results = []
    
    for scenario in scenarios:
        obs = scenario['observation']
        
        # Get attention-based attribution
        attention = model.get_attention(obs)
        attention_attr = convert_attention_to_attribution(attention, model)
        
        # Get gradient-based attribution (ground truth)
        ig = xai.IntegratedGradients(model)
        ig_attr = ig(obs)
        
        # Correlation
        corr = xai.faithfulness.attention_gradient_correlation(attention_attr, ig_attr)
        
        # Deletion curves
        attn_del_pcts, attn_del_out = xai.faithfulness.deletion_curve(model, obs, attention_attr)
        ig_del_pcts, ig_del_out = xai.faithfulness.deletion_curve(model, obs, ig_attr)
        
        results.append({
            'scenario_id': scenario['id'],
            'correlation': corr['pearson_r'],
            'attention_deletion_auc': xai.faithfulness.area_under_deletion_curve(attn_del_out),
            'ig_deletion_auc': xai.faithfulness.area_under_deletion_curve(ig_del_out),
        })
    
    return results
```

---

# 9. Implementation Priority

## Phase 1: Core (Must Have)

1. **Model loader** - Load any V-Max model
2. **Perceiver wrapper** - First architecture to support
3. **Vanilla Gradient** - Simplest method
4. **Integrated Gradients** - Gold standard
5. **Category importance visualization** - See results quickly

## Phase 2: Extended Methods

6. **Perturbation** - Model-agnostic baseline
7. **SARFA** - RL-specific
8. **SmoothGrad** - Noise reduction
9. **Feature Ablation** - High-level analysis

## Phase 3: Additional Architectures

10. **MTR wrapper**
11. **Wayformer wrapper**
12. **MGAIL wrapper**
13. **MLP wrapper**

## Phase 4: Metrics & Experiments

14. **Faithfulness metrics** - Deletion/insertion curves
15. **Sparsity metrics** - Gini, entropy
16. **Consistency metrics** - Cross-scenario agreement
17. **Experiment scripts** - Architecture comparison, faithfulness study

## Phase 5: Polish

18. **BEV visualization** - Pretty figures
19. **Documentation** - Usage examples
20. **Tests** - Unit tests for each component

---

# 10. Notes for Claude Code

## Key Dependencies

```
jax
jaxlib
flax
numpy
scipy
matplotlib
pandas
pyyaml
```

## V-Max Integration

The framework needs to work with V-Max's:
- Observation format
- Network builders (for reconstructing models)
- Config structure

You'll need to import from V-Max:
```python
from vmax.networks import build_encoder, build_policy_head, build_value_head
from vmax.observation import ObservationConfig
```

## Testing Strategy

For each method:
1. Test on simple synthetic input first
2. Verify gradients are non-zero
3. Check attribution sums to expected value
4. Compare with known baselines

## Common Pitfalls

1. **JIT compilation with string arguments** - Use `static_argnums`
2. **Attention extraction** - May need to modify Flax modules
3. **Observation structure** - V-Max configs vary, parse carefully
4. **Memory with vmap** - Large batches can OOM, chunk if needed

---

*This document provides a complete specification for implementing the Post-Hoc XAI Framework. Follow the implementation priority order and test thoroughly at each phase.*
