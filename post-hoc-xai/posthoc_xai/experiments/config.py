"""Experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from posthoc_xai.methods import METHOD_REGISTRY


ALL_METHODS = list(METHOD_REGISTRY.keys())

# Fast methods suitable for quick testing
QUICK_METHODS = ["vanilla_gradient", "gradient_x_input", "feature_ablation"]


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run.

    Controls scanning, selection, analysis, and output settings.
    """

    # Model
    model_dir: str = ""
    data_path: str = "data/training.tfrecord"

    # Scanning
    n_scenarios: int = 50

    # Selection filters (scenarios must pass ALL active filters)
    min_valid_agents: int = 0
    require_traffic_lights: bool = False
    include_failures: bool = True
    max_selected: int = 15

    # Analysis
    methods: list[str] = field(default_factory=lambda: list(ALL_METHODS))
    timestep_strategy: str = "key_moments"  # "key_moments" | "fixed_interval" | "all"
    timestep_interval: int = 20
    target_action: int | None = None
    compute_faithfulness: bool = True
    faithfulness_steps: int = 10
    save_raw_arrays: bool = False
    save_plots: bool = True

    # Output
    output_dir: str = "results"
    experiment_name: str | None = None

    # Derived (computed at runtime via __post_init__)
    model_name: str = ""

    def __post_init__(self) -> None:
        if self.model_dir:
            self.model_name = Path(self.model_dir).name
        if self.experiment_name is None:
            self.experiment_name = "default"

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def quick(cls, model_dir: str, **overrides) -> ExperimentConfig:
        """Quick preset: 10 scenarios, 3 fast methods, no faithfulness."""
        defaults = dict(
            model_dir=model_dir,
            n_scenarios=10,
            max_selected=5,
            methods=list(QUICK_METHODS),
            timestep_strategy="key_moments",
            compute_faithfulness=False,
            save_raw_arrays=False,
            save_plots=True,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def standard(cls, model_dir: str, **overrides) -> ExperimentConfig:
        """Standard preset: 50 scenarios, all methods, faithfulness on."""
        defaults = dict(
            model_dir=model_dir,
            n_scenarios=50,
            max_selected=15,
            methods=list(ALL_METHODS),
            timestep_strategy="key_moments",
            compute_faithfulness=True,
            save_raw_arrays=False,
            save_plots=True,
        )
        defaults.update(overrides)
        return cls(**defaults)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def results_dir(self) -> str:
        """Base results directory for this experiment."""
        return os.path.join(self.output_dir, self.experiment_name)

    @property
    def model_results_dir(self) -> str:
        """Model-specific results directory."""
        return os.path.join(self.results_dir, self.model_name)
