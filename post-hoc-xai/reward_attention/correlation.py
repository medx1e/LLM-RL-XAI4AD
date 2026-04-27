"""CorrelationAnalyzer: Pearson + Spearman between risk metrics and attention categories.

Hypothesis table (from task_reward_attention.md):
  collision_risk  ↔ attn_agents    : positive (TTC risk → look at agents)
  collision_risk  ↔ attn_to_threat : positive (look at the threatening agent)
  safety_risk     ↔ attn_agents    : positive (safety risk driven by vehicle proximity)
  navigation_risk ↔ attn_gps      : positive (off-route → more path attention)
  behavior_risk   ↔ attn_roadgraph : unclear
  collision_risk  ↔ attn_roadgraph : negative (when focused on agents, less road attention)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from reward_attention.config import TimestepRecord, AnalysisConfig


# ---------------------------------------------------------------------------
# CorrelationResult
# ---------------------------------------------------------------------------


@dataclass
class CorrelationResult:
    variable_x: str
    variable_y: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n_samples: int
    subgroup: str = "all"

    def to_dict(self) -> dict:
        return {
            "variable_x": self.variable_x,
            "variable_y": self.variable_y,
            "pearson_r": round(self.pearson_r, 4),
            "pearson_p": round(self.pearson_p, 4),
            "spearman_rho": round(self.spearman_rho, 4),
            "spearman_p": round(self.spearman_p, 4),
            "n_samples": self.n_samples,
            "subgroup": self.subgroup,
            "significant_05": self.spearman_p < 0.05,
        }

    def summary_line(self) -> str:
        sig = "**" if self.spearman_p < 0.01 else ("*" if self.spearman_p < 0.05 else "")
        return (
            f"{self.variable_x:20s} × {self.variable_y:20s}  "
            f"ρ={self.spearman_rho:+.3f}{sig}  r={self.pearson_r:+.3f}  "
            f"p={self.spearman_p:.4f}  n={self.n_samples}  [{self.subgroup}]"
        )


# ---------------------------------------------------------------------------
# Hypothesized pairs
# ---------------------------------------------------------------------------

HYPOTHESIZED_PAIRS: list[tuple[str, str, str]] = [
    # (risk_col, attention_col, expected_direction)
    ("collision_risk",  "attn_agents",    "positive"),
    ("collision_risk",  "attn_to_threat", "positive"),
    ("safety_risk",     "attn_agents",    "positive"),
    ("navigation_risk", "attn_gps",       "positive"),
    ("behavior_risk",   "attn_roadgraph", "unclear"),
    ("collision_risk",  "attn_roadgraph", "negative"),
]

RISK_COLS = ["collision_risk", "safety_risk", "navigation_risk", "behavior_risk"]
ATTN_COLS = ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps",
             "attn_to_nearest", "attn_to_threat"]


# ---------------------------------------------------------------------------
# CorrelationAnalyzer
# ---------------------------------------------------------------------------


class CorrelationAnalyzer:
    """Compute correlations between risk metrics and attention categories.

    Args:
        records: List of TimestepRecord from AttentionTimestepCollector.
        config: AnalysisConfig with subgroup thresholds.
    """

    def __init__(self, records: List[TimestepRecord], config: AnalysisConfig):
        self.config = config
        self.df = self._build_dataframe(records)

    @classmethod
    def from_records(cls, records: List[TimestepRecord], config: AnalysisConfig) -> "CorrelationAnalyzer":
        return cls(records, config)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_correlation(
        self,
        var_x: str,
        var_y: str,
        subgroup: str = "all",
    ) -> Optional[CorrelationResult]:
        """Compute Pearson + Spearman for one (x, y) pair.

        Args:
            var_x: Column name of x variable (risk metric).
            var_y: Column name of y variable (attention).
            subgroup: 'all', 'high_risk', 'collision_steps', 'braking'.

        Returns:
            CorrelationResult or None if insufficient data.
        """
        df = self._filter_subgroup(subgroup)
        if len(df) < 10:
            return None

        x = df[var_x].values
        y = df[var_y].values

        # Remove NaN/inf
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < 10:
            return None

        # Skip if either variable is constant (correlation undefined)
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return None

        try:
            with np.errstate(invalid="ignore"):
                pearson_r, pearson_p = stats.pearsonr(x, y)
                spearman_rho, spearman_p = stats.spearmanr(x, y)
            if not (np.isfinite(pearson_r) and np.isfinite(spearman_rho)):
                return None
        except Exception:
            return None

        return CorrelationResult(
            variable_x=var_x,
            variable_y=var_y,
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            spearman_rho=float(spearman_rho),
            spearman_p=float(spearman_p),
            n_samples=int(len(x)),
            subgroup=subgroup,
        )

    def compute_all_hypothesized(
        self, subgroups: list[str] | None = None
    ) -> list[CorrelationResult]:
        """Run all hypothesized (risk, attention) pairs across requested subgroups."""
        if subgroups is None:
            subgroups = ["all", "high_risk"]

        results = []
        for risk_col, attn_col, direction in HYPOTHESIZED_PAIRS:
            for sg in subgroups:
                res = self.compute_correlation(risk_col, attn_col, subgroup=sg)
                if res is not None:
                    results.append(res)
        return results

    def compute_per_scenario_correlations(
        self,
        var_x: str,
        var_y: str,
        min_x_std: float = 0.0,
    ) -> pd.DataFrame:
        """Compute within-scenario Spearman ρ for each scenario.

        This is the key analysis for publication: report mean ± std of within-
        scenario correlations, which avoids between-scenario confounds.

        Returns:
            DataFrame with columns: scenario_id, spearman_rho, pearson_r, n.
        """
        rows = []
        for sid, sub in self.df.groupby("scenario_id"):
            x = sub[var_x].values
            y = sub[var_y].values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
                continue
            # Optional: filter to scenarios with sufficient X variation
            if min_x_std > 0 and np.std(x) < min_x_std:
                continue
            with np.errstate(invalid="ignore"):
                rho, p = stats.spearmanr(x, y)
                r, _ = stats.pearsonr(x, y)
            if np.isfinite(rho):
                rows.append({
                    "scenario_id": sid,
                    "x_std": float(np.std(x)),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "pearson_r": float(r),
                    "n": int(len(x)),
                })
        return pd.DataFrame(rows)

    def compute_per_scenario_summary(
        self,
        var_x: str,
        var_y: str,
        min_x_std: float = 0.0,
    ) -> dict:
        """Summary of within-scenario correlations: mean ± std ρ.

        Uses Fisher z-transformation for correct averaging of correlation
        coefficients across scenarios.

        Returns:
            Dict with mean_rho, std_rho, ci_95, n_scenarios, direction, significant_pct.
        """
        df = self.compute_per_scenario_correlations(var_x, var_y, min_x_std=min_x_std)
        if len(df) == 0:
            return {"n_scenarios": 0, "mean_rho": np.nan, "min_x_std_filter": min_x_std}

        rhos = df["spearman_rho"].values
        # Fisher z-transform for averaging
        z = np.arctanh(np.clip(rhos, -0.999, 0.999))
        mean_z = np.mean(z)
        se_z = np.std(z) / np.sqrt(len(z))
        mean_rho = float(np.tanh(mean_z))
        ci_lower = float(np.tanh(mean_z - 1.96 * se_z))
        ci_upper = float(np.tanh(mean_z + 1.96 * se_z))
        sig_pct = 100.0 * np.mean(df["spearman_p"] < 0.05)

        return {
            "variable_x": var_x,
            "variable_y": var_y,
            "mean_rho": round(mean_rho, 4),
            "std_rho": round(float(np.std(rhos)), 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "n_scenarios": int(len(df)),
            "significant_pct": round(sig_pct, 1),
            "mean_n_per_scenario": round(float(df["n"].mean()), 1),
        }

    def compute_all_per_scenario_summaries(
        self,
        min_x_std: float = 0.0,
    ) -> list[dict]:
        """Compute within-scenario summaries for all hypothesized pairs.

        Args:
            min_x_std: Minimum std of X variable to include a scenario.
                       0.2 is a useful threshold for filtering near-constant-risk scenarios.
        """
        results = []
        for risk_col, attn_col, direction in HYPOTHESIZED_PAIRS:
            summary = self.compute_per_scenario_summary(risk_col, attn_col, min_x_std=min_x_std)
            summary["expected_direction"] = direction
            results.append(summary)
        return results

    def compute_full_correlation_matrix(self) -> pd.DataFrame:
        """Compute full matrix: all risk_cols × all attn_cols.

        Returns:
            DataFrame indexed by risk_col, columns by attn_col,
            values = Spearman rho (NaN if insufficient data).
        """
        matrix = pd.DataFrame(index=RISK_COLS, columns=ATTN_COLS, dtype=float)
        for risk_col in RISK_COLS:
            for attn_col in ATTN_COLS:
                res = self.compute_correlation(risk_col, attn_col, subgroup="all")
                matrix.loc[risk_col, attn_col] = res.spearman_rho if res else np.nan
        return matrix

    def compute_action_conditioned_attention(self) -> pd.DataFrame:
        """Mean attention per category, conditioned on action type.

        Action categories:
            braking   : ego_accel < -braking_threshold
            steering  : |steering| > steering_threshold (and not braking)
            neutral   : otherwise

        Returns:
            DataFrame with action_type as index, attn cols as columns.
        """
        df = self.df.copy()

        braking_thresh = self.config.braking_threshold
        steering_thresh = self.config.steering_threshold

        braking_mask = df["accel"] < -braking_thresh
        steering_mask = (df["accel"].abs() <= braking_thresh) & (
            df["steering"].abs() > steering_thresh
        )
        neutral_mask = ~braking_mask & ~steering_mask

        rows = []
        for label, mask in [("braking", braking_mask), ("steering", steering_mask), ("neutral", neutral_mask)]:
            sub = df[mask]
            row = {"action_type": label, "n": int(len(sub))}
            for col in ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]:
                row[col] = float(sub[col].mean()) if len(sub) > 0 else np.nan
            rows.append(row)

        return pd.DataFrame(rows).set_index("action_type")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_records(self) -> int:
        return len(self.df)

    @property
    def n_scenarios(self) -> int:
        return self.df["scenario_id"].nunique()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataframe(records: List[TimestepRecord]) -> pd.DataFrame:
        """Convert list of TimestepRecord to DataFrame."""
        rows = []
        for r in records:
            row = {
                "scenario_id":    r.scenario_id,
                "timestep":       r.timestep,
                "attn_sdc":       r.attn_sdc,
                "attn_agents":    r.attn_agents,
                "attn_roadgraph": r.attn_roadgraph,
                "attn_lights":    r.attn_lights,
                "attn_gps":       r.attn_gps,
                "attn_to_nearest": r.attn_to_nearest,
                "attn_to_threat":  r.attn_to_threat,
                "collision_risk":  r.collision_risk,
                "safety_risk":     r.safety_risk,
                "navigation_risk": r.navigation_risk,
                "behavior_risk":   r.behavior_risk,
                "min_ttc":         r.min_ttc,
                "accel":           r.accel,
                "steering":        r.steering,
                "ego_speed":       r.ego_speed,
                "num_valid_agents": r.num_valid_agents,
                "is_collision_step": r.is_collision_step,
                "is_offroad_step":   r.is_offroad_step,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def _filter_subgroup(self, subgroup: str) -> pd.DataFrame:
        df = self.df
        if subgroup == "all":
            return df
        elif subgroup == "high_risk":
            return df[df["safety_risk"] >= self.config.high_risk_threshold]
        elif subgroup == "collision_steps":
            return df[df["is_collision_step"]]
        elif subgroup == "braking":
            return df[df["accel"] < -self.config.braking_threshold]
        else:
            return df
