"""Causal steering: modify a single SAE feature and measure behavioral change.

Intervention protocol for feature j at temperature α:
    h  (residual stream)
    ↓  SAE encode
    f  = ReLU(h @ W_enc + b_enc)
    ↓  modify
    f' = f.at[j].add(α)
    ↓  SAE decode
    h_steered = f' @ W_dec + b_dec
    ↓  apply delta in residual stream space
    h_final = h + (h_steered - h_reconstructed)
    ↓  policy / value FC heads
    Δvalue, Δaccel, Δsteer  ← behavioral effect

Outputs per temperature step:
  • delta_accel  – shift in acceleration action dimension
  • delta_steer  – shift in steering action dimension
  • policy_base / policy_steered – full raw action vectors for audit
  • delta_value  – mean shift in value estimate

Aggregation over scenarios:
  • mean / std / median / p5 / p95 of Δvalue distribution
  • sign_flip_frac: fraction of scenarios where Δvalue changes sign
    (a feature with mean≈0 but high sign_flip_frac is still important)
  • Histogram PNGs saved alongside the JSON output.

A large Δvalue that scales monotonically with α confirms that feature j
causally drives the agent's value estimate.

Usage:
    python -m sae_interpretability.causal_steering \\
        --run_dir ../../runs/PPO_VEC_WAYFORMER \\
        --dataset ../../training.tfrecord \\
        --sae_checkpoint sae_model.pt \\
        --feature_idx 42 \\
        --temperatures -5.0 -2.0 -1.0 1.0 2.0 5.0
"""

from __future__ import annotations
from sae_interpretability.sae_model import SparseAutoencoder
from sae_interpretability.config import SAEConfig
from xai.attention_analysis.offline_extraction import OfflineExtractor

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class CausalSteerer(OfflineExtractor):
    """Apply causal interventions on SAE features and measure behavioral change.

    Inherits env setup, model loading, and data iteration from OfflineExtractor.
    Adds SAE weight loading and a steering-specific forward pass.
    """

    def __init__(
        self,
        run_dir: str,
        dataset_path: str,
        sae_checkpoint: str,
        cfg: SAEConfig,
        checkpoint_name: str = "model_final.pkl",
    ):
        super().__init__(run_dir, dataset_path, checkpoint_name)
        self.sae_checkpoint = sae_checkpoint
        self.sae_cfg = cfg
        self._sae_w: Optional[Dict[str, jnp.ndarray]] = None
        self._act_mean_jnp: Optional[jnp.ndarray] = None
        self._act_std_jnp:  Optional[jnp.ndarray] = None
        self._policy_fc_params = None
        self._policy_dense_params = None
        self._value_fc_params = None
        self._value_dense_params = None

    def setup(self) -> CausalSteerer:
        super().setup()
        self._load_sae_weights()
        self._extract_head_params()
        return self

    def _load_sae_weights(self):
        """Load the trained SAE checkpoint and convert weights to JAX arrays."""
        print(f"[Steerer] Loading SAE from {self.sae_checkpoint}")
        ckpt = torch.load(self.sae_checkpoint,
                          map_location='cpu', weights_only=False)
        sae = SparseAutoencoder(
            input_dim=ckpt['config']['input_dim'],
            latent_dim=ckpt['config']['latent_dim'],
            l1_coeff=ckpt['config']['l1_coeff'],
        )
        sae.load_state_dict(ckpt['model_state_dict'])
        np_w = sae.to_numpy_weights()
        self._sae_w = {k: jnp.array(v) for k, v in np_w.items()}

        input_dim = ckpt['config']['input_dim']
        act_mean_np = ckpt.get('act_mean', np.zeros(input_dim, dtype=np.float32))
        act_std_np  = ckpt.get('act_std',  np.ones(input_dim,  dtype=np.float32))
        self._act_mean_jnp = jnp.array(act_mean_np.squeeze())
        self._act_std_jnp  = jnp.array(act_std_np.squeeze())
        print(
            f"[Steerer] SAE ready: "
            f"{ckpt['config']['input_dim']}D → {ckpt['config']['latent_dim']}D"
        )

    def _extract_head_params(self):
        """Pull policy and value FC-head params out of the loaded checkpoint."""
        policy_p = self.params.policy.get('params', self.params.policy)
        self._policy_fc_params = policy_p.get('fully_connected_layer')
        self._policy_dense_params = policy_p.get('Dense_0')

        if hasattr(self.params, 'value'):
            value_p = self.params.value.get('params', self.params.value)
            self._value_fc_params = value_p.get('fully_connected_layer')
            self._value_dense_params = value_p.get('Dense_0')

    # ------------------------------------------------------------------
    # SAE encode / decode  (pure JAX, JIT-able)
    # ------------------------------------------------------------------

    def _sae_encode(self, h: jnp.ndarray) -> jnp.ndarray:
        w = self._sae_w
        h_norm = (h - self._act_mean_jnp) / self._act_std_jnp
        return jnp.maximum((h_norm - w['pre_bias']) @ w['W_enc'] + w['b_enc'], 0.0)

    def _sae_decode(self, f: jnp.ndarray) -> jnp.ndarray:
        w = self._sae_w
        return f @ w['W_dec'] + w['b_dec']

    # ------------------------------------------------------------------
    # FC head application
    # ------------------------------------------------------------------

    def _apply_fc_head(
        self,
        h: jnp.ndarray,
        fc_params: Optional[Dict],
        dense_params: Optional[Dict],
    ) -> Optional[jnp.ndarray]:
        """Apply MLP + final Dense layer to a latent vector.

        Returns None when parameters are unavailable.
        """
        if fc_params is None or dense_params is None:
            return None
        x = h
        try:
            for key in sorted(fc_params.keys()):
                layer = fc_params[key]
                if 'kernel' in layer and 'bias' in layer:
                    x = jnp.maximum(x @ layer['kernel'] + layer['bias'], 0.0)
            x = x @ dense_params['kernel'] + dense_params['bias']
        except Exception as e:
            print(f"[Steerer] FC head error: {e}")
            return None
        return x

    # ------------------------------------------------------------------
    # Single-observation steering
    # ------------------------------------------------------------------

    def steer_observation(
        self,
        obs: jnp.ndarray,
        feature_idx: int,
        temperatures: List[float],
    ) -> Dict[str, Any]:
        """Measure the causal effect of steering feature j on a single observation.

        Args:
            obs: Vectorized observation [obs_dim] or [1, obs_dim].
            feature_idx: SAE feature index to modify.
            temperatures: List of α values to add to feature j.

        Returns:
            Dict containing baseline activation and per-temperature behavioral shift.
        """
        if obs.ndim == 1:
            obs = obs[None]  # → [1, obs_dim]

        h, _ = self.encoder.apply({'params': self.encoder_params}, obs)
        if h.ndim > 1:
            h = h[0]  # [D]

        f_baseline = self._sae_encode(h)
        h_reconstructed = self._sae_decode(f_baseline)  # baseline SAE output

        h_batch = h[None]  # [1, D] for FC heads
        policy_base = self._apply_fc_head(
            h_batch, self._policy_fc_params, self._policy_dense_params)
        value_base = self._apply_fc_head(
            h_batch, self._value_fc_params, self._value_dense_params)

        # ── Collect & print baseline values ──────────────────────────────
        pb_raw: Optional[np.ndarray] = None
        vb_scalar: Optional[float] = None
        if policy_base is not None:
            pb_raw = np.array(policy_base).ravel()
            accel_b = float(pb_raw[0]) if len(pb_raw) > 0 else float('nan')
            steer_b = float(pb_raw[1]) if len(pb_raw) > 1 else float('nan')
            print(
                f"    [baseline] accel={accel_b:+.4f}  steer={steer_b:+.4f}  "
                f"(full policy={pb_raw.tolist()})"
            )
        if value_base is not None:
            vb_scalar = float(jnp.mean(value_base))
            print(f"    [baseline] value_estimate={vb_scalar:+.4f}")

        result: Dict[str, Any] = {
            'feature_idx': feature_idx,
            'baseline_activation': float(f_baseline[feature_idx]),
            'baseline_policy': pb_raw.tolist() if pb_raw is not None else None,
            'baseline_value':  vb_scalar,
            'temperatures': [],
        }

        for alpha in temperatures:
            f_steered = f_baseline.at[feature_idx].add(alpha)
            h_steered = self._sae_decode(f_steered)

            # Delta is in normalized space; scale back to raw residual stream space
            delta = (h_steered - h_reconstructed) * self._act_std_jnp
            assert float(jnp.linalg.norm(delta)) > 0, (
                f"Steering had no effect on h for alpha={alpha}: delta norm is zero. "
                f"Feature {feature_idx} decoder direction may be null."
            )
            h_final = (h + delta)[None]  # [1, D]

            entry: Dict[str, Any] = {
                'alpha': alpha,
                'steered_activation': float(f_steered[feature_idx]),
                'h_delta_norm': float(jnp.linalg.norm(h_steered - h_reconstructed)),
            }

            policy_steered = self._apply_fc_head(
                h_final, self._policy_fc_params, self._policy_dense_params
            )
            value_steered = self._apply_fc_head(
                h_final, self._value_fc_params, self._value_dense_params
            )

            # ── Policy decomposition ──────────────────────────────────────
            if policy_base is not None and policy_steered is not None:
                pb = np.array(policy_base).ravel()
                ps = np.array(policy_steered).ravel()
                delta_policy = ps - pb
                entry['policy_base']    = pb.tolist()
                entry['policy_steered'] = ps.tolist()
                entry['delta_accel'] = float(delta_policy[0]) if len(delta_policy) > 0 else None
                entry['delta_steer'] = float(delta_policy[1]) if len(delta_policy) > 1 else None
                entry['delta_policy_per_dim'] = delta_policy.tolist()

                accel_s = float(ps[0]) if len(ps) > 0 else float('nan')
                steer_s = float(ps[1]) if len(ps) > 1 else float('nan')
                print(
                    f"      α={alpha:+.2f}  accel: {accel_b:+.4f}→{accel_s:+.4f} "
                    f"(Δ{delta_policy[0]:+.4f})  "
                    f"steer: {steer_b:+.4f}→{steer_s:+.4f} "
                    f"(Δ{delta_policy[1]:+.4f})"
                )

            # ── Value shift ───────────────────────────────────────────────
            if value_base is not None and value_steered is not None:
                vs_scalar = float(jnp.mean(value_steered))
                dv = vs_scalar - (vb_scalar or 0.0)
                entry['value_base']    = vb_scalar
                entry['value_steered'] = vs_scalar
                entry['delta_value']   = dv
                print(
                    f"             value: {vb_scalar:+.4f}→{vs_scalar:+.4f} "
                    f"(Δ{dv:+.4f})"
                )

            result['temperatures'].append(entry)

        return result

    # ------------------------------------------------------------------
    # Multi-scenario experiment
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        feature_idx: int,
        temperatures: List[float],
        n_scenarios: int = 10,
        output_path: str = "steering_results.json",
    ) -> Dict[str, Any]:
        """Run the steering experiment across multiple scenarios and aggregate.

        Args:
            feature_idx: SAE feature to steer.
            temperatures: α values to try.
            n_scenarios: How many scenarios to evaluate.
            output_path: Where to write the JSON results.

        Returns:
            Dict with per-scenario results and aggregated statistics.
        """
        from vmax.simulator import make_data_generator, datasets

        data_gen = make_data_generator(
            path=datasets.get_dataset(self.dataset_path),
            max_num_objects=self.config["max_num_objects"],
            include_sdc_paths=not self.config.get("waymo_dataset", False),
            batch_dims=(1,),
            seed=42,
            repeat=1,
        )

        all_results: List[Dict] = []
        print(
            f"[Steerer] Feature {feature_idx}  α={temperatures}  "
            f"{n_scenarios} scenarios"
        )

        for i, scenario_batch in enumerate(data_gen):
            if i >= n_scenarios:
                break
            env_transition = self.env.reset(scenario_batch)
            obs = env_transition.observation

            result = self.steer_observation(obs, feature_idx, temperatures)
            result['scenario_id'] = i
            all_results.append(result)

            baseline = result['baseline_activation']
            print(f"  Scenario {i+1:3d}: baseline_activation={baseline:.4f}")

        # ── Aggregate Δvalue distribution per temperature ─────────────────
        n_temps = len(temperatures)
        n_scenarios_ran = len(all_results)
        delta_matrix = np.zeros((n_scenarios_ran, n_temps))
        for si, res in enumerate(all_results):
            for ti, entry in enumerate(res['temperatures']):
                delta_matrix[si, ti] = entry.get('delta_value', 0.0)

        # ── Aggregate per-action-dimension policy shifts ───────────────────
        accel_matrix = np.zeros((n_scenarios_ran, n_temps))
        steer_matrix = np.zeros((n_scenarios_ran, n_temps))
        for si, res in enumerate(all_results):
            for ti, entry in enumerate(res['temperatures']):
                accel_matrix[si, ti] = entry.get('delta_accel') or 0.0
                steer_matrix[si, ti] = entry.get('delta_steer') or 0.0

        def _dist_stats(mat: np.ndarray) -> Dict:
            """Return rich distribution stats for a (scenarios × temps) matrix."""
            sign_pos = (mat > 0).sum(axis=0)
            sign_neg = (mat < 0).sum(axis=0)
            sign_flip = np.minimum(sign_pos, sign_neg) / max(n_scenarios_ran, 1)
            return {
                'mean':           mat.mean(axis=0).tolist(),
                'std':            mat.std(axis=0).tolist(),
                'median':         np.median(mat, axis=0).tolist(),
                'p5':             np.percentile(mat, 5,  axis=0).tolist(),
                'p95':            np.percentile(mat, 95, axis=0).tolist(),
                'sign_flip_frac': sign_flip.tolist(),
            }

        summary = {
            'feature_idx':  feature_idx,
            'temperatures': temperatures,
            'n_scenarios':  n_scenarios_ran,
            'mean_baseline_activation': float(
                np.mean([r['baseline_activation'] for r in all_results])
            ),
            'delta_value':  _dist_stats(delta_matrix),
            'delta_accel':  _dist_stats(accel_matrix),
            'delta_steer':  _dist_stats(steer_matrix),
        }

        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(
                {'summary': summary, 'per_scenario': all_results}, f, indent=2)

        print(f"\n[Steerer] Results → {output_path}")
        print("  Mean Δvalue / Δaccel / Δsteer per α:")
        for ti, alpha in enumerate(temperatures):
            dv = summary['delta_value']['mean'][ti]
            da = summary['delta_accel']['mean'][ti]
            ds = summary['delta_steer']['mean'][ti]
            sf = summary['delta_value']['sign_flip_frac'][ti]
            sign = '+' if dv >= 0 else ''
            bar  = '▮' * int(abs(dv) * 20)
            print(
                f"    α={alpha:+5.1f}:  Δval={sign}{dv:.4f}  "
                f"Δaccel={da:+.4f}  Δsteer={ds:+.4f}  "
                f"sign_flip={sf:.0%}  {bar}"
            )

        # ── Histogram plots ────────────────────────────────────────────────
        plot_histograms(
            delta_matrix, accel_matrix, steer_matrix,
            temperatures, feature_idx, out_dir,
        )

        return {'summary': summary, 'per_scenario': all_results}


# ---------------------------------------------------------------------------
# Histogram visualisation
# ---------------------------------------------------------------------------

def plot_histograms(
    delta_value:  np.ndarray,
    delta_accel:  np.ndarray,
    delta_steer:  np.ndarray,
    temperatures: List[float],
    feature_idx:  int,
    out_dir:      Path,
) -> None:
    """Save histogram PNG files showing the distribution of Δvalue, Δaccel, Δsteer
    across scenarios for every temperature.

    One PNG per metric is emitted so figures stay readable regardless of the
    number of temperature levels.

    Args:
        delta_value:  (n_scenarios, n_temps) Δvalue matrix.
        delta_accel:  (n_scenarios, n_temps) Δaccel matrix.
        delta_steer:  (n_scenarios, n_temps) Δsteer matrix.
        temperatures: List of α values (used for labels).
        feature_idx:  SAE feature index being steered (used in titles).
        out_dir:      Directory where PNG files are written.
    """
    n_temps = len(temperatures)
    metrics = [
        ('delta_value', delta_value,  'Δvalue',  '#4C72B0'),
        ('delta_accel', delta_accel,  'Δaccel',  '#DD8452'),
        ('delta_steer', delta_steer,  'Δsteer',  '#55A868'),
    ]

    for metric_name, mat, label, color in metrics:
        fig, axes = plt.subplots(
            1, n_temps,
            figsize=(4 * n_temps, 4),
            sharey=False,
        )
        if n_temps == 1:
            axes = [axes]

        for ti, (ax, alpha) in enumerate(zip(axes, temperatures)):
            vals = mat[:, ti]
            ax.hist(vals, bins=min(20, len(vals)), color=color,
                    edgecolor='white', alpha=0.85)
            ax.axvline(0, color='crimson', linewidth=1.2, linestyle='--')
            mean_v = float(vals.mean())
            ax.axvline(mean_v, color='gold', linewidth=1.2,
                       linestyle='-', label=f'mean={mean_v:+.3f}')
            ax.set_title(f'α={alpha:+.1f}', fontsize=10)
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel('# scenarios', fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=8)

        fig.suptitle(
            f'Feature {feature_idx} — {label} distribution across scenarios',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout()
        out_path = out_dir / f'f{feature_idx}_{metric_name}_hist.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[Steerer] Histogram → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Causal steering experiment on SAE features.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sae_checkpoint", required=True)
    parser.add_argument("--feature_idx", type=int, required=True)
    parser.add_argument("--temperatures", type=float,
                        nargs="+", default=[-5.0, -2.0, -1.0, 1.0, 2.0, 5.0])
    parser.add_argument("--n_scenarios", type=int, default=10)
    parser.add_argument("--output", default="data/sae_interpretability/steering_results.json")
    parser.add_argument("--checkpoint", default="model_final.pkl")
    args = parser.parse_args()

    cfg = SAEConfig()
    steerer = CausalSteerer(
        args.run_dir, args.dataset, args.sae_checkpoint, cfg, args.checkpoint
    )
    steerer.setup()
    steerer.run_experiment(
        args.feature_idx, args.temperatures, args.n_scenarios, args.output
    )


if __name__ == "__main__":
    main()
