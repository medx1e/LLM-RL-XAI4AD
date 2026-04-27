"""Main experiment: reward-conditioned attention analysis for one model.

Usage:
    cd /home/med1e/post-hoc-xai
    eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
    export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH

    # Quick test (1 scenario):
    python reward_attention/run_experiment.py --n-scenarios 1 --quick

    # Full run (50 scenarios):
    python reward_attention/run_experiment.py --n-scenarios 50

    # Custom:
    python reward_attention/run_experiment.py \\
        --model runs_rlc/womd_sac_road_perceiver_complete_42 \\
        --data data/training.tfrecord \\
        --n-scenarios 50 \\
        --output results/reward_attention
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Reward-conditioned attention experiment")
    parser.add_argument(
        "--model",
        default="runs_rlc/womd_sac_road_perceiver_complete_42",
        help="Path to model directory",
    )
    parser.add_argument(
        "--data",
        default="data/training.tfrecord",
        help="Path to tfrecord data",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=50,
        help="Number of scenarios to run",
    )
    parser.add_argument(
        "--output",
        default="results/reward_attention",
        help="Output directory root (model subdir appended automatically)",
    )
    parser.add_argument(
        "--attention-layer",
        default="avg",
        help="Which attention layer: 'avg' or '0'-'3'",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip temporal analysis and some figure types",
    )
    args = parser.parse_args()

    # Build output dir
    model_name = Path(args.model).name
    out_dir = Path(args.output) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    DIVIDER = "=" * 70
    print(DIVIDER)
    print(f"REWARD-CONDITIONED ATTENTION EXPERIMENT")
    print(f"  Model     : {args.model}")
    print(f"  Scenarios : {args.n_scenarios}")
    print(f"  Output    : {out_dir}")
    print(DIVIDER)

    # ----------------------------------------------------------------
    # 1. Load model
    # ----------------------------------------------------------------
    print("\n[1] Loading model ...")
    import posthoc_xai as xai
    model = xai.load_model(args.model, data_path=args.data)
    print(f"    Wrapper: {type(model).__name__}")
    print(f"    has_attention: {model.has_attention}")

    # ----------------------------------------------------------------
    # 2. Config
    # ----------------------------------------------------------------
    from reward_attention.config import AnalysisConfig
    config = AnalysisConfig(
        model_path=args.model,
        data_path=args.data,
        n_scenarios=args.n_scenarios,
        output_dir=str(out_dir),
        attention_layer=args.attention_layer,
    )

    # ----------------------------------------------------------------
    # 3. Collect data
    # ----------------------------------------------------------------
    print(f"\n[2] Collecting data from {args.n_scenarios} scenario(s) ...")
    from reward_attention.extractor import AttentionTimestepCollector

    collector = AttentionTimestepCollector(config)
    all_records = []

    t_start = time.time()
    for scenario_id in range(args.n_scenarios):
        t_s = time.time()
        try:
            scenario = next(model._loaded.data_gen)
            records = collector.collect(model, scenario, scenario_id=scenario_id)
            all_records.extend(records)
            elapsed = time.time() - t_s
            print(
                f"    Scenario {scenario_id:3d}: {len(records):3d} steps  "
                f"({elapsed:.1f}s)  total={len(all_records):,}"
            )
        except StopIteration:
            print(f"    No more scenarios after {scenario_id}.")
            break
        except Exception as e:
            print(f"    Scenario {scenario_id}: ERROR — {e}")
            continue

    total_elapsed = time.time() - t_start
    print(f"\n    Collected {len(all_records):,} timestep records in {total_elapsed:.1f}s")

    if len(all_records) == 0:
        print("    ERROR: No records collected. Check model and data path.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 4. Save raw records
    # ----------------------------------------------------------------
    pkl_path = out_dir / "timestep_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_records, f)
    print(f"\n[3] Saved {len(all_records):,} records to {pkl_path}")

    # ----------------------------------------------------------------
    # 5. Correlation analysis
    # ----------------------------------------------------------------
    print("\n[4] Running correlation analysis ...")
    from reward_attention.correlation import CorrelationAnalyzer, HYPOTHESIZED_PAIRS

    analyzer = CorrelationAnalyzer(all_records, config)
    print(f"    DataFrame: {analyzer.n_records:,} rows, {analyzer.n_scenarios} scenarios")

    # Per-scenario correlations (key for publication — avoids between-scenario confounds)
    per_scenario_summaries = analyzer.compute_all_per_scenario_summaries()
    per_scenario_summaries_hv = analyzer.compute_all_per_scenario_summaries(min_x_std=0.2)

    def _print_summaries(summaries, label):
        print(f"\n    {label}:")
        print(f"    {'Pair':<42s}  {'mean_ρ':>7s}  {'std_ρ':>6s}  {'95% CI':>18s}  {'n_sig%':>7s}  {'n_scen':>7s}  {'dir':>8s}")
        for s in summaries:
            if s.get("n_scenarios", 0) == 0:
                continue
            ci = f"[{s['ci_95_lower']:+.3f}, {s['ci_95_upper']:+.3f}]"
            pair = f"{s['variable_x']} × {s['variable_y']}"
            print(f"    {pair:<42s}  {s['mean_rho']:>+7.3f}  {s['std_rho']:>6.3f}  {ci:>18s}  {s['significant_pct']:>5.0f}%  {s['n_scenarios']:>7d}  {s['expected_direction']:>8s}")

    _print_summaries(per_scenario_summaries, "Within-scenario ρ (all scenarios)")
    _print_summaries(per_scenario_summaries_hv, "Within-scenario ρ (high-variation scenarios only, std(risk)>0.2)")

    # All hypothesized correlations (pooled)
    hyp_results = analyzer.compute_all_hypothesized(subgroups=["all", "high_risk"])
    print("\n    Pooled correlations (all scenarios combined):")
    for res in hyp_results:
        print(f"      {res.summary_line()}")

    # Full correlation matrix
    corr_matrix = analyzer.compute_full_correlation_matrix()
    print(f"\n    Full correlation matrix ({corr_matrix.shape}):")
    print(corr_matrix.to_string(float_format="{:+.3f}".format))

    # Action-conditioned attention
    action_attn = analyzer.compute_action_conditioned_attention()
    print("\n    Action-conditioned attention:")
    print(action_attn.to_string(float_format="{:.3f}".format))

    # ----------------------------------------------------------------
    # 6. Save results
    # ----------------------------------------------------------------
    results_dict = {
        "model": args.model,
        "n_scenarios": args.n_scenarios,
        "n_records": len(all_records),
        "per_scenario_summaries_all": per_scenario_summaries,
        "per_scenario_summaries_high_variation": per_scenario_summaries_hv,
        "hypothesized_correlations": [r.to_dict() for r in hyp_results],
        "correlation_matrix": corr_matrix.to_dict(),
        "action_conditioned_attention": action_attn.to_dict(),
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[5] Saved results.json to {results_path}")

    # Summary CSV
    csv_path = out_dir / "summary_table.csv"
    summary_rows = [r.to_dict() for r in hyp_results if r.subgroup == "all"]
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"    Saved summary_table.csv to {csv_path}")

    # ----------------------------------------------------------------
    # 7. Figures
    # ----------------------------------------------------------------
    print("\n[6] Generating figures ...")
    from reward_attention.visualization import (
        plot_scatter,
        plot_correlation_heatmap,
        plot_temporal_event,
        plot_action_conditioned,
    )

    df = analyzer.df

    # Fig 1: Key scatter plots
    scatter_pairs = [
        ("collision_risk", "attn_agents"),
        ("safety_risk",    "attn_agents"),
        ("navigation_risk","attn_gps"),
        ("collision_risk", "attn_to_threat"),
    ]
    for risk_col, attn_col in scatter_pairs:
        fname = out_dir / f"fig1_scatter_{risk_col}_vs_{attn_col}.png"
        try:
            plot_scatter(df, risk_col, attn_col, save_path=fname)
            print(f"    Saved {fname.name}")
        except Exception as e:
            print(f"    fig1 {risk_col}×{attn_col}: ERROR — {e}")

    # Fig 2: Correlation heatmap
    try:
        heatmap_path = out_dir / "fig2_correlation_heatmap.png"
        plot_correlation_heatmap(corr_matrix, save_path=heatmap_path)
        print(f"    Saved {heatmap_path.name}")
    except Exception as e:
        print(f"    fig2 heatmap: ERROR — {e}")

    # Fig 3: Temporal event plots (skip if --quick)
    if not args.quick:
        from reward_attention.temporal import TemporalAnalyzer
        print("\n[7] Running temporal analysis ...")
        temporal = TemporalAnalyzer(all_records)
        summary_temporal = temporal.summary()
        print(f"    Temporal summary: {summary_temporal}")

        trajectories = temporal.analyze_all_events(
            event_types=["hazard_onset", "collision_imminent"]
        )
        print(f"    Matched {len(trajectories)} event trajectories")

        for key, traj_df in list(trajectories.items())[:5]:  # max 5 plots
            try:
                event_peak = int(traj_df[traj_df["event_phase"] == "peak"]["timestep"].iloc[0])
            except (IndexError, KeyError):
                event_peak = 0
            fname = out_dir / f"fig3_temporal_{key}.png"
            try:
                plot_temporal_event(traj_df, event_peak=event_peak,
                                    title=f"Event: {key}", save_path=fname)
                print(f"    Saved {fname.name}")
            except Exception as e:
                print(f"    fig3 {key}: ERROR — {e}")

        results_dict["temporal_summary"] = summary_temporal
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2)

    # Fig 4: Action-conditioned attention
    try:
        action_path = out_dir / "fig4_action_attention.png"
        plot_action_conditioned(action_attn, save_path=action_path)
        print(f"    Saved {action_path.name}")
    except Exception as e:
        print(f"    fig4 action: ERROR — {e}")

    # ----------------------------------------------------------------
    # 8. Print summary
    # ----------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("EXPERIMENT SUMMARY")
    print(DIVIDER)
    print(f"  Records collected : {len(all_records):,}")
    print(f"  Scenarios run     : {analyzer.n_scenarios}")
    print(f"\n  Within-scenario mean ρ — high-variation scenarios (KEY result):")
    for s in per_scenario_summaries_hv:
        if s.get("n_scenarios", 0) == 0:
            continue
        ci = f"[{s['ci_95_lower']:+.3f}, {s['ci_95_upper']:+.3f}]"
        sig = "**" if s["significant_pct"] == 100 else ("*" if s["significant_pct"] >= 50 else "")
        print(f"    {s['variable_x']:18s} × {s['variable_y']:16s}  ρ={s['mean_rho']:+.3f}{sig}  95%CI={ci}  sig={s['significant_pct']:.0f}%  n={s['n_scenarios']}  [{s['expected_direction']}]")

    print(f"\n  Pooled correlations (all timesteps combined, Spearman ρ):")
    for res in hyp_results:
        if res.subgroup == "all":
            sig = "**" if res.spearman_p < 0.01 else ("*" if res.spearman_p < 0.05 else "")
            print(f"    {res.variable_x:18s} × {res.variable_y:16s}  ρ={res.spearman_rho:+.3f}{sig}  p={res.spearman_p:.4f}")
    print(f"\n  Output files in: {out_dir}")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    main()
