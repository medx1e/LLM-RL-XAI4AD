#!/usr/bin/env python3
"""Offline precompute for the LLM Narration tab.

Two-phase pipeline:

  Phase 1 (reports)     For each (scenario × driving_model), runs the XAI
                        computation loop (Semantic Graph + Counterfactuals +
                        Necessity + Grounding + Decision Class + Report
                        Builder) and writes one structured JSON report per
                        narration event (every NARRATION_INTERVAL_STEPS).

  Phase 2 (narrations)  For each (scenario × driving_model × LLM × toggle),
                        filters each report per the toggle combo, builds the
                        LLM prompt via the existing prompt_builder, calls the
                        LLM via the existing llm_narrator, and writes one
                        aggregated JSON list with all narration events.

Outputs (under data/llm_narration/):
    scenario_{idx:04d}/{model_slug}/reports/t_{step:04d}.json
    scenario_{idx:04d}/{model_slug}/narrations/{llm_key}__{toggle_key}.json

Usage:
    python scripts/precompute_llm_narration.py --phase all
    python scripts/precompute_llm_narration.py --phase reports --scenarios 0 1 2
    python scripts/precompute_llm_narration.py --phase narrations --llm gpt
    python scripts/precompute_llm_narration.py --phase narrations \\
        --llm claude --toggles full minimal

To process a different LLM: edit `active:` in config/llm_narration.yaml and
re-run, or pass --llm <key>. Each run writes one set of files; existing
narrations from other LLMs stay on disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ─── Path bootstrap ─────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CBM_ROOT = _PROJECT_ROOT / "cbm"
_POSTHOC_ROOT = _PROJECT_ROOT / "post-hoc-xai"
_LLM_INTEGRATION_ROOT = _PROJECT_ROOT / "llm_integration"

for _p in (
    str(_PROJECT_ROOT),
    str(_CBM_ROOT),
    str(_POSTHOC_ROOT),
    str(_CBM_ROOT / "V-Max"),
    str(_LLM_INTEGRATION_ROOT),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Override the stdlib `platform` module with the project's `platform` package
# (same trick as scripts/precompute_posthoc_demo.py).
_pkg_init = _PROJECT_ROOT / "platform" / "__init__.py"
_spec = importlib.util.spec_from_file_location(
    "platform", str(_pkg_init),
    submodule_search_locations=[str(_PROJECT_ROOT / "platform")],
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["platform"] = _mod
_spec.loader.exec_module(_mod)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# ─── Imports that require the path bootstrap above ──────────────────────────
import platform  # noqa: E402  triggers shared path setup
from platform.llm_narration import (  # noqa: E402
    ACTIVE_LLM_KEY,
    ACTIVE_TOGGLE_COMBOS,
    DATA_ROOT,
    DRIVING_MODELS,
    LLM_MODELS,
    NARRATION_INTERVAL_STEPS,
    SCENARIO_CATALOG,
    TOGGLE_COMBOS,
    driving_model_slug,
    get_narrations_path,
    get_report_path,
    get_reports_dir,
)
from platform.shared.model_catalog import PLATFORM_MODELS  # noqa: E402


# =============================================================================
# CONFIG LOADING
# =============================================================================

_DEFAULT_XAI_CONFIG = _LLM_INTEGRATION_ROOT / "xai" / "config" / "xai_config.yaml"


def _load_pipeline_cfg(config_path: Path) -> dict[str, Any]:
    """Load the xai pipeline YAML, overriding narration_mode to offline."""
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["narration_mode"] = "offline"  # phase 1 must not call any LLM
    # Pin the report frequency so reports are emitted at platform-known steps.
    cfg.setdefault("frequencies", {})
    cfg["frequencies"]["report"] = NARRATION_INTERVAL_STEPS
    cfg["frequencies"]["counterfactual"] = NARRATION_INTERVAL_STEPS
    cfg["frequencies"].setdefault("graph", 1)
    return cfg


# =============================================================================
# PHASE 1 — REPORTS
# =============================================================================

def _state_to_dict(state) -> dict[str, Any]:
    """Mirrors xai.run_xai_eval.state_to_dict."""
    import jax
    import numpy as np
    idx = int(np.array(jax.device_get(state.timestep))[0])
    traj = state.sim_trajectory

    def get_field(tk):
        val = jax.device_get(tk)
        if val.ndim == 3:
            return np.array(val[0, :, idx])
        if val.ndim == 2:
            return np.array(val[:, idx])
        return np.array(val)

    return {
        "current": {
            "x":     get_field(traj.x),
            "y":     get_field(traj.y),
            "vx":    get_field(traj.vel_x),
            "vy":    get_field(traj.vel_y),
            "yaw":   get_field(traj.yaw),
            "valid": get_field(traj.valid),
        }
    }


def _label_action(accel: float, steer: float) -> str:
    parts: list[str] = []
    if accel > 0.5:
        parts.append("Accelerate")
    elif accel < -0.5:
        parts.append("Brake")
    else:
        parts.append("Maintain Speed")
    if steer > 0.05:
        parts.append("Steer Left")
    elif steer < -0.05:
        parts.append("Steer Right")
    return " + ".join(parts)


def _reports_already_present(scenarios: list[int], model_key: str) -> bool:
    """True if every requested scenario has at least one report file on disk."""
    return all(
        get_reports_dir(s, model_key).is_dir()
        and any(get_reports_dir(s, model_key).glob("t_*.json"))
        for s in scenarios
    )


def _save_report(scenario_idx: int, model_key: str, step: int, report: dict) -> None:
    path = get_report_path(scenario_idx, model_key, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def run_phase_1_reports(
    scenarios: list[int],
    driving_model_keys: list[str],
    pipeline_cfg: dict[str, Any],
    dataset: str,
    seed: int,
    force: bool,
) -> None:
    """Run the XAI compute loop and write per-event reports.

    This replicates the inner loop of llm_integration/xai/run_xai_eval.py
    so we can place reports in the platform's data layout directly, without
    relying on setup_evaluation's eval_path.
    """
    # Heavy imports deferred so phase-2-only runs don't pay JAX startup.
    import jax
    import numpy as np

    from vmax.scripts.evaluate import utils as vmax_utils
    from vmax.simulator import datasets as vmax_datasets, make_data_generator

    from xai.counterfactual_explainer import (
        CounterfactualExplainer, CounterfactualConfig,
    )
    from xai.computation.semantic_graph import SemanticGraphBuilder
    from xai.computation.adaptive_action_grid import build_action_grid
    from xai.computation import (
        necessity_scorer, attention_grounder, decision_classifier, report_builder,
    )
    from xai.computation.attention_grounder import reconstruct_token_agent_ids

    scenario_set = set(scenarios)
    scenario_max = max(scenarios)

    for dm_key in driving_model_keys:
        entry = PLATFORM_MODELS.get(dm_key)
        if entry is None:
            print(f"[reports] SKIP — '{dm_key}' missing from PLATFORM_MODELS.")
            continue
        if not entry.exists_on_disk:
            print(f"[reports] SKIP — model dir not found: {entry.model_dir}")
            continue

        if not force and _reports_already_present(scenarios, dm_key):
            print(f"[reports] {dm_key}: already present (use --force to rebuild)")
            continue

        slug = driving_model_slug(dm_key)
        print(f"\n[reports] === {dm_key}  ({slug}) ===")

        # Split the absolute model_dir into (src_dir, path_model) — same convention
        # as run_xai_eval (utils.setup_evaluation builds f"{src_dir}/{path_model}/").
        model_dir = Path(entry.model_dir)
        src_dir = str(model_dir.parent)
        path_model = model_dir.name

        env, step_fn, _eval_path, _term_keys = vmax_utils.setup_evaluation(
            "ai",
            path_model,
            src_dir,
            dataset,
            f"llm_narration_precompute_{slug}",
            64,                                # max_num_objects
            False,                             # noisy_init
            sdc_paths_from_data=True,
        )

        # Auto-detect encoder layout (same logic as run_xai_eval).
        try:
            model_cfg = vmax_utils.load_yaml_config(
                f"{src_dir}/{path_model}/.hydra/config.yaml",
            )
            obs_cfg = model_cfg.get("observation_config", {})
            objects_cfg = obs_cfg.get("objects", {})
            rg_cfg = obs_cfg.get("roadgraphs", {})
            tl_cfg = obs_cfg.get("traffic_lights", {})
            pt_cfg = obs_cfg.get("path_target", {})
            num_closest_objs = objects_cfg.get("num_closest_objects", 8)
            obs_steps = obs_cfg.get("obs_past_num_steps", 5)
            pipeline_cfg["encoder_layout"] = {
                "n_sdc_timesteps":    obs_steps,
                "num_objects":        num_closest_objs,
                "timestep_agent":     obs_steps,
                "roadgraph_top_k":    rg_cfg.get("roadgraph_top_k", 1000),
                "num_traffic_lights": tl_cfg.get("num_closest_traffic_lights", 16),
                "tl_timesteps":       obs_steps,
                "gps_path_len":       pt_cfg.get("num_points", 80),
            }
            pipeline_cfg["num_closest_objects"] = num_closest_objs
        except Exception as e:
            print(f"  [WARN] could not auto-detect encoder layout: {e}")

        data_generator = make_data_generator(
            path=vmax_datasets.get_dataset(dataset),
            max_num_objects=64,
            include_sdc_paths=True,
            batch_dims=(1, 1),
            seed=seed,
            repeat=1,
        )

        graph_builder = SemanticGraphBuilder()
        cf_explainer = CounterfactualExplainer(
            config=CounterfactualConfig(
                horizon_steps=pipeline_cfg.get("rollout_horizon_steps", 10),
            ),
        )

        freqs = pipeline_cfg["frequencies"]
        f_graph = freqs["graph"]
        f_cf = freqs["counterfactual"]
        f_report = freqs["report"]

        jitted_step_fn = jax.jit(step_fn)
        jitted_reset = jax.jit(env.reset)
        rng_key = jax.random.PRNGKey(seed)

        for scenario_idx, scenario in enumerate(data_generator):
            if scenario_idx > scenario_max:
                break
            if scenario_idx not in scenario_set:
                continue

            print(f"  [reports] scenario {scenario_idx}…", flush=True)
            scenario = jax.tree_map(lambda x: x.squeeze(0), scenario)

            rng_key, reset_key = jax.random.split(rng_key)
            env_transition = jitted_reset(scenario, jax.random.split(reset_key, 1))

            is_sdc_raw = jax.device_get(env_transition.state.object_metadata.is_sdc)
            is_sdc_arr = np.array(is_sdc_raw)[0] if is_sdc_raw.ndim == 2 else np.array(is_sdc_raw)
            ego_idx = int(np.argmax(is_sdc_arr))

            st_dict = _state_to_dict(env_transition.state)
            graph = graph_builder.build(st_dict, ego_idx)
            context_categories = graph["context_categories"]

            step_count = 0
            done = False
            scenario_report_count = 0

            while not done:
                rng_key, step_key = jax.random.split(rng_key)
                current_transition = env_transition
                env_transition, rl_transition = jitted_step_fn(
                    current_transition, key=jax.random.split(step_key, 1),
                )
                done = bool(jax.device_get(env_transition.done)[0])

                # Attention weights from the policy.
                attention_weights: dict[str, Any] = {}
                try:
                    extras = rl_transition.extras
                    policy_extras = (
                        extras.get("policy_extras", {})
                        if hasattr(extras, "get") else {}
                    )
                    raw_attn = (
                        policy_extras.get("encoder_attn_weights", {})
                        if hasattr(policy_extras, "get") else {}
                    )
                    if raw_attn:
                        attention_weights = jax.device_get(raw_attn)
                except Exception:
                    pass

                if step_count % f_graph == 0:
                    st_dict = _state_to_dict(current_transition.state)
                    graph = graph_builder.build(st_dict, ego_idx)
                    context_categories = graph["context_categories"]

                cf_result = None
                if step_count % f_cf == 0:
                    dynamic_grid = build_action_grid(context_categories, pipeline_cfg)
                    cf_explainer.set_action_grid(dynamic_grid)
                    try:
                        cf_result = cf_explainer.explain_vectorized(
                            env_transition=current_transition,
                            env=env,
                            step_fn=jitted_step_fn,
                        )
                    except Exception as e:
                        print(f"    [WARN] CF failed at step {step_count}: {e}")

                necessity = {"necessity_score": 0.0, "threat_agents": []}
                if cf_result is not None:
                    necessity = necessity_scorer.compute(cf_result.get("alternatives", []))

                n_closest = pipeline_cfg.get("num_closest_objects", 8)
                token_agent_ids = reconstruct_token_agent_ids(
                    current_transition.state, ego_idx, n_closest,
                )
                grounding = attention_grounder.compute(
                    attention_weights=attention_weights,
                    encoder_layout=pipeline_cfg.get("encoder_layout", {}),
                    threat_agents=necessity["threat_agents"],
                    config=pipeline_cfg,
                    token_agent_ids=token_agent_ids,
                )

                dec_class = decision_classifier.classify(
                    necessity_score=necessity["necessity_score"],
                    grounding_score=grounding["grounding_score"],
                    config=pipeline_cfg,
                )

                if step_count % f_report == 0 or step_count == 79 or step_count == 59:
                    true_action_arr = jax.device_get(rl_transition.action)
                    if true_action_arr.ndim > 1:
                        true_action_arr = true_action_arr[0]
                    accel = float(true_action_arr[0])
                    steer = float(true_action_arr[1])
                    chosen_action = {
                        "label":   _label_action(accel, steer),
                        "accel":   round(accel, 2),
                        "steer":   round(steer, 2),
                        "outcome": None,
                    }

                    # Extract traffic light state for the current timestep
                    from xai.run_xai_eval import extract_traffic_light_state
                    tl_state = extract_traffic_light_state(current_transition.state)

                    report = report_builder.build(
                        step=step_count,
                        timestamp_s=step_count * 0.1,
                        ego_state=graph["scenario_info"],
                        chosen_action=chosen_action,
                        context_categories=context_categories,
                        scene_edges=graph["semantic_edges"],
                        alternatives=(
                            cf_result.get("alternatives", []) if cf_result else []
                        ),
                        necessity_score=necessity["necessity_score"],
                        attention_grounding=grounding,
                        decision_class=dec_class,
                        traffic_light=tl_state,
                    )
                    _save_report(scenario_idx, dm_key, step_count, report)
                    scenario_report_count += 1

                step_count += 1

            print(
                f"    ✓ {scenario_report_count} reports written "
                f"({step_count} sim steps).",
                flush=True,
            )


# =============================================================================
# PHASE 2 — NARRATIONS
# =============================================================================

def _filter_report_for_toggles(report: dict, toggles: dict) -> dict:
    """Apply the toggle filter the user expects.

    - grounding=False     → blank out attention_grounding
    - counterfactual=False → blank out alternatives + necessity, force ROUTINE
    """
    out = dict(report)
    if not toggles["grounding"]:
        out["attention_grounding"] = {"grounding_score": None, "per_agent_breakdown": []}
    if not toggles["counterfactual"]:
        out["alternatives"] = []
        out["necessity_score"] = 0.0
        # decision_class was derived from data we're now hiding — neutralize it.
        out["decision_class"] = "ROUTINE"
    return out


def _load_reports_from_disk(scenario_idx: int, model_key: str) -> dict[int, dict]:
    rd = get_reports_dir(scenario_idx, model_key)
    if not rd.is_dir():
        return {}
    out: dict[int, dict] = {}
    for f in sorted(rd.glob("t_*.json")):
        try:
            step = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        with open(f, "r") as fh:
            out[step] = json.load(fh)
    return out


def _generate_narrations_for_combo(
    reports: dict[int, dict],
    llm_cfg: dict,
    toggles: dict,
) -> list[dict]:
    """Run the LLM over each (filtered) report. Returns list-of-dicts."""
    from xai.narration import narration_router, prompt_builder, llm_narrator

    config_for_narrator = {"llm": llm_cfg}
    out: list[dict] = []
    for step in sorted(reports.keys()):
        filtered = _filter_report_for_toggles(reports[step], toggles)
        template_key, _ = narration_router.route(filtered)
        system_prompt, user_prompt = prompt_builder.build_prompt(template_key, filtered)
        text, response_time = llm_narrator.narrate(
            system_prompt, user_prompt, config_for_narrator,
        )
        out.append({
            "timestep":         step,
            "timestamp_sec":    round(step * 0.1, 2),
            "tone":             template_key,
            "text":             text,
            "response_time_s":  round(response_time, 4),
        })
    return out


def run_phase_2_narrations(
    scenarios: list[int],
    driving_model_keys: list[str],
    llm_key: str,
    toggle_keys: list[str],
    force: bool,
) -> None:
    """Generate narrations for ONE LLM across the (scenario × model × toggle) grid."""
    if llm_key not in LLM_MODELS:
        print(
            f"[narrations] unknown LLM key '{llm_key}'. "
            f"Known: {sorted(LLM_MODELS.keys())}"
        )
        return
    llm_cfg = LLM_MODELS[llm_key]
    print(f"[narrations] LLM = {llm_key}  ({llm_cfg.get('display', llm_cfg.get('model'))})")

    for dm_key in driving_model_keys:
        entry = PLATFORM_MODELS.get(dm_key)
        if entry is None:
            print(f"[narrations] SKIP — '{dm_key}' missing from PLATFORM_MODELS.")
            continue
        for s in scenarios:
            reports = _load_reports_from_disk(s, dm_key)
            if not reports:
                print(
                    f"[narrations] {dm_key} / scenario {s}: no reports on disk — "
                    f"run phase 1 first."
                )
                continue
            for tk in toggle_keys:
                if tk not in TOGGLE_COMBOS:
                    print(f"[narrations] unknown toggle key '{tk}' — skipping.")
                    continue
                out_path = get_narrations_path(s, dm_key, llm_key, tk)
                if out_path.is_file() and not force:
                    print(f"  [skip] {out_path.relative_to(DATA_ROOT)} (exists)")
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                entries = _generate_narrations_for_combo(
                    reports, llm_cfg, TOGGLE_COMBOS[tk],
                )
                with open(out_path, "w") as f:
                    json.dump(entries, f, indent=2)
                print(
                    f"  ✓ {out_path.relative_to(DATA_ROOT)} "
                    f"({len(entries)} events, {time.time()-t0:.1f}s)"
                )


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Precompute LLM narration data for the platform viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--phase", choices=("reports", "narrations", "all"), default="all",
        help="Which phase(s) to run.",
    )
    ap.add_argument(
        "--scenarios", type=int, nargs="*",
        default=[idx for idx, _ in SCENARIO_CATALOG],
        help="Scenario indices to process.",
    )
    ap.add_argument(
        "--driving-models", nargs="*",
        default=list(DRIVING_MODELS.values()),
        help="PLATFORM_MODELS keys to process (full label, may need quoting).",
    )
    ap.add_argument(
        "--llm", default=ACTIVE_LLM_KEY,
        help=(
            "Which LLM to run in phase 2 (single key from config/llm_narration.yaml::llms). "
            "Defaults to the YAML's `active:` value."
        ),
    )
    ap.add_argument(
        "--toggles", nargs="*", default=ACTIVE_TOGGLE_COMBOS,
        help="Toggle combo keys to materialize in phase 2.",
    )
    ap.add_argument(
        "--dataset", default="local_womd_valid",
        help="V-Max dataset id passed to make_data_generator.",
    )
    ap.add_argument(
        "--config", type=Path, default=_DEFAULT_XAI_CONFIG,
        help="Path to xai pipeline YAML.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing outputs instead of skipping.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    if args.phase in ("reports", "all"):
        pipeline_cfg = _load_pipeline_cfg(args.config)
        run_phase_1_reports(
            scenarios=args.scenarios,
            driving_model_keys=args.driving_models,
            pipeline_cfg=pipeline_cfg,
            dataset=args.dataset,
            seed=args.seed,
            force=args.force,
        )

    if args.phase in ("narrations", "all"):
        print("\n=== PHASE 2 — narrations ===")
        if args.llm is None:
            print(
                "[narrations] no LLM selected — set `active:` in "
                "config/llm_narration.yaml or pass --llm <key>."
            )
        else:
            run_phase_2_narrations(
                scenarios=args.scenarios,
                driving_model_keys=args.driving_models,
                llm_key=args.llm,
                toggle_keys=args.toggles,
                force=args.force,
            )

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
