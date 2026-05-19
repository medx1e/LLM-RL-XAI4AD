
import argparse
import json
import math
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from vmax.scripts.evaluate import utils
from vmax.scripts.training.train_utils import str2bool
from vmax.simulator import datasets, make_data_generator

# Pipeline modules
from xai.counterfactual_explainer import (
    CounterfactualExplainer,
    CounterfactualConfig,
)
from xai.computation.semantic_graph import SemanticGraphBuilder
from xai.computation.adaptive_action_grid import build_action_grid
from xai.computation import necessity_scorer
from xai.computation import attention_grounder
from xai.computation.attention_grounder import reconstruct_token_agent_ids
from xai.computation import decision_classifier
from xai.computation import report_builder
from xai.narration import narration_router
from xai.narration.prompt_builder import build_prompt
from xai.narration import llm_narrator

# Disable GPU preallocation for JAX if needed
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# =============================================================================
# GRAPH VISUALIZATION (adapted from viz_record_graph.py)
# =============================================================================
import matplotlib.pyplot as plt
from pathlib import Path


def plot_graph_on_scene(
    state,
    graph: Dict[str, Any],
    ego_idx: int,
    output_path: str,
    max_dist: float = 80.0,
    show_ids: bool = True,
):
    """
    Plots the current scene with agents, roadgraph, and semantic graph edges.
    Saves the plot to ``output_path``.
    """

    def get_aa(x):
        arr = np.array(jax.device_get(x))
        return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr

    idx = int(get_aa(state.timestep))
    traj = state.sim_trajectory

    def get_field(tk):
        val = jax.device_get(tk)
        if val.ndim == 3:
            return np.array(val[0, :, idx])
        elif val.ndim == 2:
            return np.array(val[:, idx])
        return np.array(val)

    cur_x = get_field(traj.x)
    cur_y = get_field(traj.y)
    cur_yaw = get_field(traj.yaw)
    cur_valid = get_field(traj.valid).astype(bool)

    # Roadgraph
    rg_xyz, rg_valid = None, None
    if hasattr(state, "roadgraph_points"):
        rg = state.roadgraph_points
        if hasattr(rg, "xyz") and hasattr(rg, "valid"):
            rg_xyz_raw = jax.device_get(rg.xyz)
            rg_valid_raw = jax.device_get(rg.valid)
            if rg_xyz_raw.ndim == 3:
                rg_xyz = np.array(rg_xyz_raw[0])
                rg_valid = np.array(rg_valid_raw[0]).astype(bool)
            else:
                rg_xyz = np.array(rg_xyz_raw)
                rg_valid = np.array(rg_valid_raw).astype(bool)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    if rg_xyz is not None and rg_valid is not None:
        pts = rg_xyz[rg_valid]
        ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3)

    valid_idx = np.where(cur_valid)[0]
    ax.scatter(cur_x[valid_idx], cur_y[valid_idx], s=25)

    ex, ey = float(cur_x[ego_idx]), float(cur_y[ego_idx])
    ego_yaw = float(cur_yaw[ego_idx])
    ax.scatter([ex], [ey], s=120, marker="*")

    if show_ids:
        for i in valid_idx:
            ax.text(cur_x[i], cur_y[i], str(int(i)), fontsize=8)

    ax.arrow(
        ex, ey,
        5.0 * math.cos(ego_yaw), 5.0 * math.sin(ego_yaw),
        head_width=1.5, length_includes_head=True,
    )

    edges = graph.get("semantic_edges", [])
    for e_ in edges:
        dst = int(e_["target_id"])
        dist = float(e_.get("raw_distance", 0.0))
        if dist > max_dist:
            continue
        ax.plot(
            [ex, float(cur_x[dst])], [ey, float(cur_y[dst])],
            linewidth=1, alpha=0.6,
        )

    ax.set_title(
        f"nodes={len(graph.get('context_nodes', []))}, "
        f"edges={len(edges)}, ego={ego_idx}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    -> Saved graph plot: {out_path}")


# =============================================================================
# STATE CONVERSION UTILITY
# =============================================================================


def state_to_dict(state) -> Dict[str, Any]:
    """Convert JAX SimulatorState to a Python dictionary of numpy arrays."""

    def get_aa(x):
        return np.array(jax.device_get(x))[0]

    idx = int(get_aa(state.timestep))
    traj = state.sim_trajectory

    def get_field(tk):
        val = jax.device_get(tk)
        if val.ndim == 3:
            return np.array(val[0, :, idx])
        elif val.ndim == 2:
            return np.array(val[:, idx])
        return np.array(val)

    return {
        "current": {
            "x": get_field(traj.x),
            "y": get_field(traj.y),
            "vx": get_field(traj.vel_x),
            "vy": get_field(traj.vel_y),
            "yaw": get_field(traj.yaw),
            "valid": get_field(traj.valid),
        }
    }


# =============================================================================
# LOAD PIPELINE CONFIG
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline config from YAML file."""
    default_path = os.path.join(os.path.dirname(__file__), "config", "xai_config.yaml")
    path = config_path if config_path else default_path
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================


def run_xai_eval(args):
    print(f"-> Setting up evaluation for {args.sdc_actor}...")

    # ── Load pipeline config ──
    pipeline_cfg = load_config(args.config)

    # ── 1. Setup Env & Policy ──
    env, step_fn, eval_path, termination_keys = utils.setup_evaluation(
        args.sdc_actor,
        args.path_model,
        args.src_dir,
        args.path_dataset,
        args.eval_name,
        args.max_num_objects,
        args.noisy_init,
        sdc_paths_from_data=(not args.waymo_dataset),
    )

    # ── 1b. Auto-detect encoder layout from model's observation_config ──
    if args.sdc_actor == "ai":
        try:
            run_path = f"{args.src_dir}/{args.path_model}/"
            model_cfg = utils.load_yaml_config(run_path + ".hydra/config.yaml")
            obs_cfg = model_cfg.get("observation_config", {})
            objects_cfg = obs_cfg.get("objects", {})
            rg_cfg = obs_cfg.get("roadgraphs", {})
            tl_cfg = obs_cfg.get("traffic_lights", {})
            pt_cfg = obs_cfg.get("path_target", {})

            # num_closest_objects is the actual count of non-SDC agents
            # in the encoder's token sequence (NOT max_num_objects).
            num_closest_objs = objects_cfg.get("num_closest_objects", 8)
            obs_steps = obs_cfg.get("obs_past_num_steps", 5)

            auto_layout = {
                "n_sdc_timesteps": obs_steps,
                "num_objects": num_closest_objs,
                "timestep_agent": obs_steps,
                "roadgraph_top_k": rg_cfg.get("roadgraph_top_k", 1000),
                "num_traffic_lights": tl_cfg.get("num_closest_traffic_lights", 16),
                "tl_timesteps": obs_steps,
                "gps_path_len": pt_cfg.get("num_points", 80),
            }
            total_tokens = (
                auto_layout["n_sdc_timesteps"]
                + auto_layout["num_objects"] * auto_layout["timestep_agent"]
                + auto_layout["roadgraph_top_k"]
                + auto_layout["num_traffic_lights"] * auto_layout["tl_timesteps"]
                + auto_layout["gps_path_len"]
            )
            print(f"    [INFO] Auto-detected encoder layout: {auto_layout} → {total_tokens} tokens")
            pipeline_cfg["encoder_layout"] = auto_layout
            pipeline_cfg["num_closest_objects"] = num_closest_objs
        except Exception as e:
            print(f"    [WARN] Could not auto-detect encoder layout: {e}. Using xai_config defaults.")

    # ── 2. Data Generator ──
    batch_dims = (1, 1)
    data_generator = make_data_generator(
        path=datasets.get_dataset(args.path_dataset),
        max_num_objects=args.max_num_objects,
        include_sdc_paths=(not args.waymo_dataset),
        batch_dims=batch_dims,
        seed=args.seed,
        repeat=1,
    )

    # ── 3. Pipeline Helpers ──
    graph_builder = SemanticGraphBuilder()

    cf_config = CounterfactualConfig(
        horizon_steps=pipeline_cfg.get("rollout_horizon_steps", 10),
    )
    cf_explainer = CounterfactualExplainer(config=cf_config)

    # Re-extract frequencies for easy access
    freqs = pipeline_cfg.get("frequencies", {})
    f_graph = freqs.get("graph", 1)
    f_cf = freqs.get("counterfactual", 20)
    f_report = freqs.get("report", 20)
    f_plot = freqs.get("plotting", 20)
    f_narration = freqs.get("narration", 20)

    results_log: List[Dict[str, Any]] = []

    # JIT
    jitted_step_fn = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)
    rng_key = jax.random.PRNGKey(args.seed)

    scenarios_processed = 0
    narration_mode = pipeline_cfg.get("narration_mode", "offline")

    print(f"-> Starting loop. Limit: {args.limit} scenarios.")

    for scenario_idx, scenario in enumerate(data_generator):
        if scenarios_processed >= args.limit:
            break

        if args.scenario_indexes and scenario_idx not in args.scenario_indexes:
            continue

        print(f"Processing Scenario {scenario_idx}...")

        scenario = jax.tree_map(lambda x: x.squeeze(0), scenario)

        # Reset
        rng_key, reset_key = jax.random.split(rng_key)
        reset_key = jax.random.split(reset_key, 1)
        env_transition = jitted_reset(scenario, reset_key)

        scenario_events: List[Dict[str, Any]] = []
        episode_images: List[Any] = []
        done = False
        step_count = 0

        # SDC index
        is_sdc_raw = jax.device_get(env_transition.state.object_metadata.is_sdc)
        if is_sdc_raw.ndim == 2:
            is_sdc_arr = np.array(is_sdc_raw)[0]
        else:
            is_sdc_arr = np.array(is_sdc_raw)
        ego_idx = int(np.argmax(is_sdc_arr))

        # Build the initial semantic graph before entering the loop
        # so the first step always has a valid graph regardless of f_graph.
        st_dict = state_to_dict(env_transition.state)
        graph = graph_builder.build(st_dict, ego_idx)
        context_categories = graph["context_categories"]

        while not done:
            # ================================================================
            # K. STEP — do this first so we get the policy's attention weights
            #    for the *current* observation (pre-step state).
            # ================================================================
            rng_key, step_key = jax.random.split(rng_key)
            step_key = jax.random.split(step_key, 1)
            current_transition = env_transition  # state before action
            env_transition, rl_transition = jitted_step_fn(current_transition, key=step_key)
            done = bool(jax.device_get(env_transition.done)[0])

            # Extract attention weights produced alongside the action.
            # For non-AI policies (expert, idm, etc.) this will be {}.
            attention_weights: Dict[str, Any] = {}
            try:
                extras = rl_transition.extras
                if step_count == 0 and scenario_idx == 0:
                    print(f"    [DEBUG] rl_transition type: {type(rl_transition)}")
                    print(f"    [DEBUG] extras type: {type(extras)}")
                    print(f"    [DEBUG] extras keys: {extras.keys() if hasattr(extras, 'keys') else 'N/A'}")
                    if hasattr(extras, 'keys') and "policy_extras" in extras:
                        pe = extras["policy_extras"]
                        print(f"    [DEBUG] policy_extras type: {type(pe)}")
                        print(f"    [DEBUG] policy_extras keys: {pe.keys() if hasattr(pe, 'keys') else 'N/A'}")
                        if hasattr(pe, 'keys') and "encoder_attn_weights" in pe:
                            aw = pe["encoder_attn_weights"]
                            print(f"    [DEBUG] encoder_attn_weights type: {type(aw)}")
                            if hasattr(aw, 'keys'):
                                print(f"    [DEBUG] attn keys: {list(aw.keys())}")
                                for k, v in aw.items():
                                    print(f"    [DEBUG]   {k}: shape={v.shape}")
                            elif hasattr(aw, 'shape'):
                                print(f"    [DEBUG] attn shape: {aw.shape}")

                policy_extras = extras.get("policy_extras", {}) if hasattr(extras, 'get') else {}
                raw_attn = policy_extras.get("encoder_attn_weights", {}) if hasattr(policy_extras, 'get') else {}

                if raw_attn is not None and raw_attn:
                    attention_weights = jax.device_get(raw_attn)
            except Exception as e:
                if step_count == 0:
                    print(f"    [DEBUG] Failed to extract attention weights: {e}")

            # ================================================================
            # A. SEMANTIC GRAPH (Module 2)
            # ================================================================
            if step_count % f_graph == 0:
                st_dict = state_to_dict(current_transition.state)
                graph = graph_builder.build(st_dict, ego_idx)
                context_categories = graph["context_categories"]

            # ================================================================
            # B. COUNTERFACTUAL ROLLOUTS (Modules 3 + 4)
            # ================================================================
            counterfactual_result = None
            if args.enable_counterfactuals and step_count % f_cf == 0:
                try:
                    # Module 3: Adaptive action grid
                    dynamic_grid = build_action_grid(context_categories, pipeline_cfg)
                    cf_explainer.set_action_grid(dynamic_grid)

                    print(
                        f"    [XAI] Counterfactuals: {len(dynamic_grid)} actions "
                        f"(context: {context_categories})"
                    )

                    counterfactual_result = cf_explainer.explain_vectorized(
                        env_transition=current_transition,
                        env=env,
                        step_fn=jitted_step_fn,
                    )
                except Exception as e:
                    print(f"    ⚠️ Counterfactual generation failed: {e}")
                    counterfactual_result = None

            # ================================================================
            # C. NECESSITY SCORE (Module 5)
            # ================================================================
            necessity_result = {"necessity_score": 0.0, "threat_agents": []}
            if counterfactual_result is not None:
                necessity_result = necessity_scorer.compute(
                    counterfactual_result.get("alternatives", [])
                )

            # ================================================================
            # D. ATTENTION GROUNDING (Module 6)
            #    attention_weights come from the policy step above —
            #    these are the exact weights used to make the decision.
            #    token_agent_ids maps token-slot → original agent ID,
            #    replicating the feature extractor's distance-sort.
            # ================================================================
            n_closest = pipeline_cfg.get("num_closest_objects",
                                        pipeline_cfg.get("encoder_layout", {}).get("num_objects", 64))
            token_agent_ids = reconstruct_token_agent_ids(
                current_transition.state, ego_idx, n_closest
            )
            grounding_result = attention_grounder.compute(
                attention_weights=attention_weights,
                encoder_layout=pipeline_cfg.get("encoder_layout", {}),
                threat_agents=necessity_result["threat_agents"],
                config=pipeline_cfg,
                token_agent_ids=token_agent_ids,
            )

            # ================================================================
            # E. DECISION CLASSIFICATION (Module 7)
            # ================================================================
            dec_class = decision_classifier.classify(
                necessity_score=necessity_result["necessity_score"],
                grounding_score=grounding_result["grounding_score"],
                config=pipeline_cfg,
            )

            # ================================================================
            # F. BUILD & SAVE REPORT (Module 8)
            # ================================================================
            # Extract the REAL mathematical continuous action from the RL policy
            true_action_arr = jax.device_get(rl_transition.action)
            if true_action_arr.ndim > 1:
                true_action_arr = true_action_arr[0]
                
            accel, steer = float(true_action_arr[0]), float(true_action_arr[1])
            
            # Simple heuristic labeling for the prompt
            label_parts = []
            if accel > 0.5: label_parts.append("Accelerate")
            elif accel < -0.5: label_parts.append("Brake")
            else: label_parts.append("Maintain Speed")
            
            if steer > 0.05: label_parts.append("Steer Left")
            elif steer < -0.05: label_parts.append("Steer Right")
            
            action_label = " + ".join(label_parts)
            
            chosen_action = {
                "label": action_label,
                "accel": round(accel, 2),
                "steer": round(steer, 2),
                "outcome": None
            }

            report = None
            if step_count % f_report == 0:
                report = report_builder.build(
                    step=step_count,
                    timestamp_s=step_count * 0.1,
                    ego_state=graph["scenario_info"],
                    chosen_action=chosen_action,
                    context_categories=context_categories,
                    scene_edges=graph["semantic_edges"],
                    alternatives=(
                        counterfactual_result.get("alternatives", [])
                        if counterfactual_result
                        else []
                    ),
                    necessity_score=necessity_result["necessity_score"],
                    attention_grounding=grounding_result,
                    decision_class=dec_class,
                )

                # G. LLM NARRATION (Modules 9 → 10 → 11)  [online mode]
                narration = None
                llm_model_name = None
                if narration_mode == "online":
                    llm_model_name = pipeline_cfg.get("llm", {}).get("model", "unknown_model")
                    template_key, _ = narration_router.route(report)
                    system_prompt, user_prompt = build_prompt(template_key, report)
                    narration, response_time = llm_narrator.narrate(
                        system_prompt, user_prompt, pipeline_cfg
                    )
                    report["narration"] = narration
                    report["narration_response_time_s"] = response_time
                    print(f"    [LLM] {narration[:120]}... ({response_time}s)")

                report_path = report_builder.save(
                    report, 
                    eval_path, 
                    f"scenario_{scenario_idx}", 
                    step_count,
                    model_name=llm_model_name
                )
                print(f"    [XAI] Report saved: {report_path} [{dec_class}]")

            # ================================================================
            # H. GRAPH VISUALIZATION
            # ================================================================
            if args.render_plots and step_count % f_plot == 0:
                graph_plot_path = os.path.join(
                    eval_path,
                    "graph_plots",
                    f"scenario_{scenario_idx}_step_{step_count}.png",
                )
                plot_graph_on_scene(
                    current_transition.state, graph, ego_idx, graph_plot_path
                )

            # ================================================================
            # I. RENDER (optional)
            # ================================================================
            if args.render:
                img = utils.plot_scene(env, current_transition, sdc_pov=False)
                episode_images.append(img)

            # ================================================================
            # J. LOG
            # ================================================================
            scenario_events.append(
                {
                    "step": step_count,
                    "timestamp": round(step_count * 0.1, 2),
                    "graph": graph,
                    "counterfactual": counterfactual_result,
                    "necessity": necessity_result,
                    "grounding": grounding_result,
                    "decision_class": dec_class,
                    "narration": narration,
                    "has_attention": bool(attention_weights),
                }
            )

            step_count += 1

        # End of Scenario
        results_log.append(
            {
                "scenario_id": f"scenario_{scenario_idx}",
                "events": scenario_events,
            }
        )

        if args.render and episode_images:
            utils.write_video(eval_path, episode_images, idx=scenario_idx)

        scenarios_processed += 1

    # ── Save Logs ──
    out_file = os.path.join(eval_path, "eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results_log, f, indent=2, default=str)

    print(f"✅ Evaluation finished. Logs saved to {out_file}")


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic Eval Args
    parser.add_argument("--sdc_actor", default="expert", help="Actor type")
    parser.add_argument("--path_model", default="", help="Model identifier")
    parser.add_argument("--path_dataset", default="local_womd_valid")
    parser.add_argument("--limit", type=int, default=20, help="Max scenarios")
    parser.add_argument("--render", type=str2bool, default=False)

    # Boilerplate for setup_evaluation
    parser.add_argument("--src_dir", default="runs")
    parser.add_argument("--eval_name", default="xai_eval")
    parser.add_argument("--max_num_objects", type=int, default=64)
    parser.add_argument("--noisy_init", type=str2bool, default=False)
    parser.add_argument("--waymo_dataset", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario_indexes", nargs="*", type=int)

    # Counterfactual / Pipeline Args
    parser.add_argument(
        "--enable_counterfactuals",
        type=str2bool,
        default=True,
        help="Enable counterfactual explanations",
    )
    parser.add_argument(
        "--render_plots",
        type=str2bool,
        default=False,
        help="Enable semantic graph plots and scene rendering",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to xai_config.yaml (defaults to xai/xai_config.yaml)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_xai_eval(args)
