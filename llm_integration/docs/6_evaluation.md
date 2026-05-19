# Evaluation Script Documentation

This document explains how to use the V-Max evaluation script, its workflow, and the available arguments. It is intended for newcomers and users who want to evaluate trained models or rule-based agents on driving scenarios.

## Overview

The evaluation script allows you to assess the performance of a policy (either a learned model or a rule-based agent) on a dataset of driving scenarios. It supports batch evaluation, rendering of episodes, and saving of results and videos.

## How Evaluation Works

1. **Setup**: The script sets up the simulation environment and loads the specified policy (AI or rule-based).
2. **Data Generation**: It loads scenarios from the chosen dataset, batching them if requested.
3. **Simulation**: For each scenario, the policy is run in the environment, and metrics are collected.
4. **Rendering (Optional)**: If enabled, the script renders each episode as a video.
5. **Results**: Aggregated metrics and per-scenario results are saved to disk for analysis.

## Running the Evaluation Script

You can run the evaluation script from the command line:

```bash
python -m vmax.scripts.evaluate.evaluate [arguments]
```

## Arguments

- `--sdc_actor` (`-sdc`): Type of actor to evaluate. Use `ai` for a learned policy, or a rule-based type (e.g., `expert`, `idm`). Default: `expert`.
- `--max_num_objects` (`-o`): Maximum number of objects in the scene. Default: `64`.
- `--scenario_indexes` (`-si`): List of scenario indexes to evaluate. If not set, evaluates all scenarios.
- `--render` (`-r`): Whether to render the evaluation as videos. Default: `False`.
- `--sdc_pov` (`-pov`): Render from the self-driving car's point of view. Default: `False`.
- `--path_dataset` (`-pd`): Path or name of the dataset to use. Default: `local_womd_valid`.
- `--path_model` (`-pm`): Path to the trained model (required if `--sdc_actor ai`).
- `--eval_name` (`-en`): Name or directory for evaluation outputs. Default: `benchmark`.
- `--noisy_init` (`-ni`): Enable noisy initialization for scenarios. Default: `False`.
- `--src_dir` (`-sd`): Source directory for model checkpoints. Default: `runs`.
- `--seed`: Random seed for reproducibility. Default: `0`.
- `--batch_size` (`-bs`): Number of scenarios to process in parallel. Default: `1`.
- `--waymo_dataset` (`-wd`): Use the Waymo dataset. Default: `False`.
- `--plot-failures` (`-pf`): Plot failed scenarios (accuracy=0) from a previous evaluation. Default: `False`.

## Example Usage

Evaluate a trained AI model:

```bash
python -m vmax.scripts.evaluate.evaluate --sdc_actor ai --path_model name_of_the_run --path_dataset womd_valid --batch_size 8
```

Render failed scenarios from a previous evaluation:

```bash
python -m vmax.scripts.evaluate.evaluate --sdc_actor ai --path_model name_of_the_run --plot-failures
```

## Output Files

- `evaluation_episodes.csv`: Per-scenario metrics.
- `evaluation_results.txt`: Aggregated metrics summary.
- `mp4/`: Directory containing rendered videos (if rendering is enabled).

## Tips

- When rendering (`--render` or `--sdc_pov`), set `--batch_size 1`.
- For AI policies, always provide `--path_model`.
- Use `--scenario_indexes` to evaluate specific scenarios.

For more details, see the code in `vmax/scripts/evaluate/evaluate.py` and `vmax/scripts/evaluate/utils.py`.
