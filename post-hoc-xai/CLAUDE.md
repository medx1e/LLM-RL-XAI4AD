# Project Context

## What This Project Is
Research project analyzing pretrained V-MAX autonomous driving models. V-MAX is an RL framework built on Waymax (Waymo's simulator) using JAX/Flax. We received 35+ pretrained model weights from the paper authors in `runs_rlc/`.

## Goal
Explore, load, and analyze the pretrained models — particularly attention visualization and XAI (explainable AI) on the encoder attention weights.

## Environment Setup
- Conda env: `vmax`
- Activation: `eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax`
- GPU: NVIDIA GeForce GTX 1660 Ti (6GB) — JIT compilation takes ~10 min on first run
- Required extra dependency: `pip install tensorboardX`

## Directory Structure
- `V-Max/` — V-MAX codebase (git submodule, importable from conda vmax env)
- `runs_rlc/` — 35+ pretrained model weights from the paper authors
- `data/training.tfrecord` — 1GB Waymo dataset (training scenarios)
- `Post-hoc-xai-framework-docs/` — planning docs (GUIDE2LOAD_MODELS.md, VMAX_MODELS_GUIDE.md, posthoc_xai_framework_plan.md)
- `posthoc_xai/` — **Post-Hoc XAI framework** (modular, JAX-native)
- `.xai_progress.md` — implementation progress tracker (read this to resume work)
- `explore_vmax.py` — full exploration script (load model, run scenario, extract attention params, capture intermediates)
- `run_inference.py` — quick inference script (currently set to wayformer, change MODEL_DIR to switch)

## Key Compatibility Issues (5 total, all documented in docs/GUIDE2LOAD_MODELS.md)
1. Pickle module path mismatch — fixed by `load_params()` + tensorboardX
2. Encoder type aliases — `perceiver` -> `lq`, `mgail` -> `lqh`
3. Observation type aliases — `road`/`lane` -> `vec`
4. Parameter key mismatch — `perceiver_attention` -> `lq_attention`, `mgail_attention` -> `lq_attention`
5. `speed_limit` feature — `sac_seed0/42/69` models are BROKEN (cannot run inference), use `womd_*` models instead

## Model Status
- All 5 encoder types verified working: perceiver/lq, mtr, wayformer, mgail/lqh, none
- `sac_seed0/42/69` — broken due to speed_limit feature not in Waymax
- Recommended model: `womd_sac_road_perceiver_minimal_42` (97.47% accuracy, #2 ranked)

## Inference Results So Far
- `womd_sac_road_perceiver_minimal_42`: Perfect episode, 80 steps, no violations, full route completion
- `womd_sac_road_wayformer_minimal_42`: Went offroad at step 46, comfort=0, straddling lanes 72%

## Known Bugs
- `explore_vmax.py` line 348 has a typo: `sdsc_traj_feat` should be `sdc_traj_feat`

## Post-Hoc XAI Framework
- Phase 1 COMPLETE: base classes, VanillaGradient, model loader, Perceiver wrapper, GenericWrapper
- Phase 2 TODO: IntegratedGradients, SmoothGrad, GradientXInput, Perturbation, FeatureAblation, SARFA
- Phase 3 TODO: architecture-specific attention wrappers (MTR, Wayformer, MGAIL)
- Phase 4 TODO: metrics (faithfulness, sparsity, consistency), experiments
- Phase 5 TODO: visualization (BEV overlay, temporal, comparison plots)
- Full progress: see `.xai_progress.md`
- Full plan: see `Post-hoc-xai-framework-docs/posthoc_xai_framework_plan.md`
