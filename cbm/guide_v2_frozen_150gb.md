# CBM-V2 Cluster Training Guide (150GB Frozen)

**WARNING:** DO NOT run `train_joint.py` or switch the config to `mode: joint` yet! Our evaluations on the 10GB/150GB datasets proved that jumping straight to Joint mode causes **Catastrophic Interference**, dropping task accuracy directly to 22%. 

For Phase 2 of our project, we are evaluating the 15 new V2 spatial concepts in a **Frozen Encoder Phase**.

---

## 1. The Strategy
We have expanded the Concept Bottleneck to 15 concepts (we added 4 new complex spatial path parameters). 
The goal of this cluster run is to pass the **entire 150GB dataset (15,000,000 steps)** through the untouched frozen encoder. This allows the Actor network and the Concept Head to safely and organically map to the new spatial constraints without corrupting the natively learned features.

Our target is to verify if pushing 150GB of data through the extended 15-concept Bottleneck allows the network to surpass the previous 90% hard ceiling.

## 2. Hardware Requirements
Because we increased the batch parallelization (`num_episodes_per_env: 8`, `batch_size: 128`), this config is tailored for a single high-performance cluster GPU.
- **Minimum VRAM:** 12GB+
- **Recommended VRAM:** 24GB (A10G, A100, RTX 3090/4090)

## 3. How to Launch

The config file is already perfectly tuned and prepared. Simply run:

```bash
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_frozen_v2_150gb.yaml
```

## 4. What to Expect (Monitoring)
- **Startup Delay:** The script will stay silent for a few minutes while it collects `10,000` steps of initial data to pre-fill its memory buffer. This is normal.
- **Logging Frequency:** To prevent flooding the disk over 150GB, TensorBoard logs are dumped every 10 iterations (roughly every `~6,400` environment steps). 
- **Checkpoints:** The huge `.pkl` weights are saved every `1000` iterations (`~640,000` env steps).

### Important Metrics to watch on TensorBoard (`runs_cbm/cbm_v2_frozen_womd_150gb/tb`):
- `train/concept_loss`: Should steadily decrease over the millions of steps as the expanded Concept Head learns the complex JAX math of the curve parameters.
- `train/policy_loss`: Should become increasingly negative.
- `metrics/at_fault_collision`: Must trend firmly downwards towards `0.00`.
