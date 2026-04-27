# CBM Training From Scratch (End-to-End)

This guide is for training the Concept Bottleneck Model (CBM) fully from scratch on the 150GB dataset, without initializing from a pretrained V-Max encoder.

---

## 1. Why Train From Scratch?
In our previous experiments (Frozen/Joint), we relied on an encoder that was already optimized for non-interpretable driving. Training from scratch ensures that the **Representation Learning** is entirely guided by the Concept Bottleneck from Day 1. This is a true "end-to-end" CBM experiment.

## 2. Configuration
We use the dedicated `cbm_v1/config_womd_scratch.yaml`. Unlike other configs:
- It **does not** require a `pretrained_dir`.
- It includes all architecture, reward, and observation specs **inline**.
- It is set to `mode: scratch`.

## 3. How to Launch

Simply run the standard training script with the scratch config:

```bash
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml
```

## 4. Key Parameters for 150GB Run
- **Total Timesteps:** `15,000,000` (Approx. 1 full epoch over 150GB data).
- **Log Frequency:** `10` (Logs every ~6,400 steps).
- **Save Frequency:** `1000` (Checkpoint every ~640k steps).
- **VRAM Required:** ~12-24GB (A100/RTX 3090/4090 recommended).

## 5. Expected Behavior
- **Initialization:** You will see a log message: `Scratch mode: encoder initialized with random weights.`
- **Early Performance:** Driving performance will be very poor initially (random steering).
- **Concept Learning:** You should see `train/concept_loss` drop rapidly in the first 100k steps as the model quickly learns the basic geometry from nothing.
- **Task Learning:** The agent should start following the path and avoiding collisions as the concept representations stabilize.
