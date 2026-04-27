#!/usr/bin/env python3
import argparse
import dataclasses

import mediapy as media
from tqdm import tqdm

from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import visualization


def get_config_with_local_path(
    local_tfrecord: str,
    max_num_objects: int = 128,
    max_num_rg_points: int = 30000,
):
    """
    Make a DatasetConfig that reads from your local TFRecord file.
    """
    base = _config.WOD_1_0_0_TESTING if hasattr(_config, "WOD_1_0_0_TESTING") else _config.WOD_1_1_0_TRAINING
    cfg = dataclasses.replace(
        base,
        path=local_tfrecord,
        max_num_objects=max_num_objects,
        max_num_rg_points=max_num_rg_points,  # <-- key fix
        shuffle_seed=None,
        shuffle_buffer_size=1,
        num_shards=1,
        deterministic=True,
        repeat=None,
    )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--record", type=int, default=0, help="which scenario/record to render (0-based)")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_objects", type=int, default=128)
    ap.add_argument("--use_log_traj", action="store_true", help="render logged trajectory (recommended)")
    ap.add_argument("--max_rg_points", type=int, default=30000)
    
    args = ap.parse_args()

    
    cfg = get_config_with_local_path(
    args.tfrecord,
    max_num_objects=args.max_objects,
    max_num_rg_points=args.max_rg_points,
)
    data_iter = dataloader.simulator_state_generator(config=cfg)

    # grab record N
    scenario = None
    for _ in range(args.record + 1):
        scenario = next(data_iter)

    # build frames
    imgs = []
    state = scenario
    imgs.append(visualization.plot_simulator_state(state, use_log_traj=args.use_log_traj))

    for _ in tqdm(range(state.remaining_timesteps), desc="Rendering"):
        state = datatypes.update_state_by_log(state, num_steps=1)
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=args.use_log_traj))

    media.write_video(args.out, imgs, fps=args.fps)
    print(f"Saved video to: {args.out}")


if __name__ == "__main__":
    main()
