#!/usr/bin/env python3
"""
Build Goal-A semantic graphs from ScenarioMax-converted Waymo TFExamples.

Fixes included vs your previous version:
1) Reads state/type as int64 (not float), robust to missing keys
2) Produces ONE edge per (ego, agent) with a `relations` list (not many duplicate edges)
3) Adds left/right (ego frame)
4) Uses past_valid to ignore invalid past timesteps when computing dynamics
5) Loops over ALL TFRecord records and writes JSONL (one graph per record)

Relations implemented (ego -> other):
- distance_bucket: very_close / close / medium / far
- ahead_of / behind_of / left_of / right_of (ego frame)
- approaching / moving_away (closing speed)
- ttc_bucket: imminent / soon / later (if approaching)
- leading / following (approx, lane-proxy using |y_local| < lane_width)
- other_braking / other_accelerating (trend over past valid steps)

You can extend later with map/lane semantics, interaction, etc.
"""

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------
# Utilities: TFExample readers
# -----------------------------

def read_agent_types(f, N: int):
    """
    Returns np.int64 array of shape (N,) or None.
    Handles:
      - state/type stored as float_list (your case)
      - state/type stored as int64_list (other cases)
    """
    if "state/type" not in f:
        return None

    feat = f["state/type"]

    if len(feat.float_list.value) > 0:
        arr = np.array(feat.float_list.value, dtype=np.float32)
        if arr.size != N:
            return None
        return np.rint(arr).astype(np.int64)

    if len(feat.int64_list.value) > 0:
        arr = np.array(feat.int64_list.value, dtype=np.int64)
        if arr.size != N:
            return None
        return arr

    return None

def _example_from_bytes(raw: bytes) -> tf.train.Example:
    return tf.train.Example.FromString(raw)


def _has_key(f: Dict[str, tf.train.Feature], key: str) -> bool:
    """True only if key exists AND contains at least one value."""
    if key not in f:
        return False
    feat = f[key]
    return (
        len(feat.float_list.value) > 0
        or len(feat.int64_list.value) > 0
        or len(feat.bytes_list.value) > 0
    )

def _maybe_float(f: Dict[str, tf.train.Feature], key: str) -> Optional[np.ndarray]:
    if not _has_key(f, key):
        return None
    return np.array(f[key].float_list.value, dtype=np.float32)

def _maybe_int(f: Dict[str, tf.train.Feature], key: str) -> Optional[np.ndarray]:
    if not _has_key(f, key):
        return None
    return np.array(f[key].int64_list.value, dtype=np.int64)



def _get_float(f: Dict[str, tf.train.Feature], key: str) -> np.ndarray:
    if not _has_key(f, key):
        raise KeyError(f"Missing float feature: {key}")
    return np.array(f[key].float_list.value, dtype=np.float32)


def _get_int(f: Dict[str, tf.train.Feature], key: str) -> np.ndarray:
    if not _has_key(f, key):
        raise KeyError(f"Missing int64 feature: {key}")
    return np.array(f[key].int64_list.value, dtype=np.int64)





def _reshape_or_fail(arr: np.ndarray, shape: Tuple[int, ...], key: str) -> np.ndarray:
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Feature {key} has size {arr.size}, expected {expected} for shape {shape}")
    return arr.reshape(shape)


def infer_N_Kpast_Kfuture(f: Dict[str, tf.train.Feature]) -> Tuple[int, int, int]:
    cur_x = _get_float(f, "state/current/x")
    N = cur_x.shape[0]

    past_x = _get_float(f, "state/past/x")
    fut_x = _get_float(f, "state/future/x")

    if past_x.size % N != 0 or fut_x.size % N != 0:
        raise ValueError("Could not infer Kpast/Kfuture: lengths not divisible by N.")

    Kpast = past_x.size // N
    Kfuture = fut_x.size // N
    return N, Kpast, Kfuture


# -----------------------------
# Semantics helpers
# -----------------------------
TYPE_MAP = {
    0: "unknown",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}

def type_label(code: Optional[int]) -> str:
    if code is None:
        return "unknown"
    return TYPE_MAP.get(int(code), f"unknown_{int(code)}")


def ego_frame(dx: float, dy: float, ego_yaw: float) -> Tuple[float, float]:
    """
    Convert world delta (dx,dy) into ego-local coordinates:
    x_local: forward, y_local: left (by convention).
    """
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    x_local = dx * c - dy * s
    y_local = dx * s + dy * c
    return x_local, y_local


def distance_bucket(dist: float) -> str:
    if dist < 5.0:
        return "very_close"
    if dist < 15.0:
        return "close"
    if dist < 30.0:
        return "medium"
    return "far"


def ttc_bucket(ttc: float) -> str:
    if ttc < 1.0:
        return "ttc_imminent"
    if ttc < 3.0:
        return "ttc_soon"
    if ttc < 6.0:
        return "ttc_later"
    return "ttc_far"


def compute_closing_speed(
    ex: float, ey: float, evx: float, evy: float,
    ox: float, oy: float, ovx: float, ovy: float
) -> Tuple[float, float]:
    """
    closing_speed > 0 => approaching, < 0 => separating.
    Also returns dist.
    """
    rel = np.array([ox - ex, oy - ey], dtype=np.float32)
    dist = float(np.linalg.norm(rel) + 1e-6)

    ego_v = np.array([evx, evy], dtype=np.float32)
    oth_v = np.array([ovx, ovy], dtype=np.float32)
    rel_v = oth_v - ego_v

    closing = float(-np.dot(rel, rel_v) / dist)
    return closing, dist


def analyze_dynamics_from_past(
    past_vx: np.ndarray, past_vy: np.ndarray, past_valid: np.ndarray
) -> Dict[str, bool]:
    """
    past_* shapes: (K,)
    past_valid shape: (K,)
    Uses only valid steps; crude acceleration estimate from last two valid points.
    """
    idx = np.where(past_valid)[0]
    if idx.size < 3:
        return {"braking": False, "accelerating": False}

    # Use last 3 valid steps for a slightly more stable trend
    i1, i2, i3 = idx[-3], idx[-2], idx[-1]
    s1 = float(math.hypot(past_vx[i1], past_vy[i1]))
    s2 = float(math.hypot(past_vx[i2], past_vy[i2]))
    s3 = float(math.hypot(past_vx[i3], past_vy[i3]))

    # Approx accel per timestep (dt cancels if we just threshold deltas)
    a1 = s2 - s1
    a2 = s3 - s2
    a = 0.5 * (a1 + a2)

    return {
        "braking": a < -0.2,       # tune
        "accelerating": a > 0.2,   # tune
    }


# -----------------------------
# Core: Record -> Graph
# -----------------------------
def record_to_graph(
    raw: bytes,
    record_index: int,
    K_use: Optional[int] = None,
    lane_width_m: float = 2.0
) -> Dict[str, Any]:
    e = _example_from_bytes(raw)
    f = e.features.feature

    N, Kpast, Kfuture = infer_N_Kpast_Kfuture(f)
    if K_use is None:
        K = Kpast
    else:
        K = min(int(K_use), int(Kpast))

    # Identify ego
    is_sdc = _get_int(f, "state/is_sdc")
    ego_idx = int(np.argmax(is_sdc))

    # Current tensors
    cur_x = _get_float(f, "state/current/x")
    cur_y = _get_float(f, "state/current/y")
    cur_vx = _get_float(f, "state/current/velocity_x")
    cur_vy = _get_float(f, "state/current/velocity_y")
    cur_yaw = _maybe_float(f, "state/current/bbox_yaw")
    if cur_yaw is None:
        # fallback: some formats store "vel_yaw" or similar
        cur_yaw = _maybe_float(f, "state/current/vel_yaw")
    cur_valid = _get_int(f, "state/current/valid").astype(bool)

    # Past tensors (reshape to N x Kpast, then slice last K)
    past_x = _reshape_or_fail(_get_float(f, "state/past/x"), (N, Kpast), "state/past/x")[:, -K:]
    past_y = _reshape_or_fail(_get_float(f, "state/past/y"), (N, Kpast), "state/past/y")[:, -K:]
    past_vx = _reshape_or_fail(_get_float(f, "state/past/velocity_x"), (N, Kpast), "state/past/velocity_x")[:, -K:]
    past_vy = _reshape_or_fail(_get_float(f, "state/past/velocity_y"), (N, Kpast), "state/past/velocity_y")[:, -K:]
    past_valid = _reshape_or_fail(_get_int(f, "state/past/valid"), (N, Kpast), "state/past/valid")[:, -K:].astype(bool)

    # Agent type (robust)
    # Many Waymo TFExamples have "state/type" (int64) at agent-level.
    types = read_agent_types(f, N)
    # Optional stable track IDs
    track_id = _maybe_int(f, "state/id")
    if track_id is not None and track_id.size != N:
        # Some records have empty or mismatched id fields; ignore.
        track_id = None

    # Ego yaw for ego frame
    ego_yaw = float(cur_yaw[ego_idx]) if cur_yaw is not None else 0.0

    # Nodes
    nodes: List[Dict[str, Any]] = []
    valid_indices = np.where(cur_valid)[0]

    # Precompute dynamics per valid agent
    dyn_flags: Dict[int, Dict[str, bool]] = {}
    for i in valid_indices:
        dyn_flags[int(i)] = analyze_dynamics_from_past(
            past_vx[i], past_vy[i], past_valid[i]
        )

    # Helper to reduce JSON size a bit
    def q(x: float) -> float:
        return float(round(x, 3))

    for i in valid_indices:
        i = int(i)
        node = {
            "id": i,
            "track_id": int(track_id[i]) if track_id is not None else None,
            "type": type_label(int(types[i])) if types is not None else "unknown",
            "is_ego": (i == ego_idx),
            "current": {
                "x": q(float(cur_x[i])),
                "y": q(float(cur_y[i])),
                "vx": q(float(cur_vx[i])),
                "vy": q(float(cur_vy[i])),
                "yaw": q(float(cur_yaw[i])) if cur_yaw is not None else None,
            },
            "past": {
                "x": [q(float(v)) for v in past_x[i]],
                "y": [q(float(v)) for v in past_y[i]],
                "vx": [q(float(v)) for v in past_vx[i]],
                "vy": [q(float(v)) for v in past_vy[i]],
                "valid": [bool(v) for v in past_valid[i]],
            },
        }
        nodes.append(node)

    # Edges: one edge per ego->other with relations list
    edges: List[Dict[str, Any]] = []

    ex, ey = float(cur_x[ego_idx]), float(cur_y[ego_idx])
    evx, evy = float(cur_vx[ego_idx]), float(cur_vy[ego_idx])

    for j in valid_indices:
        j = int(j)
        if j == ego_idx:
            continue

        ox, oy = float(cur_x[j]), float(cur_y[j])
        ovx, ovy = float(cur_vx[j]), float(cur_vy[j])

        closing, dist = compute_closing_speed(ex, ey, evx, evy, ox, oy, ovx, ovy)

        dx, dy = ox - ex, oy - ey
        x_local, y_local = ego_frame(dx, dy, ego_yaw)

        relations: List[str] = []
        # distance bucket
        relations.append(distance_bucket(dist))

        # relative position (ego frame)
        relations.append("ahead_of" if x_local > 0 else "behind_of")
        relations.append("left_of" if y_local > 0 else "right_of")

        # approach/separate
        if closing > 0.5:
            relations.append("approaching")
            ttc = dist / max(closing, 1e-6)
            relations.append(ttc_bucket(ttc))
        elif closing < -0.5:
            relations.append("moving_away")

        # leading/following (approx lane proxy)
        if abs(y_local) < lane_width_m:
            if x_local > 0:
                relations.append("leading")
            elif x_local < 0:
                relations.append("following")

        # dynamics (for the other agent)
        d = dyn_flags.get(j, {"braking": False, "accelerating": False})
        if d["braking"]:
            relations.append("other_braking")
        if d["accelerating"]:
            relations.append("other_accelerating")

        edge = {
            "src": int(ego_idx),
            "dst": int(j),
            "distance_m": q(dist),
            "closing_speed": q(closing),
            "x_local_m": q(float(x_local)),
            "y_local_m": q(float(y_local)),
            "relations": relations,
        }
        edges.append(edge)

    graph = {
        "record_index": record_index,
        "N": int(N),
        "Kpast": int(K),
        "Kfuture": int(Kfuture),
        "ego_id": int(ego_idx),
        "num_valid_agents_current": int(valid_indices.size),
        "nodes": nodes,
        "edges": edges,
    }
    return graph


# -----------------------------
# Main: TFRecord -> JSONL graphs
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True, help="Path to training.tfrecord produced by ScenarioMax")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Max records to process (0 = all)")
    ap.add_argument("--K", type=int, default=0, help="History steps to include (0 = use full Kpast)")
    ap.add_argument("--lane_width", type=float, default=2.0, help="Lane width proxy (meters) for leading/following")
    args = ap.parse_args()

    K_use = None if args.K <= 0 else args.K

    ds = tf.data.TFRecordDataset([args.tfrecord])
    count = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for idx, raw in enumerate(ds):
            if args.limit > 0 and idx >= args.limit:
                break
            g = record_to_graph(raw.numpy(), record_index=idx, K_use=K_use, lane_width_m=args.lane_width)
            fout.write(json.dumps(g) + "\n")
            count += 1

    print(f"✅ Wrote {count} graphs to {args.out}")


if __name__ == "__main__":
    main()
