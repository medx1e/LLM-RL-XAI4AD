#!/usr/bin/env python3
import argparse, json, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

def F(f, key):
    return np.array(f[key].float_list.value, dtype=np.float32)

def I(f, key):
    return np.array(f[key].int64_list.value, dtype=np.int64)

def has(f, key):
    if key not in f: return False
    feat = f[key]
    return (len(feat.float_list.value)>0) or (len(feat.int64_list.value)>0) or (len(feat.bytes_list.value)>0)

def ego_frame(dx, dy, ego_yaw):
    c = math.cos(-ego_yaw); s = math.sin(-ego_yaw)
    x_local = dx * c - dy * s
    y_local = dx * s + dy * c
    return x_local, y_local

def load_record(tfrecord_path, record_index):
    ds = tf.data.TFRecordDataset([tfrecord_path])
    for i, raw in enumerate(ds):
        if i == record_index:
            ex = tf.train.Example.FromString(raw.numpy())
            return ex
    raise IndexError(f"record_index={record_index} not found")

def load_graph_line(jsonl_path, record_index):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == record_index:
                return json.loads(line)
    raise IndexError(f"graph line {record_index} not found in {jsonl_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--graphs", required=True, help="graphs.jsonl")
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--show_ids", action="store_true")
    ap.add_argument("--max_dist", type=float, default=80.0, help="only draw edges within this distance for clarity")
    ap.add_argument("--out", type=str, default=None, help="output PNG path")
    args = ap.parse_args()

    ex = load_record(args.tfrecord, args.record)
    g = load_graph_line(args.graphs, args.record)
    f = ex.features.feature

    # --- pull current state tensors ---
    cur_x = F(f, "state/current/x")
    cur_y = F(f, "state/current/y")
    cur_vx = F(f, "state/current/velocity_x")
    cur_vy = F(f, "state/current/velocity_y")
    cur_valid = I(f, "state/current/valid").astype(bool)
    is_sdc = I(f, "state/is_sdc")
    ego_idx = int(np.argmax(is_sdc))

    ego_yaw = 0.0
    if has(f, "state/current/bbox_yaw"):
        ego_yaw = float(F(f, "state/current/bbox_yaw")[ego_idx])
    elif has(f, "state/current/vel_yaw"):
        ego_yaw = float(F(f, "state/current/vel_yaw")[ego_idx])

    # --- optional: roadgraph ---
    rg_xyz = None
    rg_valid = None
    if has(f, "roadgraph_samples/xyz") and has(f, "roadgraph_samples/valid"):
        rg_xyz = F(f, "roadgraph_samples/xyz").reshape(-1, 3)
        rg_valid = I(f, "roadgraph_samples/valid").astype(bool)

    # --- plot ---
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # roadgraph points
    if rg_xyz is not None:
        pts = rg_xyz[rg_valid]
        ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3)

    # agents
    valid_idx = np.where(cur_valid)[0]
    ax.scatter(cur_x[valid_idx], cur_y[valid_idx], s=25)

    # highlight ego
    ax.scatter([cur_x[ego_idx]], [cur_y[ego_idx]], s=120, marker="*")

    if args.show_ids:
        for i in valid_idx:
            ax.text(cur_x[i], cur_y[i], str(int(i)), fontsize=8)

    # heading arrow for ego
    ax.arrow(
        cur_x[ego_idx], cur_y[ego_idx],
        5.0 * math.cos(ego_yaw), 5.0 * math.sin(ego_yaw),
        head_width=1.5, length_includes_head=True
    )

    # overlay graph edges + sanity checks
    mismatches = []
    ex0, ey0 = float(cur_x[ego_idx]), float(cur_y[ego_idx])
    for e_ in g["edges"]:
        dst = int(e_["dst"])
        dist = float(e_["distance_m"])
        if dist > args.max_dist:
            continue

        # draw edge
        ax.plot([ex0, float(cur_x[dst])], [ey0, float(cur_y[dst])], linewidth=1, alpha=0.6)

        # check label consistency (only for the relations you use)
        rels = set(e_.get("relations", []))
        x_local = float(e_.get("x_local_m", 0.0))
        y_local = float(e_.get("y_local_m", 0.0))
        closing = float(e_.get("closing_speed", 0.0))

        if ("ahead_of" in rels) and not (x_local > 0):
            mismatches.append((dst, "ahead_of", x_local, y_local, closing))
        if ("behind_of" in rels) and not (x_local < 0):
            mismatches.append((dst, "behind_of", x_local, y_local, closing))
        if ("left_of" in rels) and not (y_local > 0):
            mismatches.append((dst, "left_of", x_local, y_local, closing))
        if ("right_of" in rels) and not (y_local < 0):
            mismatches.append((dst, "right_of", x_local, y_local, closing))
        if ("approaching" in rels) and not (closing > 0.5):
            mismatches.append((dst, "approaching", x_local, y_local, closing))
        if ("moving_away" in rels) and not (closing < -0.5):
            mismatches.append((dst, "moving_away", x_local, y_local, closing))

    ax.set_title(f"Record {args.record}: nodes={len(g['nodes'])}, edges={len(g['edges'])}, ego={ego_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
    else:
        plt.show()

    print(f"\nMismatches found: {len(mismatches)}")
    for m in mismatches[:20]:
        dst, lab, xl, yl, cl = m
        print(f"  dst={dst:3d} label={lab:12s} x_local={xl:7.2f} y_local={yl:7.2f} closing={cl:6.2f}")
    if len(mismatches) > 20:
        print("  ... (truncated)")

if __name__ == "__main__":
    main()
