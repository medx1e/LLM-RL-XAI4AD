import json
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def plot_scene_clean(
    data: dict,
    *,
    k_agents: int = 20,        # how many non-ego agents to show
    label_n: int = 10,         # how many agents to label (incl ego)
    dist_max: float | None = None,  # optional distance cutoff (meters)
    edge_label_n: int = 0,     # label only this many nearest edges (0 = none)
    out_path: Path | None = None,
):
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    edges = data.get("edges", [])
    ego_id = data.get("ego_id", 0)
    if ego_id not in nodes:
        raise ValueError(f"ego_id={ego_id} not found in nodes")

    ego = nodes[ego_id]
    ego_cur = ego.get("current", {})
    x_ego, y_ego = float(ego_cur.get("x", 0.0)), float(ego_cur.get("y", 0.0))
    yaw_ego = float(ego_cur.get("yaw", 0.0))
    R = rot2d(-yaw_ego)

    def to_local_xy(xg: float, yg: float) -> tuple[float, float]:
        v = np.array([xg - x_ego, yg - y_ego], dtype=float)
        xl, yl = (R @ v).tolist()
        return float(xl), float(yl)

    def to_local_v(vxg: float, vyg: float) -> tuple[float, float]:
        v = np.array([vxg, vyg], dtype=float)
        vxl, vyl = (R @ v).tolist()
        return float(vxl), float(vyl)

    # Positions in ego frame: prefer edge-provided local coords when available
    pos = {ego_id: (0.0, 0.0)}
    for e in edges:
        dst = e.get("dst")
        if dst is None or dst not in nodes:
            continue
        if "x_local_m" in e and "y_local_m" in e:
            pos[dst] = (float(e["x_local_m"]), float(e["y_local_m"]))

    # Fallback for any node without edge-local coords
    for nid, n in nodes.items():
        if nid in pos:
            continue
        cur = n.get("current", {})
        pos[nid] = to_local_xy(float(cur.get("x", x_ego)), float(cur.get("y", y_ego)))

    # Distances and selection: keep ego + nearest K others (optionally within dist_max)
    dists = {nid: math.hypot(xy[0], xy[1]) for nid, xy in pos.items()}
    others = [nid for nid in pos.keys() if nid != ego_id]
    others_sorted = sorted(others, key=lambda i: dists.get(i, 1e9))
    if dist_max is not None:
        others_sorted = [i for i in others_sorted if dists[i] <= dist_max]

    keep = [ego_id] + others_sorted[:k_agents]
    keep_set = set(keep)

    # Filter edges: only ego -> kept nodes
    ego_edges = [
        e for e in edges
        if e.get("src") == ego_id and e.get("dst") in keep_set
    ]
    ego_edges = sorted(ego_edges, key=lambda e: float(e.get("distance_m", 1e9)))

    # --- Styling helpers ---
    def node_color(n: dict) -> str:
        if n.get("is_ego"):
            return "#ffcc00"
        t = n.get("type", "unknown")
        if t == "vehicle":
            return "#4aa3ff"
        if t in ("pedestrian", "ped"):
            return "#ff6b6b"
        if t == "cyclist":
            return "#66ff99"
        return "#dddddd"

    def node_size(n: dict) -> float:
        if n.get("is_ego"):
            return 190
        t = n.get("type", "unknown")
        return 105 if t in ("pedestrian", "ped", "cyclist") else 90

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"record_index={data.get('record_index')}  |  showing {len(keep)-1} closest agents",
        fontsize=13,
        pad=12,
    )

    # Faint edges
    for e in ego_edges:
        dst = e["dst"]
        x1, y1 = pos[ego_id]
        x2, y2 = pos[dst]
        ax.plot([x1, x2], [y1, y2], linewidth=1.1, alpha=0.20)

    # Optional: label only the nearest few edges with distance
    for e in ego_edges[:max(0, edge_label_n)]:
        dst = e["dst"]
        x1, y1 = pos[ego_id]
        x2, y2 = pos[dst]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = float(e.get("distance_m", math.hypot(x2, y2)))
        ax.text(
            mx, my, f"{dist:.1f}m",
            fontsize=8,
            alpha=0.8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.12),
        )

    # Past trails (kept nodes only) — faint
    for nid in keep:
        n = nodes[nid]
        past = n.get("past", {})
        xs = past.get("x", [])
        ys = past.get("y", [])
        val = past.get("valid", [True] * len(xs))
        if not xs or not ys:
            continue

        pts = []
        for xg, yg, ok in zip(xs, ys, val):
            if not ok:
                pts.append((np.nan, np.nan))
            else:
                pts.append(to_local_xy(float(xg), float(yg)))
        pts = np.array(pts, dtype=float)

        ax.plot(
            pts[:, 0], pts[:, 1],
            linewidth=1.6,
            alpha=0.16,
            color=node_color(n),
        )

    # Nodes + velocity arrows
    for nid in keep:
        n = nodes[nid]
        x, y = pos[nid]
        ax.scatter(
            [x], [y],
            s=node_size(n),
            color=node_color(n),
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

        cur = n.get("current", {})
        vxl, vyl = to_local_v(float(cur.get("vx", 0.0)), float(cur.get("vy", 0.0)))
        speed = math.hypot(vxl, vyl)
        if speed > 0.2:
            scale = 2.0
            ax.arrow(
                x, y,
                vxl * scale, vyl * scale,
                head_width=0.8,
                length_includes_head=True,
                alpha=0.55,
                zorder=6,
            )

    # Labels: only ego + nearest label_n-1 agents
    label_ids = [ego_id] + others_sorted[:max(0, label_n - 1)]
    for nid in label_ids:
        if nid not in keep_set:
            continue
        n = nodes[nid]
        x, y = pos[nid]
        t = n.get("type", "unknown")
        txt = "EGO" if nid == ego_id else f"id={nid} ({t})"
        ax.annotate(
            txt,
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.12),
            zorder=10,
        )

    # Auto-zoom to kept nodes
    pts = np.array([pos[i] for i in keep], dtype=float)
    xmin, ymin = np.nanmin(pts, axis=0)
    xmax, ymax = np.nanmax(pts, axis=0)
    pad = 6.0
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m, ego frame)")
    ax.set_ylabel("y (m, ego frame)")
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--k_agents", type=int, default=20)
    p.add_argument("--label_n", type=int, default=10)
    p.add_argument("--dist_max", type=float, default=None)
    p.add_argument("--edge_label_n", type=int, default=0)
    p.add_argument("--save_dir", type=str, default=None)
    args = p.parse_args()

    path = Path(args.jsonl)
    save_dir = Path(args.save_dir) if args.save_dir else None

    for i, g in enumerate(iter_jsonl(path)):
        if i >= args.n:
            break
        out = None
        if save_dir:
            out = save_dir / f"graph_{i:02d}_record_{g.get('record_index', i)}.png"
        plot_scene_clean(
            g,
            k_agents=args.k_agents,
            label_n=args.label_n,
            dist_max=args.dist_max,
            edge_label_n=args.edge_label_n,
            out_path=out,
        )

    plt.show()


if __name__ == "__main__":
    main()
