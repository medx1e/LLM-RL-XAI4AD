"""CLI for event mining: mine, summary, export, render commands."""

from __future__ import annotations

import argparse
import sys


def cmd_mine(args):
    """Mine events from model rollouts."""
    from event_mining.miner import mine_events

    print(f"Mining events from {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Scenarios: {args.n_scenarios}")

    catalog = mine_events(
        model_or_dir=args.model,
        n_scenarios=args.n_scenarios,
        data_path=args.data,
        save_path=args.output,
    )

    summary = catalog.summary()
    print(f"\nMining complete!")
    print(f"  Total events: {summary['total_events']}")
    print(f"  Unique scenarios: {summary['unique_scenarios']}")
    print(f"  By type: {summary['by_type']}")
    print(f"  By severity: {summary['by_severity']}")


def cmd_summary(args):
    """Print summary of an existing catalog."""
    from event_mining.catalog import EventCatalog

    catalog = EventCatalog.load(args.catalog)
    summary = catalog.summary()

    print(f"Event Catalog: {args.catalog}")
    print(f"  Total events: {summary['total_events']}")
    print(f"  Unique scenarios: {summary['unique_scenarios']}")
    print(f"\n  By type:")
    for t, count in sorted(summary["by_type"].items()):
        print(f"    {t}: {count}")
    print(f"\n  By severity:")
    for s, count in sorted(summary["by_severity"].items()):
        print(f"    {s}: {count}")

    # Per-scenario breakdown
    by_scenario = catalog.by_scenario()
    print(f"\n  Per scenario:")
    for sid, events in sorted(by_scenario.items()):
        types = [e.event_type.value for e in events]
        print(f"    {sid}: {len(events)} events — {', '.join(types)}")


def cmd_export(args):
    """Export catalog to CSV."""
    from event_mining.catalog import EventCatalog
    from event_mining.integration.xai_bridge import XAIBridge

    catalog = EventCatalog.load(args.catalog)
    bridge = XAIBridge(catalog)
    df = bridge.to_dataframe()
    df.to_csv(args.output, index=False)
    print(f"Exported {len(df)} events to {args.output}")


def cmd_render(args):
    """Render BEV video for a scenario."""
    from event_mining.catalog import EventCatalog
    from event_mining.integration.vmax_adapter import VMaxAdapter
    import posthoc_xai as xai
    import os

    catalog = EventCatalog.load(args.catalog)
    scenario_events = catalog.filter(scenario_id=args.scenario)

    if not scenario_events:
        print(f"No events found for scenario {args.scenario}")
        return

    # Load model and re-extract scenario data
    print(f"Loading model from {args.model}...")
    model = xai.load_model(args.model, data_path=args.data)

    # Find the scenario index from the scenario_id
    scenario_idx = int(args.scenario.lstrip("s"))

    from vmax.simulator import make_data_generator
    loaded = model._loaded
    data_gen = make_data_generator(
        path=args.data,
        max_num_objects=loaded.config.get("max_num_objects", 64),
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=42,
        repeat=1,
    )

    # Skip to the right scenario
    data_iter = iter(data_gen)
    for i in range(scenario_idx + 1):
        scenario = next(data_iter)

    adapter = VMaxAdapter(store_raw_obs=False)
    scenario_data = adapter.extract_scenario_data(
        model, scenario, args.scenario, rng_seed=scenario_idx
    )

    os.makedirs(args.output, exist_ok=True)

    from event_mining.visualization.bev_video import render_scenario_video, render_event_clip

    # Full scenario video
    video_path = os.path.join(args.output, f"{args.scenario}_full.mp4")
    render_scenario_video(scenario_data, scenario_events, video_path)

    # Individual event clips
    for i, event in enumerate(scenario_events):
        clip_path = os.path.join(
            args.output,
            f"{args.scenario}_{event.event_type.value}_{i}.mp4",
        )
        render_event_clip(scenario_data, event, clip_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="event_mining",
        description="Event Mining for Autonomous Driving Scenarios",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # mine
    p_mine = subparsers.add_parser("mine", help="Mine events from model rollouts")
    p_mine.add_argument("--model", required=True, help="Path to model directory")
    p_mine.add_argument("--data", default="data/training.tfrecord", help="Path to .tfrecord")
    p_mine.add_argument("--n-scenarios", type=int, default=50, help="Number of scenarios")
    p_mine.add_argument("--output", default="events/catalog.json", help="Output catalog path")

    # summary
    p_summary = subparsers.add_parser("summary", help="Print catalog summary")
    p_summary.add_argument("--catalog", required=True, help="Path to catalog JSON")

    # export
    p_export = subparsers.add_parser("export", help="Export catalog to CSV")
    p_export.add_argument("--catalog", required=True, help="Path to catalog JSON")
    p_export.add_argument("--output", default="events.csv", help="Output CSV path")

    # render
    p_render = subparsers.add_parser("render", help="Render BEV video for a scenario")
    p_render.add_argument("--catalog", required=True, help="Path to catalog JSON")
    p_render.add_argument("--model", required=True, help="Path to model directory")
    p_render.add_argument("--data", default="data/training.tfrecord", help="Path to .tfrecord")
    p_render.add_argument("--scenario", required=True, help="Scenario ID (e.g., s000)")
    p_render.add_argument("--output", default="videos/", help="Output directory")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "mine": cmd_mine,
        "summary": cmd_summary,
        "export": cmd_export,
        "render": cmd_render,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
