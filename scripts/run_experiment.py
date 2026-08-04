#!/usr/bin/env python3
"""Run the complete TRACE mission and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

# office3/room0/hotel0 are larger environments -- trace_bigarea.yaml lowers
# ctrl_reg_pos to allow bigger per-waypoint strides (see that config's
# comment). Everything else uses the standard trace_default.yaml.
BIGAREA_SCENES = {"office3", "room0", "hotel0"}

DEFAULT_HORIZON_BUDGET_FILE = ROOT / "config" / "horizon_budget.yaml"


def load_horizon_budget(path_str: str) -> dict[str, int]:
    if not path_str or not Path(path_str).exists():
        return {}
    with open(path_str) as f:
        raw = yaml.safe_load(f) or {}
    return {k: v for k, v in raw.items() if v is not None}


def scene_name(value: str) -> str:
    return value.split("/", 1)[-1]


def display(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run(command: list[str], dry_run: bool) -> int:
    print(f"\n$ {display(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run mission -> rendering eval."
    )
    p.add_argument("--scene", nargs="+", default=["office0"])
    p.add_argument(
        "--exp-id",
        default=f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    p.add_argument(
        "--planner",
        default=None,
        help=(
            "Override the auto-selected planner config. If unset, "
            f"{sorted(BIGAREA_SCENES)} use trace_bigarea, everything else "
            "uses trace_default."
        ),
    )
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--budget", type=float, default=300.0)
    p.add_argument("--record-interval", type=float, default=60.0)
    p.add_argument("--extra-save-times", type=float, nargs="+", default=None)
    p.add_argument(
        "--max-horizons",
        type=int,
        default=None,
        help=(
            "Stop after this many horizons instead of the time budget. "
            "Overrides --horizon-budget-file for every scene."
        ),
    )
    p.add_argument(
        "--horizon-budget-file",
        default="",
        help=(
            "YAML file mapping scene name -> horizon count, applied per "
            "scene when --max-horizons isn't set. Overrides --budget for "
            f"scenes present in the file (e.g. {DEFAULT_HORIZON_BUDGET_FILE}). "
            "Unset by default, so --budget is a pure time budget."
        ),
    )
    p.add_argument("--output-dir", default="experiments")
    p.add_argument("--test-root", default="dataset/replica")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--with-gui", action="store_true")
    p.add_argument("--generate-eval-poses", action="store_true")
    p.add_argument("--skip-mission", action="store_true")
    p.add_argument("--skip-render-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def phase_commands(args: argparse.Namespace, scene: str) -> list[tuple[str, list[str]]]:
    scene = scene_name(scene)
    hydra_scene = f"replica/{scene}"
    planner = args.planner or (
        "trace_bigarea" if scene in BIGAREA_SCENES else "trace_default"
    )
    common = [
        f"planner={planner}",
        f"scene={hydra_scene}",
        f"experiment.output_dir={args.output_dir}",
        f"experiment.exp_id={args.exp_id}",
        f"experiment.run_id={args.run_id}",
    ]
    phases: list[tuple[str, list[str]]] = []
    eval_dir = str((ROOT / args.test_root / scene).resolve())

    if args.generate_eval_poses and not (ROOT / args.test_root / scene / "traj.txt").exists():
        phases.append(
            (
                "evaluation poses",
                [args.python, "data_generation.py", f"scene={hydra_scene}"],
            )
        )
    if not args.skip_mission:
        mission_command = [
            args.python,
            "main.py",
            *common,
            f"experiment.budget={args.budget}",
            f"experiment.record_interval={args.record_interval}",
            f"use_gui={str(args.with_gui).lower()}",
            f"hydra.run.dir=outputs/hydra/{args.exp_id}/{scene}",
        ]
        if args.extra_save_times:
            times = ",".join(str(t) for t in args.extra_save_times)
            mission_command.append(f"experiment.extra_save_times=[{times}]")
        max_horizons = args.max_horizons
        if max_horizons is None:
            max_horizons = load_horizon_budget(args.horizon_budget_file).get(scene)
        if max_horizons is not None:
            mission_command.append(f"experiment.max_horizons={max_horizons}")
        phases.append(("mission", mission_command))
    if not args.skip_render_eval:
        phases.append(
            (
                "rendering evaluation",
                [
                    args.python,
                    "eval.py",
                    *common,
                    f"test_folder={eval_dir}",
                ],
            )
        )
    return phases


def prerequisites(args: argparse.Namespace, scene: str) -> list[str]:
    scene = scene_name(scene)
    issues = []
    config = ROOT / "config" / "scene" / "replica" / f"{scene}.yaml"
    if not config.exists():
        issues.append(f"missing scene config: {config.relative_to(ROOT)}")
    scene_dir = ROOT / "data" / "replica_v1" / scene.replace("office", "office_").replace("room", "room_")
    if scene == "hotel0":
        scene_dir = ROOT / "data" / "replica_v1" / "hotel_0"
    if not scene_dir.exists():
        issues.append(f"missing Replica scene: {scene_dir.relative_to(ROOT)}")
    needs_eval = not args.skip_render_eval
    trajectory = ROOT / args.test_root / scene / "traj.txt"
    if needs_eval and not trajectory.exists() and not args.generate_eval_poses:
        issues.append(
            f"missing evaluation poses: {trajectory.relative_to(ROOT)} "
            "(use --generate-eval-poses)"
        )
    return issues


def main() -> int:
    args = parser().parse_args()
    print(f"TRACE experiment: {args.exp_id}")
    print(f"Repository: {ROOT}")
    manifest: dict[str, object] = {
        "experiment_id": args.exp_id,
        "run_id": args.run_id,
        "scenes": [scene_name(item) for item in args.scene],
        "budget_seconds": args.budget,
        "commands": [],
    }

    for raw_scene in args.scene:
        scene = scene_name(raw_scene)
        issues = prerequisites(args, scene)
        if issues:
            print(f"\nPrerequisite check for {scene}:")
            for issue in issues:
                print(f"  - {issue}")
            if not args.dry_run:
                return 2

        for phase, command in phase_commands(args, scene):
            print(f"\n=== {scene}: {phase} ===")
            status = run(command, args.dry_run)
            manifest["commands"].append(
                {"scene": scene, "phase": phase, "command": command, "status": status}
            )
            if status:
                print(f"Phase failed with exit code {status}: {phase}", file=sys.stderr)
                return status

    if args.dry_run:
        print("\nDry run complete; no files were written.")
        return 0

    output_root = (ROOT / args.output_dir / args.exp_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nRun manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
