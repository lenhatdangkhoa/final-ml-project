#!/usr/bin/env python3
"""
Run DUET + local branch across all multivariate dataset scripts and summarize results.

This script:
1) Reads commands from scripts/multivariate_forecast/*_script/DUET.sh
2) Keeps horizons in {96, 192, 336, 720}
3) Injects local-branch hyperparameters into --model-hyper-params
4) Runs benchmarks (optional --dry-run)
5) Collects mse_norm/mae_norm into a single CSV summary table
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple


TARGET_HORIZONS = {96, 192, 336, 720}


@dataclass
class RunSpec:
    dataset: str
    horizon: int
    base_command: List[str]
    save_path: str


def _extract_flag(tokens: List[str], flag: str) -> Optional[str]:
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def _replace_flag(tokens: List[str], flag: str, new_value: str) -> List[str]:
    out = tokens[:]
    for i, tok in enumerate(out):
        if tok == flag and i + 1 < len(out):
            out[i + 1] = new_value
            return out
    out.extend([flag, new_value])
    return out


def parse_duet_scripts(repo_root: Path) -> List[RunSpec]:
    pattern = repo_root / "scripts" / "multivariate_forecast" / "*_script" / "DUET.sh"
    specs: List[RunSpec] = []

    for script_path in sorted(glob.glob(str(pattern))):
        with open(script_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "run_benchmark.py" not in line:
                    continue

                tokens = shlex.split(line)
                strategy_args_raw = _extract_flag(tokens, "--strategy-args")
                data_name = _extract_flag(tokens, "--data-name-list")
                save_path = _extract_flag(tokens, "--save-path")
                if not strategy_args_raw or not data_name or not save_path:
                    continue

                try:
                    strategy_args = json.loads(strategy_args_raw)
                    horizon = int(strategy_args.get("horizon"))
                except Exception:
                    continue

                if horizon not in TARGET_HORIZONS:
                    continue

                specs.append(
                    RunSpec(
                        dataset=data_name,
                        horizon=horizon,
                        base_command=tokens,
                        save_path=save_path,
                    )
                )
    return specs


def apply_local_branch(
    tokens: List[str], save_path_suffix: str, local_kernel_size: int, local_alpha: float
) -> Tuple[List[str], str]:
    model_hparams_raw = _extract_flag(tokens, "--model-hyper-params")
    if not model_hparams_raw:
        raise ValueError("Missing --model-hyper-params")

    model_hparams = json.loads(model_hparams_raw)
    model_hparams["use_local_branch"] = True
    model_hparams["local_kernel_size"] = int(local_kernel_size)
    model_hparams["local_alpha"] = float(local_alpha)

    out = _replace_flag(tokens, "--model-hyper-params", json.dumps(model_hparams))

    orig_save_path = _extract_flag(out, "--save-path")
    if not orig_save_path:
        raise ValueError("Missing --save-path")
    new_save_path = orig_save_path.rsplit("/", 1)[0] + f"/{save_path_suffix}"
    out = _replace_flag(out, "--save-path", new_save_path)
    return out, new_save_path


def newest_test_report(result_dir: Path) -> Optional[Path]:
    files = sorted(result_dir.glob("test_report*.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def parse_metrics(report_file: Path) -> Tuple[Optional[float], Optional[float]]:
    mse = None
    mae = None
    with open(report_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            metric_name = row[1]
            metric_value = row[2] if len(row) == 3 else row[3]
            metric_value = row[-1]
            try:
                value = float(metric_value)
            except ValueError:
                continue
            if metric_name == "mse_norm":
                mse = value
            elif metric_name == "mae_norm":
                mae = value
    return mse, mae


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DUET local branch on all datasets")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to DUET repo root",
    )
    parser.add_argument(
        "--save-path-suffix",
        type=str,
        default="DUET_localbranch_all",
        help="Suffix used in --save-path for generated runs",
    )
    parser.add_argument("--local-kernel-size", type=int, default=5)
    parser.add_argument("--local-alpha", type=float, default=0.1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help='Comma-separated GPU ids to schedule jobs on, e.g. "0,1,2,3"',
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum concurrent training jobs",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("result") / "duet_localbranch_all_summary.csv",
        help="Summary table output path (relative to repo root by default)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs when a matching result report already exists in target save path",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)

    specs = parse_duet_scripts(repo_root)
    if not specs:
        print("No DUET commands found for horizons 96/192/336/720.", file=sys.stderr)
        return 1

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    run_items: List[Tuple[RunSpec, List[str], str]] = []
    results: List[Dict[str, object]] = []

    for spec in specs:
        try:
            cmd, new_save_path = apply_local_branch(
                spec.base_command,
                args.save_path_suffix,
                args.local_kernel_size,
                args.local_alpha,
            )
        except Exception as e:
            results.append(
                {
                    "dataset": spec.dataset,
                    "horizon": spec.horizon,
                    "mse_norm": "",
                    "mae_norm": "",
                    "status": f"config_error: {e}",
                    "report_file": "",
                }
            )
            continue

        if args.dry_run:
            print(f"[DRY-RUN] {spec.dataset} horizon={spec.horizon}")
            print(" ".join(shlex.quote(t) for t in cmd))
            results.append(
                {
                    "dataset": spec.dataset,
                    "horizon": spec.horizon,
                    "mse_norm": "",
                    "mae_norm": "",
                    "status": "dry_run",
                    "report_file": "",
                }
            )
            continue

        if args.skip_existing:
            dataset_dir = spec.dataset.replace(".csv", "")
            report_dir = repo_root / "result" / dataset_dir / args.save_path_suffix
            report_file = newest_test_report(report_dir)
            if report_file is not None:
                mse, mae = parse_metrics(report_file)
                print(
                    f"[SKIP] {spec.dataset} horizon={spec.horizon} "
                    f"(existing: {report_file.relative_to(repo_root)})"
                )
                results.append(
                    {
                        "dataset": spec.dataset,
                        "horizon": spec.horizon,
                        "mse_norm": "" if mse is None else mse,
                        "mae_norm": "" if mae is None else mae,
                        "status": "skipped_existing",
                        "report_file": str(report_file.relative_to(repo_root)),
                    }
                )
                continue

        run_items.append((spec, cmd, new_save_path))

    if not args.dry_run and run_items:
        semaphore = Lock()
        gpu_cursor = {"idx": 0}

        def run_one(item: Tuple[RunSpec, List[str], str]) -> Dict[str, object]:
            spec, cmd, _new_save_path = item
            with semaphore:
                gpu = gpu_ids[gpu_cursor["idx"] % len(gpu_ids)]
                gpu_cursor["idx"] += 1

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            cmd2 = _replace_flag(cmd, "--gpus", "0")

            print(f"[RUN] {spec.dataset} horizon={spec.horizon} gpu={gpu}")
            print(" ".join(shlex.quote(t) for t in cmd2))

            proc = subprocess.run(cmd2, cwd=repo_root, env=env)
            if proc.returncode != 0:
                return {
                    "dataset": spec.dataset,
                    "horizon": spec.horizon,
                    "mse_norm": "",
                    "mae_norm": "",
                    "status": f"failed({proc.returncode})",
                    "report_file": "",
                }

            dataset_dir = spec.dataset.replace(".csv", "")
            report_dir = repo_root / "result" / dataset_dir / args.save_path_suffix
            report_file = newest_test_report(report_dir)
            if not report_file:
                return {
                    "dataset": spec.dataset,
                    "horizon": spec.horizon,
                    "mse_norm": "",
                    "mae_norm": "",
                    "status": "no_report_found",
                    "report_file": str(report_dir),
                }

            mse, mae = parse_metrics(report_file)
            return {
                "dataset": spec.dataset,
                "horizon": spec.horizon,
                "mse_norm": "" if mse is None else mse,
                "mae_norm": "" if mae is None else mae,
                "status": "ok",
                "report_file": str(report_file.relative_to(repo_root)),
            }

        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            futures = [ex.submit(run_one, item) for item in run_items]
            for fut in as_completed(futures):
                results.append(fut.result())

    output_csv = args.output_csv
    if not output_csv.is_absolute():
        output_csv = repo_root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "horizon", "mse_norm", "mae_norm", "status", "report_file"]
        )
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r["dataset"], int(r["horizon"]))):
            writer.writerow(row)

    print(f"\nSaved summary: {output_csv}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
