"""Plot one run or compare arbitrary experiment runs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The host home directory may be read-only in containers or CI.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "cifar10-resnet-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_METRICS = {"train_acc", "val_acc", "test_acc"}


@dataclass
class Run:
    path: Path
    config: dict[str, Any]
    metrics: pd.DataFrame
    summary: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.config.get("run_id", self.path.name))

    @property
    def display_name(self) -> str:
        return str(self.config.get("display_name", self.run_id))

    @property
    def variant(self) -> str:
        return str(self.config.get("variant", self.display_name))

    @property
    def seed(self) -> int | str:
        return self.config.get("seed", "unknown")


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return dict(default or {})
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_run(path: str | Path) -> Run:
    path = Path(path).expanduser().resolve()
    config_path = path / "config.json"
    metrics_path = path / "metrics.csv"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"{path} is not a run directory: config.json or metrics.csv is missing"
        )

    config = read_json(config_path)
    metrics = pd.read_csv(metrics_path).sort_values("epoch")
    summary = read_json(path / "summary.json")
    return Run(path, config, metrics, summary)


def discover_run_paths(
    inputs: list[str],
    allow_empty: bool = False,
    require_metrics: bool = True,
) -> list[Path]:
    discovered: list[Path] = []
    for value in inputs:
        matches = [Path(item) for item in glob.glob(value)]
        if not matches and Path(value).exists():
            matches = [Path(value)]
        for match in matches:
            if (match / "config.json").exists() and (
                not require_metrics or (match / "metrics.csv").exists()
            ):
                discovered.append(match.resolve())
            elif match.is_dir():
                discovered.extend(
                    child.resolve()
                    for child in match.iterdir()
                    if child.is_dir()
                    and (child / "config.json").exists()
                    and (not require_metrics or (child / "metrics.csv").exists())
                )

    unique = list(dict.fromkeys(sorted(discovered)))
    if not unique and not allow_empty:
        raise FileNotFoundError("No valid run directories were found")
    return unique


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def metric_style(metric: str) -> tuple[float, str]:
    if metric in ACCURACY_METRICS or metric.endswith("_acc"):
        return 100.0, "Accuracy (%)"
    if metric.endswith("_loss"):
        return 1.0, "Loss"
    if metric == "lr":
        return 1.0, "Learning rate"
    if metric.endswith("_time_s"):
        return 1.0, "Time (seconds)"
    if metric.endswith("_memory_mb"):
        return 1.0, "Memory (MiB)"
    return 1.0, metric.replace("_", " ").title()


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_stem.with_suffix('.png')}")
    print(f"Saved: {output_stem.with_suffix('.svg')}")


def plot_single_run(run: Run, window: int) -> None:
    required = {"train_loss", "val_loss", "train_acc", "val_acc", "lr"}
    missing = sorted(required - set(run.metrics.columns))
    if missing:
        raise ValueError(f"Run {run.run_id} is missing metrics: {', '.join(missing)}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    epochs = run.metrics["epoch"]

    for column, label, color in (
        ("train_loss", "Train", 'r'),
        ("val_loss", "Validation", 'b'),
    ):
        axes[0, 0].plot(
            epochs,
            smooth(run.metrics[column], window),
            label=label,
            color=color,
        )
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    for column, label, color in (
        ("train_acc", "Train", 'r'),
        ("val_acc", "Validation", 'b'),
    ):
        axes[0, 1].plot(
            epochs,
            smooth(run.metrics[column], window) * 100,
            label=label,
            color=color,
        )
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, run.metrics["lr"], color='k')
    axes[1, 0].set_title("Learning rate")
    axes[1, 0].set_ylabel("Learning rate")

    axes[1, 1].plot(
        epochs,
        run.metrics["epoch_time_s"],
        color='k',
        label="Epoch time",
    )
    axes[1, 1].set_title("Training time")
    axes[1, 1].set_ylabel("Seconds")

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)

    fig.suptitle(f"{run.display_name}\n{run.run_id}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, run.path / "figures" / "learning_curves")


def comparison_rows(runs: list[Run]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run.run_id,
                "experiment": run.config.get("experiment"),
                "variant": run.variant,
                "display_name": run.display_name,
                "seed": run.seed,
                "model": run.config.get("model", {}).get("name"),
                "base_channels": run.config.get("model", {}).get("base_channels"),
                "status": run.summary.get("status"),
                "best_epoch": run.summary.get("best_epoch"),
                "best_val_acc": run.summary.get("best_val_acc"),
                "test_acc": run.summary.get("test_acc"),
                "num_parameters": run.summary.get("num_parameters"),
                "training_time_s": run.summary.get("training_time_s"),
            }
        )
    return rows


def write_comparison_csv(runs: list[Run], output_dir: Path) -> None:
    rows = comparison_rows(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def plot_comparison(
    runs: list[Run],
    metric: str,
    output_dir: Path,
    window: int,
    aggregate_seeds: bool,
    title: str | None,
) -> None:
    for run in runs:
        if metric not in run.metrics.columns:
            raise ValueError(f"Run {run.run_id} does not contain metric {metric}")

    scale, ylabel = metric_style(metric)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    if aggregate_seeds:
        groups: dict[str, list[Run]] = {}
        for run in runs:
            groups.setdefault(run.variant, []).append(run)

        for variant, group in sorted(groups.items()):
            frames = [
                run.metrics[["epoch", metric]].assign(seed=str(run.seed))
                for run in group
            ]
            combined = pd.concat(frames, ignore_index=True)
            stats = combined.groupby("epoch")[metric].agg(["mean", "std", "count"])
            mean = smooth(stats["mean"], window) * scale
            label = group[0].display_name
            x_values = stats.index.to_numpy()
            mean_values = mean.to_numpy()
            ax.plot(x_values, mean_values, linewidth=2, label=f"{label} (n={len(group)})")
            if len(group) > 1:
                std_values = (stats["std"].fillna(0) * scale).to_numpy()
                ax.fill_between(
                    x_values,
                    mean_values - std_values,
                    mean_values + std_values,
                    alpha=0.18,
                )
    else:
        for run in runs:
            values = smooth(run.metrics[metric], window) * scale
            label = f"{run.display_name} [seed={run.seed}]"
            ax.plot(run.metrics["epoch"], values, linewidth=2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title or metric.replace("_", " ").title())
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    suffix = "_mean_std" if aggregate_seeds else ""
    save_figure(fig, output_dir / f"{metric}{suffix}")
    write_comparison_csv(runs, output_dir)


def list_runs(paths: list[Path], experiment: str | None) -> None:
    headers = ("run_id", "experiment", "variant", "seed", "status", "best_val")
    print(" | ".join(headers))
    print("-" * 110)
    for path in paths:
        config = read_json(path / "config.json")
        if experiment and config.get("experiment") != experiment:
            continue
        summary = read_json(path / "summary.json")
        best = summary.get("best_val_acc")
        best_text = f"{best * 100:.2f}%" if isinstance(best, (float, int)) else "-"
        values = (
            config.get("run_id", path.name),
            config.get("experiment", "-"),
            config.get("variant", "-"),
            str(config.get("seed", "-")),
            summary.get("status", "-"),
            best_text,
        )
        print(" | ".join(map(str, values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Plot all curves for one run")
    run_parser.add_argument("run_dir", type=Path)
    run_parser.add_argument("--smooth", type=int, default=1)

    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs")
    compare_parser.add_argument("--runs", nargs="+", required=True)
    compare_parser.add_argument("--metric", default="val_acc")
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--smooth", type=int, default=1)
    compare_parser.add_argument("--aggregate-seeds", action="store_true")
    compare_parser.add_argument("--title")
    compare_parser.add_argument("--experiment")

    list_parser = subparsers.add_parser("list", help="List available runs")
    list_parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    list_parser.add_argument("--experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(args, "smooth") and args.smooth <= 0:
        raise ValueError("--smooth must be positive")

    if args.command == "run":
        plot_single_run(load_run(args.run_dir), args.smooth)
        return

    if args.command == "list":
        paths = discover_run_paths(
            [args.runs_dir],
            allow_empty=True,
            require_metrics=False,
        )
        list_runs(paths, args.experiment)
        return

    paths = discover_run_paths(args.runs)
    runs = [load_run(path) for path in paths]
    if args.experiment:
        runs = [run for run in runs if run.config.get("experiment") == args.experiment]
    if not runs:
        raise ValueError("No runs remain after applying the experiment filter")

    output_dir = args.output.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    plot_comparison(
        runs,
        args.metric,
        output_dir,
        args.smooth,
        args.aggregate_seeds,
        args.title,
    )


if __name__ == "__main__":
    main()
