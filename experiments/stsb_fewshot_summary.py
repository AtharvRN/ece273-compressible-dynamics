"""Summarize local STS-B few-shot Deep LoRA runs.

This script does not rerun BERT fine-tuning. It records the local three-seed
STS-B runs used in the report and regenerates the compact summary figure.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAW_RESULTS = [
    (16, 2, 0, 0.560592),
    (16, 2, 1, 0.538381),
    (16, 2, 2, 0.609799),
    (16, 3, 0, 0.732153),
    (16, 3, 1, 0.572263),
    (16, 3, 2, 0.699606),
    (64, 2, 0, 0.737842),
    (64, 2, 1, 0.733233),
    (64, 2, 2, 0.730990),
    (64, 3, 0, 0.772468),
    (64, 3, 1, 0.787789),
    (64, 3, 2, 0.771477),
    (256, 2, 0, 0.822508),
    (256, 2, 1, 0.817835),
    (256, 2, 2, 0.798983),
    (256, 3, 0, 0.832642),
    (256, 3, 1, 0.838574),
    (256, 3, 2, 0.828955),
]


def aggregate() -> dict[tuple[int, int], tuple[float, float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for sample_count, depth, _seed, pearson in RAW_RESULTS:
        grouped[(sample_count, depth)].append(pearson)

    summary = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        summary[key] = (float(arr.mean()), float(sem))
    return summary


def write_summary(out_dir: Path, summary: dict[tuple[int, int], tuple[float, float]]) -> None:
    lines = [
        "sample_count,method_depth,mean_pearson,sem",
    ]
    for sample_count in sorted({key[0] for key in summary}):
        for depth in [2, 3]:
            mean, sem = summary[(sample_count, depth)]
            lines.append(f"{sample_count},{depth},{mean:.6f},{sem:.6f}")
    out_path = out_dir / "stsb_fewshot_summary.csv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot(out_dir: Path, summary: dict[tuple[int, int], tuple[float, float]]) -> None:
    samples = np.array(sorted({key[0] for key in summary}))
    vanilla = np.array([summary[(int(s), 2)][0] for s in samples])
    vanilla_sem = np.array([summary[(int(s), 2)][1] for s in samples])
    deep = np.array([summary[(int(s), 3)][0] for s in samples])
    deep_sem = np.array([summary[(int(s), 3)][1] for s in samples])
    gains = deep - vanilla

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1.55, 1.0]})
    axes[0].errorbar(samples, vanilla, yerr=vanilla_sem, marker="o", lw=2.2, capsize=3, color="#f59e0b", label="Vanilla LoRA")
    axes[0].errorbar(samples, deep, yerr=deep_sem, marker="o", lw=2.2, capsize=3, color="#2563eb", label="Deep LoRA")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(samples, labels=[str(s) for s in samples])
    axes[0].set_ylim(0.53, 0.86)
    axes[0].set_xlabel("# training examples")
    axes[0].set_ylabel("Pearson correlation")
    axes[0].set_title("STS-B few-shot fine-tuning")
    axes[0].grid(axis="y", color="0.90", linewidth=0.8)
    axes[0].legend(frameon=False, loc="lower right")

    bars = axes[1].bar(np.arange(len(samples)), gains, width=0.62, color="#2563eb", alpha=0.9)
    axes[1].axhline(0, color="0.25", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(samples)), labels=[str(s) for s in samples])
    axes[1].set_ylim(0, 0.115)
    axes[1].set_xlabel("# training examples")
    axes[1].set_ylabel("Pearson gain")
    axes[1].set_title("Deep LoRA - Vanilla LoRA")
    axes[1].grid(axis="y", color="0.90", linewidth=0.8)
    for bar, gain in zip(bars, gains):
        axes[1].text(bar.get_x() + bar.get_width() / 2, gain + 0.004, f"+{gain:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_dir / "stsb_fewshot.png", bbox_inches="tight")
    fig.savefig(out_dir / "stsb_fewshot.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/stsb_fewshot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate()
    write_summary(args.out_dir, summary)
    plot(args.out_dir, summary)
    print(f"Wrote {args.out_dir / 'stsb_fewshot.png'}")


if __name__ == "__main__":
    main()
