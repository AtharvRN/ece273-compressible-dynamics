"""Run the few-shot STS-B LoRA vs. Deep LoRA experiment.

This script fine-tunes BERT-base-cased on small STS-B training subsets and
compares a two-factor LoRA adapter against a deeper compressed LoRA product.
It is intentionally compact and self-contained; it does not depend on PEFT.

Example:
    python experiments/deep_lora_stsb.py --samples 16 64 256 --seeds 0 1 2

The full default sweep performs 18 fine-tuning runs and may take a while
without a GPU. Use smaller --samples/--seeds/--steps values for a smoke test.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    sample_count: int
    depth: int
    seed: int
    rank: int
    alpha: float
    init_scale: float
    lr: float
    batch_size: int
    max_length: int
    steps: int
    target_modules: tuple[str, ...]


class DeepLoRALinear:
    """Factory namespace; actual class is created after torch is imported."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def pearson_corr(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float)
    predictions = predictions - predictions.mean()
    labels = labels - labels.mean()
    denom = np.linalg.norm(predictions) * np.linalg.norm(labels)
    if denom == 0:
        return 0.0
    return float(predictions.dot(labels) / denom)


def parse_target_modules(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def make_lora_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _DeepLoRALinear(nn.Module):
        def __init__(
            self,
            base: nn.Linear,
            rank: int,
            depth: int,
            alpha: float,
            init_scale: float,
        ) -> None:
            super().__init__()
            if depth < 2:
                raise ValueError("LoRA depth must be at least 2.")
            self.base = base
            self.rank = rank
            self.depth = depth
            self.scaling = alpha / rank
            for param in self.base.parameters():
                param.requires_grad = False

            self.a = nn.Parameter(init_scale * torch.randn(rank, base.in_features))
            self.middle = nn.ParameterList(
                [
                    nn.Parameter(torch.eye(rank) + init_scale * torch.randn(rank, rank))
                    for _ in range(depth - 2)
                ]
            )
            self.b = nn.Parameter(torch.zeros(base.out_features, rank))

        def forward(self, inputs):
            update = F.linear(inputs, self.a)
            for middle in self.middle:
                update = F.linear(update, middle)
            update = F.linear(update, self.b)
            return self.base(inputs) + self.scaling * update

    return _DeepLoRALinear


def replace_module(root, module_name: str, new_module) -> None:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora_adapters(model, cfg: RunConfig) -> int:
    import torch.nn as nn

    lora_class = make_lora_class()
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(target) for target in cfg.target_modules):
            replace_module(
                model,
                name,
                lora_class(
                    module,
                    rank=cfg.rank,
                    depth=cfg.depth,
                    alpha=cfg.alpha,
                    init_scale=cfg.init_scale,
                ),
            )
            replaced += 1
    return replaced


def prepare_data(cfg: RunConfig):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding

    set_seed(cfg.seed)
    dataset = load_dataset("glue", "stsb")
    train = dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.sample_count))
    validation = dataset["validation"]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            max_length=cfg.max_length,
            truncation=True,
        )

    train = train.map(tokenize, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
    validation = validation.map(tokenize, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
    train = train.rename_column("label", "labels")
    validation = validation.rename_column("label", "labels")
    train.set_format("torch")
    validation.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(validation, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)
    return train_loader, val_loader


def train_once(cfg: RunConfig) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_data(cfg)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=1,
        problem_type="regression",
    )
    for param in model.parameters():
        param.requires_grad = False

    replaced = inject_lora_adapters(model, cfg)
    if replaced == 0:
        raise RuntimeError("No target BERT linear layers were replaced.")

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    model.to(device)

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)
    model.train()

    step = 0
    while step < cfg.steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step >= cfg.steps:
                break

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            predictions.extend(output.logits.squeeze(-1).detach().cpu().numpy().tolist())
            labels.extend(batch["labels"].detach().cpu().numpy().tolist())
    return pearson_corr(np.asarray(predictions), np.asarray(labels))


def write_results(out_dir: Path, rows: list[dict[str, float | int]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stsb_results.csv"
    fieldnames = ["sample_count", "depth", "seed", "pearson"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize(rows: Iterable[dict[str, float | int]]) -> dict[tuple[int, int], tuple[float, float]]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        key = (int(row["sample_count"]), int(row["depth"]))
        grouped.setdefault(key, []).append(float(row["pearson"]))
    summary = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        summary[key] = (float(arr.mean()), float(sem))
    return summary


def plot_results(out_dir: Path, rows: list[dict[str, float | int]]) -> None:
    summary = summarize(rows)
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
    axes[0].set_xlabel("# training examples")
    axes[0].set_ylabel("Pearson correlation")
    axes[0].set_title("STS-B few-shot fine-tuning")
    axes[0].grid(axis="y", color="0.90", linewidth=0.8)
    axes[0].legend(frameon=False, loc="lower right")

    bars = axes[1].bar(np.arange(len(samples)), gains, width=0.62, color="#2563eb", alpha=0.9)
    axes[1].axhline(0, color="0.25", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(samples)), labels=[str(s) for s in samples])
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


def read_rows(path: Path) -> list[dict[str, float | int]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "sample_count": int(row["sample_count"]),
                    "depth": int(row["depth"]),
                    "seed": int(row["seed"]),
                    "pearson": float(row["pearson"]),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="bert-base-cased")
    parser.add_argument("--samples", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--init-scale", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--target-modules", default="query,key,value,attention.output.dense,intermediate.dense,output.dense")
    parser.add_argument("--out-dir", type=Path, default=Path("results/stsb_fewshot"))
    parser.add_argument("--plot-only", type=Path, help="Regenerate the figure from an existing CSV instead of training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = read_rows(args.plot_only)
    else:
        rows = []
        targets = parse_target_modules(args.target_modules)
        for sample_count in args.samples:
            for depth in args.depths:
                for seed in args.seeds:
                    cfg = RunConfig(
                        model_name=args.model_name,
                        sample_count=sample_count,
                        depth=depth,
                        seed=seed,
                        rank=args.rank,
                        alpha=args.alpha,
                        init_scale=args.init_scale,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        steps=args.steps,
                        target_modules=targets,
                    )
                    pearson = train_once(cfg)
                    row = {
                        "sample_count": sample_count,
                        "depth": depth,
                        "seed": seed,
                        "pearson": pearson,
                    }
                    rows.append(row)
                    print(row)
        write_results(args.out_dir, rows)

    plot_results(args.out_dir, rows)
    print(f"Wrote {args.out_dir / 'stsb_fewshot.png'}")


if __name__ == "__main__":
    main()
