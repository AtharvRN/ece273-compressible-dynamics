"""Synthetic deep matrix-completion experiment.

This is the main self-contained reproduction script for the matrix-completion
application in the report. It compares:

1. Full gradient descent on depth-L square factors.
2. Compressed gradient descent with gamma>0 basis updates.
3. Compressed gradient descent with gamma=0, the no-basis-update ablation.

The default settings reproduce the figure reported in the write-up:
d=200, rank=3, L=3, 20% observed entries, 5000 steps, 5 seeds.
Use --quick for a small smoke test.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, qr, svd


@dataclass(frozen=True)
class Config:
    d: int = 200
    rank: int = 3
    depth: int = 3
    eps: float = 0.1
    eta: float = 0.20
    gamma: float = 10.0
    obs_prob: float = 0.20
    steps: int = 5000
    seeds: int = 5
    seed0: int = 0


def orthogonal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = qr(rng.standard_normal((d, d)))
    return q


def scaled_orthogonal_init(d: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    return eps * orthogonal_matrix(d, rng)


def make_low_rank_target(d: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    u = orthogonal_matrix(d, rng)[:, :rank]
    v = orthogonal_matrix(d, rng)[:, :rank]
    singular_values = 1.0 + rng.random(rank)
    return u @ np.diag(singular_values) @ v.T


def forward(weights: list[np.ndarray]) -> np.ndarray:
    output = weights[0]
    for weight in weights[1:]:
        output = weight @ output
    return output


def prefix_suffix(weights: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    d = weights[0].shape[0]
    prefix = [np.eye(d)]
    for weight in weights:
        prefix.append(weight @ prefix[-1])

    suffix = [np.eye(d)]
    for weight in reversed(weights):
        suffix.append(suffix[-1] @ weight)
    suffix.reverse()
    return prefix, suffix


def gradients(
    weights: list[np.ndarray],
    phi: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    residual = mask * (forward(weights) - phi)
    prefix, suffix = prefix_suffix(weights)
    grads = []
    for layer in range(len(weights)):
        grads.append(suffix[layer + 1].T @ residual @ prefix[layer].T)
    return grads, residual


def nullspace_basis(matrix: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    _, singular_values, vt = svd(matrix)
    cutoff = tol * max(1.0, singular_values[0] if singular_values.size else 1.0)
    rank = int(np.sum(singular_values > cutoff))
    return vt[rank:].T


def build_compressed_initialization(
    weights0: list[np.ndarray],
    phi: np.ndarray,
    mask: np.ndarray,
    rank: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Build the compressed factors from the masked initial gradient."""
    d = weights0[0].shape[0]
    g1 = gradients(weights0, phi, mask)[0][0]

    n1 = nullspace_basis(g1)
    n2 = nullspace_basis(g1.T @ weights0[0])
    p1 = n1 @ n1.T if n1.size else np.zeros((d, d))
    p2 = n2 @ n2.T if n2.size else np.zeros((d, d))

    _, singular_values, vt = svd((np.eye(d) - p1) + (np.eye(d) - p2))
    intersection_dim = int(np.sum(singular_values < 1e-8))
    scalar_dim = d - 2 * rank
    scalar_basis = vt[d - intersection_dim :].T[:, :scalar_dim]

    complement = np.eye(d) - scalar_basis @ scalar_basis.T
    u_complement, _, _ = svd(complement)
    v_basis = np.hstack([u_complement[:, : 2 * rank], scalar_basis])

    eps = [np.sqrt(np.trace(w.T @ w) / d) for w in weights0]
    vs: list[np.ndarray | None] = [None] * len(weights0)
    us: list[np.ndarray | None] = [None] * len(weights0)
    vs[0] = v_basis
    for layer in range(len(weights0) - 1):
        us[layer] = weights0[layer] @ vs[layer] / eps[layer]  # type: ignore[index]
        vs[layer + 1] = us[layer]
    us[-1] = weights0[-1] @ vs[-1] / eps[-1]  # type: ignore[index]

    ul1 = us[-1][:, : 2 * rank]  # type: ignore[index]
    v11 = vs[0][:, : 2 * rank]  # type: ignore[index]
    tilde = []
    for layer in range(len(weights0)):
        ul = us[layer][:, : 2 * rank]  # type: ignore[index]
        vl = vs[layer][:, : 2 * rank]  # type: ignore[index]
        tilde.append(ul.T @ weights0[layer] @ vl)
    return tilde, ul1, v11


def compressed_gradients(
    tilde: list[np.ndarray],
    phi: np.ndarray,
    mask: np.ndarray,
    ul1: np.ndarray,
    v11: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    core = forward(tilde)
    full = ul1 @ core @ v11.T
    residual = mask * (full - phi)
    core_residual = ul1.T @ residual @ v11
    prefix, suffix = prefix_suffix(tilde)
    grads = []
    for layer in range(len(tilde)):
        grads.append(suffix[layer + 1].T @ core_residual @ prefix[layer].T)
    grad_u = residual @ v11 @ core.T
    grad_v = residual.T @ ul1 @ core
    return grads, grad_u, grad_v, residual


def run_full(
    weights0: list[np.ndarray],
    phi: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, float]:
    weights = [w.copy() for w in weights0]
    losses = np.empty(cfg.steps)
    start = time.perf_counter()
    for step in range(cfg.steps):
        grads, residual = gradients(weights, phi, mask)
        losses[step] = 0.5 * norm(residual) ** 2
        for idx, grad in enumerate(grads):
            weights[idx] -= cfg.eta * grad
    return losses, time.perf_counter() - start


def run_compressed(
    tilde0: list[np.ndarray],
    ul10: np.ndarray,
    v110: np.ndarray,
    phi: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
    gamma: float,
) -> tuple[np.ndarray, float]:
    tilde = [w.copy() for w in tilde0]
    ul1 = ul10.copy()
    v11 = v110.copy()
    losses = np.empty(cfg.steps)
    update_bases = gamma > 0
    start = time.perf_counter()
    for step in range(cfg.steps):
        grads, grad_u, grad_v, residual = compressed_gradients(tilde, phi, mask, ul1, v11)
        losses[step] = 0.5 * norm(residual) ** 2
        for idx, grad in enumerate(grads):
            tilde[idx] -= cfg.eta * grad
        if update_bases:
            ul1 -= gamma * cfg.eta * grad_u
            v11 -= gamma * cfg.eta * grad_v
    return losses, time.perf_counter() - start


def run_seed(seed: int, cfg: Config) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    phi = make_low_rank_target(cfg.d, cfg.rank, rng)
    mask = (rng.random((cfg.d, cfg.d)) < cfg.obs_prob).astype(float)
    weights0 = [scaled_orthogonal_init(cfg.d, cfg.eps, rng) for _ in range(cfg.depth)]
    loss_scale = max(0.5 * norm(mask * phi) ** 2, 1e-12)

    full_losses, full_time = run_full(weights0, phi, mask, cfg)
    tilde0, ul10, v110 = build_compressed_initialization(weights0, phi, mask, cfg.rank)
    basis_losses, basis_time = run_compressed(tilde0, ul10, v110, phi, mask, cfg, cfg.gamma)
    fixed_losses, fixed_time = run_compressed(tilde0, ul10, v110, phi, mask, cfg, 0.0)

    return {
        "full_losses": full_losses / loss_scale,
        "basis_losses": basis_losses / loss_scale,
        "fixed_losses": fixed_losses / loss_scale,
        "full_time": full_time,
        "basis_time": basis_time,
        "fixed_time": fixed_time,
    }


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] == 1:
        sem = np.zeros_like(mean)
    else:
        sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, sem


def write_summary(out_dir: Path, cfg: Config, runs: list[dict[str, np.ndarray | float]]) -> None:
    full_final = np.array([run["full_losses"][-1] for run in runs], dtype=float)
    basis_final = np.array([run["basis_losses"][-1] for run in runs], dtype=float)
    fixed_final = np.array([run["fixed_losses"][-1] for run in runs], dtype=float)
    full_time = np.array([run["full_time"] for run in runs], dtype=float)
    basis_time = np.array([run["basis_time"] for run in runs], dtype=float)
    fixed_time = np.array([run["fixed_time"] for run in runs], dtype=float)

    lines = [
        f"d: {cfg.d}",
        f"rank: {cfg.rank}",
        f"depth: {cfg.depth}",
        f"observed_fraction: {cfg.obs_prob}",
        f"steps: {cfg.steps}",
        f"seeds: {cfg.seeds}",
        f"gamma: {cfg.gamma}",
        f"full_final_mean: {full_final.mean():.6g}",
        f"basis_update_final_mean: {basis_final.mean():.6g}",
        f"gamma0_final_mean: {fixed_final.mean():.6g}",
        f"full_time_mean: {full_time.mean():.6g}",
        f"basis_update_time_mean: {basis_time.mean():.6g}",
        f"gamma0_time_mean: {fixed_time.mean():.6g}",
        f"basis_update_speedup: {full_time.mean() / max(basis_time.mean(), 1e-12):.6g}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_results(out_dir: Path, runs: list[dict[str, np.ndarray | float]]) -> None:
    full = np.vstack([run["full_losses"] for run in runs])
    basis = np.vstack([run["basis_losses"] for run in runs])
    fixed = np.vstack([run["fixed_losses"] for run in runs])
    series = [
        ("Full GD", full, "#1f4aff"),
        ("Compressed GD (gamma > 0)", basis, "#28c7b7"),
        ("Compressed GD (gamma = 0)", fixed, "#f59e0b"),
    ]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    steps = np.arange(full.shape[1])
    for label, values, color in series:
        mean, sem = mean_sem(values)
        axes[0].semilogy(steps, mean, label=label, color=color, lw=2.0)
        axes[0].fill_between(steps, np.maximum(mean - sem, 1e-16), mean + sem, color=color, alpha=0.16)

    axes[0].set_title(f"Mean recovery error over {full.shape[0]} seeds")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("relative masked loss")
    axes[0].legend(frameon=False, fontsize=8)

    finals = [values[:, -1] for _, values, _ in series]
    labels = [label for label, _, _ in series]
    colors = [color for _, _, color in series]
    means = [float(v.mean()) for v in finals]
    sems = [float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0 for v in finals]
    axes[1].bar(labels, means, yerr=sems, color=colors, capsize=3)
    axes[1].set_yscale("log")
    axes[1].set_title("Final relative masked loss")
    axes[1].tick_params(axis="x", labelrotation=15)

    fig.tight_layout()
    fig.savefig(out_dir / "matrix_completion.png", bbox_inches="tight")
    fig.savefig(out_dir / "matrix_completion.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/matrix_completion"))
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--obs-prob", type=float, default=0.20)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.d = 60
        args.steps = 300
        args.seeds = 1
        args.out_dir = Path("results/matrix_completion_quick")

    cfg = Config(
        d=args.d,
        rank=args.rank,
        depth=args.depth,
        eps=args.eps,
        eta=args.eta,
        gamma=args.gamma,
        obs_prob=args.obs_prob,
        steps=args.steps,
        seeds=args.seeds,
        seed0=args.seed0,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = [run_seed(cfg.seed0 + offset, cfg) for offset in range(cfg.seeds)]
    plot_results(args.out_dir, runs)
    write_summary(args.out_dir, cfg, runs)
    print(f"Wrote {args.out_dir / 'matrix_completion.png'}")


if __name__ == "__main__":
    main()
