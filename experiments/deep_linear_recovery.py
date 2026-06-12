"""Deep linear matrix-recovery simulation for the invariant-dynamics theorem.

This script runs the full-observation setting from the theory section:

    min_Theta 1/2 ||W_L ... W_1 - Phi||_F^2,

where Phi is low rank and every W_l is initialized as an epsilon-scaled
orthogonal matrix. It then constructs the fixed compressed bases from the
theorem, trains the 2r x 2r compressed problem, and logs:

1. singular values of one full factor during GD;
2. full GD loss vs. compressed GD loss;
3. end-to-end discrepancy between full and compressed trajectories.

The generated figures are used for the theorem-level appendix figures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, qr, svd


@dataclass(frozen=True)
class Config:
    d: int = 24
    rank: int = 2
    depth: int = 3
    eps: float = 0.25
    eta: float = 0.08
    steps: int = 1800
    seed: int = 7
    log_every: int = 10


def orthogonal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    q, r = qr(rng.standard_normal((d, d)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def make_low_rank_target(d: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    u = orthogonal_matrix(d, rng)[:, :rank]
    v = orthogonal_matrix(d, rng)[:, :rank]
    singular_values = np.linspace(1.5, 0.8, rank)
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


def gradients(weights: list[np.ndarray], target: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    residual = forward(weights) - target
    prefix, suffix = prefix_suffix(weights)
    grads = []
    for layer in range(len(weights)):
        grads.append(suffix[layer + 1].T @ residual @ prefix[layer].T)
    return grads, residual


def nullspace_basis(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    _, singular_values, vt = svd(matrix)
    scale = singular_values[0] if singular_values.size else 1.0
    rank = int(np.sum(singular_values > tol * max(scale, 1.0)))
    return vt[rank:].T


def intersection_basis(a: np.ndarray, b: np.ndarray, dim: int) -> np.ndarray:
    stacked = np.vstack([a, b])
    basis = nullspace_basis(stacked)
    if basis.shape[1] < dim:
        raise RuntimeError(f"Expected at least {dim} intersection directions, got {basis.shape[1]}.")
    return basis[:, :dim]


def complete_basis(scalar_basis: np.ndarray) -> np.ndarray:
    d = scalar_basis.shape[0]
    complement_projector = np.eye(d) - scalar_basis @ scalar_basis.T
    u, singular_values, _ = svd(complement_projector)
    active_dim = int(np.sum(singular_values > 1e-8))
    return np.hstack([u[:, :active_dim], scalar_basis])


def build_theorem_bases(
    weights0: list[np.ndarray],
    target: np.ndarray,
    rank: int,
    eps: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    d = target.shape[0]
    scalar_dim = d - 2 * rank
    psi = target
    for weight in reversed(weights0[1:]):
        psi = weight.T @ psi

    scalar_basis = intersection_basis(psi, psi.T @ weights0[0], scalar_dim)
    v_bases = [complete_basis(scalar_basis)]
    u_bases = []
    for layer, weight in enumerate(weights0):
        u_basis = weight @ v_bases[layer] / eps
        u_bases.append(u_basis)
        if layer < len(weights0) - 1:
            v_bases.append(u_basis)

    compressed = []
    for layer, weight in enumerate(weights0):
        compressed.append(u_bases[layer][:, : 2 * rank].T @ weight @ v_bases[layer][:, : 2 * rank])
    return compressed, u_bases, v_bases


def compressed_gradients(tilde: list[np.ndarray], target_core: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    residual = forward(tilde) - target_core
    prefix, suffix = prefix_suffix(tilde)
    grads = []
    for layer in range(len(tilde)):
        grads.append(suffix[layer + 1].T @ residual @ prefix[layer].T)
    return grads, residual


def run(cfg: Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    target = make_low_rank_target(cfg.d, cfg.rank, rng)
    weights = [cfg.eps * orthogonal_matrix(cfg.d, rng) for _ in range(cfg.depth)]
    weights0 = [weight.copy() for weight in weights]
    tilde, u_bases, v_bases = build_theorem_bases(weights0, target, cfg.rank, cfg.eps)
    target_core = u_bases[-1][:, : 2 * cfg.rank].T @ target @ v_bases[0][:, : 2 * cfg.rank]

    logs = {
        "iteration": [],
        "full_loss": [],
        "compressed_loss": [],
        "discrepancy": [],
        "singular_values": [],
    }

    for step in range(cfg.steps + 1):
        if step % cfg.log_every == 0 or step == cfg.steps:
            full_map = forward(weights)
            compressed_map = u_bases[-1][:, : 2 * cfg.rank] @ forward(tilde) @ v_bases[0][:, : 2 * cfg.rank].T
            compressed_residual = forward(tilde) - target_core
            logs["iteration"].append(step)
            logs["full_loss"].append(0.5 * norm(full_map - target) ** 2)
            logs["compressed_loss"].append(0.5 * norm(compressed_residual) ** 2)
            logs["discrepancy"].append(norm(full_map - compressed_map, "fro") ** 2)
            logs["singular_values"].append(svd(weights[0], compute_uv=False))

        if step == cfg.steps:
            break

        grads, _ = gradients(weights, target)
        for idx, grad in enumerate(grads):
            weights[idx] -= cfg.eta * grad

        compressed_grads, _ = compressed_gradients(tilde, target_core)
        for idx, grad in enumerate(compressed_grads):
            tilde[idx] -= cfg.eta * grad

    return {key: np.asarray(value) for key, value in logs.items()}


def write_summary(out_dir: Path, cfg: Config, logs: dict[str, np.ndarray]) -> None:
    lines = [
        f"d: {cfg.d}",
        f"rank: {cfg.rank}",
        f"depth: {cfg.depth}",
        f"eps: {cfg.eps}",
        f"eta: {cfg.eta}",
        f"steps: {cfg.steps}",
        f"seed: {cfg.seed}",
        f"final_full_loss: {logs['full_loss'][-1]:.6g}",
        f"final_compressed_loss: {logs['compressed_loss'][-1]:.6g}",
        f"final_discrepancy: {logs['discrepancy'][-1]:.6g}",
        f"max_discrepancy: {logs['discrepancy'].max():.6g}",
    ]
    (out_dir / "deep_linear_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_svd(out_dir: Path, cfg: Config, logs: dict[str, np.ndarray]) -> None:
    iterations = logs["iteration"]
    singular_values = logs["singular_values"]
    active_dim = 2 * cfg.rank

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for idx in range(singular_values.shape[1]):
        color = "#2563eb" if idx < active_dim else "#9ca3af"
        alpha = 0.95 if idx < active_dim else 0.32
        lw = 1.6 if idx < active_dim else 0.7
        ax.plot(iterations, singular_values[:, idx], color=color, alpha=alpha, lw=lw)
    ax.axhline(cfg.eps, color="#111111", linestyle="--", linewidth=0.8, alpha=0.75)
    ax.text(0.66, 0.10, f"initial scale epsilon={cfg.eps}", transform=ax.transAxes, fontsize=8)
    ax.set_title("Singular values of one factor during full GD")
    ax.set_xlabel("GD iteration")
    ax.set_ylabel("singular value")
    ax.grid(axis="y", color="0.90", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_svd_dynamics.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_svd_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def plot_compression(out_dir: Path, logs: dict[str, np.ndarray]) -> None:
    iterations = logs["iteration"]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax[0].semilogy(iterations, logs["full_loss"], color="#2563eb", lw=2.0, label="Full GD")
    ax[0].semilogy(iterations, logs["compressed_loss"], color="#f59e0b", lw=1.7, linestyle="--", label="Compressed GD")
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("GD iteration")
    ax[0].set_ylabel("loss")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(axis="y", color="0.90", linewidth=0.8)

    ax[1].semilogy(iterations, logs["discrepancy"], color="#7c3aed", lw=2.0)
    ax[1].set_title("Full vs. compressed map")
    ax[1].set_xlabel("GD iteration")
    ax[1].set_ylabel(r"$\|f(\Theta)-f_C(\widetilde{\Theta})\|_F^2$")
    ax[1].grid(axis="y", color="0.90", linewidth=0.8)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_dir / "fig4_compression.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig4_compression.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/deep_linear_recovery"))
    parser.add_argument("--d", type=int, default=24)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--write-report-figs", action="store_true", help="Also overwrite figs/fig3_* and figs/fig4_*.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        d=args.d,
        rank=args.rank,
        depth=args.depth,
        eps=args.eps,
        eta=args.eta,
        steps=args.steps,
        seed=args.seed,
        log_every=args.log_every,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = run(cfg)
    plot_svd(args.out_dir, cfg, logs)
    plot_compression(args.out_dir, logs)
    write_summary(args.out_dir, cfg, logs)
    if args.write_report_figs:
        figs_dir = Path("figs")
        figs_dir.mkdir(exist_ok=True)
        plot_svd(figs_dir, cfg, logs)
        plot_compression(figs_dir, logs)
    print(f"Wrote {args.out_dir / 'fig3_svd_dynamics.pdf'}")
    print(f"Wrote {args.out_dir / 'fig4_compression.pdf'}")


if __name__ == "__main__":
    main()
