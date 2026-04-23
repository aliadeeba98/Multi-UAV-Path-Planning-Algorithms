#!/usr/bin/env python3
"""SAC multi-UAV architecture figure. y-bands: training (0-0.9) | critic (1.0-1.4) | actor (1.6-2.4)."""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 7})
E = {"color": "#555", "lw": 0.18, "alpha": 0.17, "zorder": 1, "clip_on": True, "solid_capstyle": "round"}
S = {**E, "alpha": 0.13, "lw": 0.14}
R = 0.06


def ly(n, c, g):
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([c])
    s = (n - 1) * g
    return np.linspace(c + s / 2, c - s / 2, n)


def fc(ax, x0, x1, a, b, **k):
    for u in a:
        for v in b:
            ax.plot((x0, x1), (u, v), **k)


def nodes(ax, x, ys, col):
    for y in ys:
        ax.add_patch(Circle((x, y), R, facecolor=col, edgecolor="#222", lw=0.4, zorder=4))


def twin(ax, x_in, xs, xo, n_in, n_h, yu, yl, g, c0, c1):
    yc = 0.5 * (yu + yl)
    y0 = ly(n_in, yc, g)
    a1, b1 = ly(n_h, yu, g), ly(n_h, yl, g)
    a2, b2, a3, b3 = ly(n_h, yu, g), ly(n_h, yl, g), ly(n_h, yu, g), ly(n_h, yl, g)
    xa, xb = xs[0] - 0.08, xs[0] + 0.08
    for y in y0:
        for u in a1:
            ax.plot((x_in, xa), (y, u), **S)
        for u in b1:
            ax.plot((x_in, xb), (y, u), **S)
    fc(ax, xa, xs[1], a1, a2, **S)
    fc(ax, xb, xs[1], b1, b2, **S)
    fc(ax, xs[1], xs[2], a2, a3, **S)
    fc(ax, xs[1], xs[2], b2, b3, **S)
    for u in a3:
        ax.plot((xs[2], xo), (u, yu), **S)
    for u in b3:
        ax.plot((xs[2], xo), (u, yl), **S)
    nodes(ax, x_in, y0, c0)
    for x, arr, c in ((xa, a1, c1), (xb, b1, c1), (xs[1], a2, c1), (xs[1], b2, c1), (xs[2], a3, c1), (xs[2], b3, c1)):
        nodes(ax, x, arr, c)
    for i, (x, y) in enumerate(((xo, yu), (xo, yl))):
        ax.add_patch(Circle((x, y), R * 0.9, facecolor="#5c6bc0", edgecolor="#1a237e", lw=0.35, zorder=5))
        q = r"$Q_1(s,a)$" if i == 0 else r"$Q_2(s,a)$"
        ax.text(x + 0.3, y, q, fontsize=2.0, ha="left", va="center", zorder=6)
    return yu, yl, xo, xo


def build_figure():
    W, H = 20.0, 3.0
    fig, ax = plt.subplots(figsize=(14, 2.2), dpi=150)
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")
    ax.set_aspect("auto")
    g = 0.09

    # Training 0-0.55
    ax.add_patch(Rectangle((0.1, 0.02), W - 0.2, 0.48, facecolor="#fff3e0", edgecolor="#e65100", lw=0.4, ls="--", zorder=0))
    ax.text(10, 0.42, "TRAINING (SAC)", ha="center", fontsize=3, fontweight="bold", color="#bf360c", zorder=3)
    n, bw, gs = 6, 0.75, 0.12
    tot, x0 = n * bw + (n - 1) * gs, (W - (n * bw + (n - 1) * gs)) * 0.5
    yb = 0.2
    for i, t in enumerate(["1.Collect", "2.Buf", "3.Batch", "4a.Crit", "4b.Act", "5.Upd"]):
        x = x0 + i * (bw + gs)
        ax.add_patch(
            FancyBboxPatch(
                (x, yb - 0.04), bw, 0.1, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#ffe0b2", edgecolor="#e65100", lw=0.25, zorder=1
            )
        )
        ax.text(x + bw / 2, yb + 0.02, t, ha="center", va="center", fontsize=1.0, zorder=2)
    ax.text(W - 0.1, 0.18, r"6.$\circlearrowright$1", ha="right", fontsize=2.0, color="#bf360c", zorder=3)

    # Critic: twin in y 0.55-0.95
    yu, yl = 0.88, 0.58
    x_in, xs, xo = 0.4, [0.65, 1.0, 1.35], 1.5
    ax.add_patch(FancyBboxPatch((0.05, 0.52), 0.35, 0.1, boxstyle="round,pad=0.01", facecolor="#f3e5f5", edgecolor="#7b1fa2", ls="--", lw=0.3, zorder=2))
    ax.text(0.22, 0.57, "In", ha="center", fontsize=2.0, zorder=3)
    ax.add_patch(FancyBboxPatch((0.45, 0.52), 0.9, 0.1, boxstyle="round,pad=0.01", facecolor="#e3f2fd", edgecolor="#1565c0", lw=0.3, zorder=0))
    ax.text(0.9, 1.02, "SAC CRITIC (twin $Q$)", ha="center", fontsize=2.0, fontweight="bold", color="#0d47a1", zorder=3)
    twin(ax, x_in, xs, xo, 3, 2, yu, yl, g, "#ba68c8", "#64b5f6")
    ax.add_patch(FancyBboxPatch((1.5, 0.52), 0.55, 0.08, boxstyle="round,pad=0.01", facecolor="#ede7f6", edgecolor="#5e35b1", ls="--", lw=0.2, zorder=2))
    ax.text(1.8, 0.62, "Target", ha="center", fontsize=1.0, zorder=3)
    y_mid = 0.5 * (yu + yl)
    ax.annotate("", xy=(x_in - 0.03, y_mid), xytext=(0.48, 0.55), arrowprops=dict(arrowstyle="->", color="#7b1fa2", lw=0.2, mutation_scale=1))

    fig.suptitle("Soft Actor-Critic (SAC) — multi-UAV path planning", fontsize=8, y=0.99)
    return fig, ax


def main():
    fig, _ = build_figure()
    out = "/root/custody-deploy/Multi-UAV-Path-Planning-Algorithms/figures/sac_uav_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    print("Wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
