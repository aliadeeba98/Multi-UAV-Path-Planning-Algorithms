#!/usr/bin/env python3
"""Combine per-algorithm CSVs from results/ into one file: results/all_algorithms_merged.csv"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# (label in merged file, filename)
ALGORITHMS = [
    ("Astar", "astar_results.csv"),
    ("Dstar", "dstar_results.csv"),
    ("Q_learning", "q_learning_results.csv"),
    ("DQN", "dqn_results.csv"),
    ("SAC", "sac_results.csv"),
    ("Hybrid_PSO_SAC", "hybrid_pso_sac_results.csv"),
]


def main() -> int:
    frames = []
    for algo, name in ALGORITHMS:
        path = os.path.join(RESULTS, name)
        if not os.path.isfile(path):
            print(f"skip (missing): {path}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "Algorithm", algo)
        frames.append(df)

    if not frames:
        print("No CSV files found under results/. Run the experiment scripts first.")
        return 1

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "all_algorithms_merged.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
