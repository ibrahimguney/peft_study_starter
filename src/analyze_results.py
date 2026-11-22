import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_all_results(results_dir: str) -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "**", "res_*.json"), recursive=True)):
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        flat = {k: v for k, v in d.items() if k not in ["classification_report", "confusion_matrix"]}
        flat["source_file"] = os.path.basename(fp)
        rows.append(flat)
    return pd.DataFrame(rows)


def summarize_multi_seed(df: pd.DataFrame, out_dir: str):
    metrics = ["accuracy", "macro_f1", "weighted_f1", "ece", "peak_vram_gb", "train_wall_time_sec", "samples_per_sec", "trainable_pct"]
    available = [m for m in metrics if m in df.columns]
    grouped = df.groupby(["task", "method"], dropna=False)[available].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    summary_df = grouped.reset_index()
    out_csv = os.path.join(out_dir, "multi_seed_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved multi-seed summary to: {out_csv}")
    return summary_df


def plot_pareto_frontier(df: pd.DataFrame, out_dir: str):
    if df.empty or "peak_vram_gb" not in df.columns:
        return
    os.makedirs(out_dir, exist_ok=True)
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        metric_col = "macro_f1" if task == "esg" else "accuracy"
        if metric_col not in sub.columns:
            continue
        agg = sub.groupby("method", as_index=False)[["peak_vram_gb", metric_col, "train_wall_time_sec"]].mean()
        plt.figure(figsize=(7, 4.5))
        plt.scatter(agg["peak_vram_gb"], agg[metric_col], s=90)
        for _, r in agg.iterrows():
            plt.annotate(
                f"{r['method']} ({r['train_wall_time_sec']:.0f}s)",
                (r["peak_vram_gb"], r[metric_col]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )
        plt.title(f"Pareto Frontier: Peak VRAM vs. {metric_col.upper()} ({task.upper()})")
        plt.xlabel("Peak VRAM Allocated (GB)")
        plt.ylabel(metric_col.replace("_", " ").title())
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"pareto_{task}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved plot: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_all_results(args.results_dir)
    if df.empty:
        print("No result JSON files found yet.")
        return
    summarize_multi_seed(df, args.output_dir)
    plot_pareto_frontier(df, args.output_dir)


if __name__ == "__main__":
    main()
