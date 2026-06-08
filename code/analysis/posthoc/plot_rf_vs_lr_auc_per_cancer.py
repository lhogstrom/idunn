#!/usr/bin/env python3
"""
Plot RF vs LR train/test AUC per cancer type (stacked panels).

Top panel:    Random Forest AUCs.
Bottom panel: Logistic Regression AUCs.

Shared x-axis ordering: cancer types sorted by RF test AUC descending.

By default emits both variants in one invocation:
- ``rf_vs_lr_auc_per_cancer.{pdf,png}``           — paired train + test bars
- ``rf_vs_lr_auc_per_cancer_test_only.{pdf,png}`` — single test bar

Test bars get DeLong 95% CI error bars when per-sample predictions
(``test_predictions.csv``) are available alongside ``performance_summary.csv``.
Falls back to no error bars (with one warning) when predictions are missing.

Usage::

    # Auto-detect newest RF and LR runs with TIER 5 outputs
    python plot_rf_vs_lr_auc_per_cancer.py

    # Pin specific runs
    python plot_rf_vs_lr_auc_per_cancer.py \\
        --rf-run-dir runs/run_20260123_075513 \\
        --lr-run-dir runs/run_20260505_100517
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))
from auc_utils import delong_auc_ci  # noqa: E402

DEFAULT_OUTPUT = Path("/tmp/ml_res")

RF_TRAIN_COLOR = "#a6cee3"
RF_TEST_COLOR  = "#1f78b4"
LR_TRAIN_COLOR = "#fdbf6f"
LR_TEST_COLOR  = "#ff7f00"


def find_perf_csv(run_dir: Path) -> Path:
    tier5 = run_dir / "tier5_final_models"
    if not tier5.exists():
        raise FileNotFoundError(f"No tier5_final_models under {run_dir}")
    for nested in sorted(tier5.glob("run_*"), reverse=True):
        f = nested / "performance_summary.csv"
        if f.exists():
            return f
    direct = tier5 / "performance_summary.csv"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"No performance_summary.csv under {tier5}")


def find_predictions_csv(run_dir: Path) -> Path | None:
    """Locate test_predictions.csv next to performance_summary.csv if present."""
    try:
        perf = find_perf_csv(run_dir)
    except FileNotFoundError:
        return None
    candidate = perf.parent / "test_predictions.csv"
    return candidate if candidate.exists() else None


def compute_delong_cis(predictions_csv: Path) -> pd.DataFrame:
    """
    Per-cancer-type DeLong AUC + 95% CI from a test_predictions.csv.

    Returns a DataFrame indexed by cancer_type with columns
    [auc_delong, ci_low, ci_high]. AUC matches sklearn's roc_auc_score.
    """
    df = pd.read_csv(predictions_csv)
    out = []
    for ct, sub in df.groupby("cancer_type"):
        auc, lo, hi = delong_auc_ci(sub["y_true"].values, sub["y_pred_proba"].values)
        out.append({"cancer_type": ct, "auc_delong": auc, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out).set_index("cancer_type")


def autodetect_run_dir(predicate) -> Path | None:
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        return None
    for run_dir in sorted(runs_dir.glob("run_*"), reverse=True):
        if predicate(run_dir):
            return run_dir
    return None


def has_rf_outputs(run_dir: Path) -> bool:
    if not (run_dir / "tier6_shap_analysis" / "primary_values").exists():
        return False
    try:
        find_perf_csv(run_dir)
    except FileNotFoundError:
        return False
    return True


def has_lr_outputs(run_dir: Path) -> bool:
    if not (run_dir / "tier6_shap_analysis" / "lr_analysis"
            / "combined_lr_grouped_importance.csv").exists():
        return False
    try:
        find_perf_csv(run_dir)
    except FileNotFoundError:
        return False
    return True


def load_aucs(run_dir: Path) -> pd.DataFrame:
    f = find_perf_csv(run_dir)
    df = pd.read_csv(f)
    return df[["cancer_type", "train_auc", "test_auc", "n_train", "n_test"]]


def _errorbar_arrays(df_cis: pd.DataFrame | None, order: list, test_vals):
    """Return (lower, upper) error magnitudes for plt.errorbar (yerr=[lo, hi])."""
    if df_cis is None:
        return None
    lo = np.zeros(len(order))
    hi = np.zeros(len(order))
    for i, ct in enumerate(order):
        if ct not in df_cis.index:
            continue
        cl = df_cis.loc[ct, "ci_low"]
        ch = df_cis.loc[ct, "ci_high"]
        if pd.isna(cl) or pd.isna(ch):
            continue
        lo[i] = max(0.0, test_vals[i] - cl)
        hi[i] = max(0.0, ch - test_vals[i])
    return np.vstack([lo, hi])


def _draw_panel(ax, df, order, c_train, c_test, label, test_only,
                cis: pd.DataFrame | None):
    x = np.arange(len(order))
    test = df.loc[order, "test_auc"].values
    if test_only:
        bar_x = x
        ax.bar(bar_x, test, 0.7,
               color=c_test, edgecolor="black", linewidth=0.4, label="Test AUC")
    else:
        train = df.loc[order, "train_auc"].values
        ax.bar(x - 0.2, train, 0.4,
               color=c_train, edgecolor="black", linewidth=0.4, label="Train AUC")
        bar_x = x + 0.2
        ax.bar(bar_x, test, 0.4,
               color=c_test, edgecolor="black", linewidth=0.4, label="Test AUC")

    # DeLong 95% CI error bars on test bars
    yerr = _errorbar_arrays(cis, order, test)
    if yerr is not None:
        ax.errorbar(bar_x, test, yerr=yerr, fmt="none",
                    ecolor="black", elinewidth=1.0, capsize=3,
                    label="DeLong 95% CI")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    mean_test = df.loc[order, "test_auc"].mean()
    ax.axhline(mean_test, color=c_test, linestyle=":", linewidth=1.2,
               alpha=0.7, label=f"Mean test AUC = {mean_test:.3f}")
    ax.set_title(f"{label} – AUC by cancer type",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("Test AUC" if test_only else "AUC")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)


def plot(rf_df: pd.DataFrame, lr_df: pd.DataFrame, output_dir: Path,
         rf_run: Path, lr_run: Path, test_only: bool = False,
         rf_cis: pd.DataFrame | None = None,
         lr_cis: pd.DataFrame | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    rf_idx = rf_df.set_index("cancer_type")
    lr_idx = lr_df.set_index("cancer_type")

    # Common cancer types (intersection), ordered by RF test_auc descending
    shared = rf_idx.index.intersection(lr_idx.index)
    order = (
        rf_idx.loc[shared].sort_values("test_auc", ascending=False).index.tolist()
    )

    n = len(order)
    fig_w = max(14, n * 0.6 + 2) if test_only else max(14, n * 0.7 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 10), sharex=True, sharey=True)

    _draw_panel(axes[0], rf_idx, order, RF_TRAIN_COLOR, RF_TEST_COLOR,
                "Random Forest", test_only, rf_cis)
    _draw_panel(axes[1], lr_idx, order, LR_TRAIN_COLOR, LR_TEST_COLOR,
                "Logistic Regression", test_only, lr_cis)

    if test_only:
        # Use CI bounds when present so error bars don't get clipped
        rf_max = rf_cis["ci_high"].max() if rf_cis is not None and "ci_high" in rf_cis else rf_idx["test_auc"].max()
        lr_max = lr_cis["ci_high"].max() if lr_cis is not None and "ci_high" in lr_cis else lr_idx["test_auc"].max()
        ymin = min(0.45, float(min(rf_idx["test_auc"].min(), lr_idx["test_auc"].min())) - 0.04)
        ymax = max(0.85, float(max(rf_max, lr_max)) + 0.03)
    else:
        ymin = min(0.45, float(min(rf_idx["test_auc"].min(), lr_idx["test_auc"].min())) - 0.04)
        ymax = max(0.85, float(max(rf_idx["train_auc"].max(), lr_idx["train_auc"].max())) + 0.03)
    axes[0].set_ylim(ymin, ymax)

    axes[1].set_xticks(np.arange(n))
    axes[1].set_xticklabels(order, rotation=45, ha="right")
    axes[1].set_xlabel("Cancer type (ordered by RF test AUC)")

    title_extra = " (held-out test set only)" if test_only else ""
    ci_note = "" if (rf_cis is not None and lr_cis is not None) else \
              " — CIs unavailable (test_predictions.csv missing)"
    fig.suptitle(
        f"Per-cancer-type AUC: Random Forest (top) vs Logistic Regression (bottom){title_extra}{ci_note}",
        fontweight="bold", fontsize=14, y=1.00,
    )
    fig.text(
        0.5, -0.01,
        f"RF run: {rf_run.name}    |    LR run: {lr_run.name}    |    n={n} cancer types",
        ha="center", fontsize=9, color="#555",
    )
    plt.tight_layout()

    stem = "rf_vs_lr_auc_per_cancer_test_only" if test_only else "rf_vs_lr_auc_per_cancer"
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rf-run-dir", type=Path, default=None,
        help="Run dir with RF TIER 5 outputs. "
        "Default: newest runs/run_* with tier6_shap_analysis/primary_values/.",
    )
    p.add_argument(
        "--lr-run-dir", type=Path, default=None,
        help="Run dir with LR TIER 5 outputs. "
        "Default: newest runs/run_* with tier6_shap_analysis/lr_analysis/.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--test-only", action="store_true",
        help="Emit only the test-only variant (skip the with-train variant). "
        "By default both variants are emitted.",
    )
    p.add_argument(
        "--with-train-only", action="store_true",
        help="Emit only the with-train variant (skip the test-only variant). "
        "By default both variants are emitted.",
    )
    args = p.parse_args()
    if args.test_only and args.with_train_only:
        raise SystemExit("--test-only and --with-train-only are mutually exclusive.")

    rf_run = args.rf_run_dir or autodetect_run_dir(has_rf_outputs)
    if rf_run is None:
        raise SystemExit(
            "ERROR: could not auto-detect an RF run dir. Pass --rf-run-dir explicitly."
        )
    lr_run = args.lr_run_dir or autodetect_run_dir(has_lr_outputs)
    if lr_run is None:
        raise SystemExit(
            "ERROR: could not auto-detect an LR run dir. Pass --lr-run-dir explicitly."
        )

    print(f"RF run: {rf_run}")
    print(f"LR run: {lr_run}")
    rf_df = load_aucs(rf_run)
    lr_df = load_aucs(lr_run)
    print(f"RF: {len(rf_df)} cancer types | LR: {len(lr_df)} cancer types")

    rf_pred = find_predictions_csv(rf_run)
    lr_pred = find_predictions_csv(lr_run)
    rf_cis = compute_delong_cis(rf_pred) if rf_pred else None
    lr_cis = compute_delong_cis(lr_pred) if lr_pred else None
    if rf_cis is None or lr_cis is None:
        missing = []
        if rf_cis is None: missing.append(f"RF run ({rf_run.name})")
        if lr_cis is None: missing.append(f"LR run ({lr_run.name})")
        print(
            f"WARNING: test_predictions.csv missing in {', '.join(missing)} — "
            "DeLong CIs will be omitted from those panels."
        )
    else:
        print(f"DeLong CIs computed for {len(rf_cis)} RF / {len(lr_cis)} LR cancer types")

    variants = []
    if not args.test_only:
        variants.append(False)  # with-train
    if not args.with_train_only:
        variants.append(True)   # test-only
    for test_only in variants:
        plot(rf_df, lr_df, args.output_dir, rf_run, lr_run,
             test_only=test_only, rf_cis=rf_cis, lr_cis=lr_cis)


if __name__ == "__main__":
    main()
