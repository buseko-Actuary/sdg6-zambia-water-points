"""
Figures for the report.

Palette note: the two data colours (blue #2a78d6, red #d03b3b) were checked with
a colour-vision-deficiency validator before use. The intuitive green/red pair
for "working / broken" fails deuteranopia separation badly (delta-E 4.1, where 8
is the floor), so a red-green colourblind reader could not tell a working pump
from a broken one. Blue and red separate cleanly for every simulated deficiency.
Every figure also labels its categories, so colour never carries meaning alone.

Run:  python src/evaluate.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

SURFACE = "#fcfcfb"
INK = "#14213D"       # Insight Analytics navy, used as ink rather than as a data colour
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#d03b3b"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",       # solid hairline, never dashed
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _clean(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.set_axisbelow(True)


def _short_title(t: str) -> str:
    """Readable, and above all unique. Several organisations contributed more
    than one survey, so trimming to the organisation name alone produces
    duplicate axis labels."""
    t = (t.replace("Living Water International", "LWI")
          .replace("Africa Latin America", "Africa/LatAm")
          .replace("Zambia Mozambique", "Zambia/Moz")
          .replace("Africa Boreholes", "Boreholes")
          .replace("WPDx Reingestion", "WPDx re-ingestion")
          .replace("Global", "").replace("_", " "))
    # "SNV Zambia_Zambia_2014" would otherwise read "SNV Zambia Zambia 2014".
    words, out = t.split(), []
    for w in words:
        if not out or w.lower() != out[-1].lower():
            out.append(w)
    return " ".join(out)[:32]


def fig_data_integrity(raw: pd.DataFrame) -> None:
    """The two problems that had to be fixed before any model was fitted."""
    d = raw[raw.status_id.isin(["Yes", "No"])].copy()
    d["y"] = (d.status_id == "No").astype(int)

    g = d.groupby("dataset_title").agg(
        n=("y", "size"),
        failures=("y", "sum"),
        clean_functional=("status_clean", lambda s: (s == "Functional").sum()),
        id_yes=("status_id", lambda s: (s == "Yes").sum()),
    )
    # No size threshold: the panel titles count programmes, so the panels must
    # show every programme they are counting.
    g = g.sort_values("n", ascending=True)
    labels = [f"{_short_title(t)}  (n={n:,})" for t, n in zip(g.index, g.n)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ypos = np.arange(len(g))

    # Panel 1: the corrupted label
    ax = axes[0]
    pct_available = g.id_yes / g.n * 100
    pct_functional = g.clean_functional / g.n * 100
    ax.barh(ypos + 0.19, pct_available, height=0.34, color=BLUE,
            label="Water available on the visit  (status_id)")
    ax.barh(ypos - 0.19, pct_functional, height=0.34, color=RED,
            label="Labelled 'Functional'  (status_clean)")
    # A zero-length bar is invisible, and its absence is the whole point.
    for i, v in enumerate(pct_functional):
        if v == 0:
            ax.text(0.8, i - 0.19, "0%", va="center", fontsize=7.5, color=RED)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("% of the programme's water points")
    ax.set_xlim(0, 108)
    ax.set_title("The two status columns describe different things", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), fontsize=8.5,
              ncol=1, handlelength=1.4)
    _clean(ax)

    # Panel 2: the reporting bias
    ax = axes[1]
    rate = g.failures / g.n * 100
    colours = [MUTED if r == 0 else BLUE for r in rate]
    ax.barh(ypos, rate, height=0.6, color=colours)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("% of water points recorded as non-functional")
    ax.set_title("Seven programmes never record a single failure")
    for i, r in enumerate(rate):
        ax.text(r + 0.35, i, "excluded" if r == 0 else f"{r:.1f}%",
                va="center", fontsize=8,
                color=MUTED if r == 0 else INK2)
    ax.set_xlim(0, max(rate) * 1.38)
    _clean(ax)

    fig.suptitle("Two data problems found before any model was fitted",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGS / "01_data_integrity.png")
    plt.close(fig)


def fig_leakage_and_splits(metrics: dict) -> None:
    """What a careless pipeline would have reported."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel 1: leakage
    ax = axes[0]
    leak = metrics["leakage_demo"]
    vals = [leak["with_rehab_priority_auc"], leak["without_auc"]]
    bars = ax.bar(["With rehab_priority\n(leaked)", "Without it\n(honest)"],
                  vals, width=0.42, color=[RED, BLUE])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.14)
    ax.set_title("One leaked column buys a near-perfect score")
    # Sits in the clear space above the shorter bar, not across the taller one.
    ax.annotate("rehab_priority is filled in for 96.8%\n"
                "of broken points and 0.0% of working\n"
                "ones. It is a repair queue written\n"
                "after somebody already knew the answer.",
                xy=(0.40, 0.72), xycoords="axes fraction", fontsize=8,
                color=INK2, linespacing=1.5)
    _clean(ax)

    # Panel 2: split optimism
    ax = axes[1]
    sc = metrics["split_comparison"]
    names = list(sc.keys())
    short = [n.replace(" ", "\n") for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.19, [sc[n]["random_ap"] for n in names], width=0.34,
           color=RED, label="Random split of points")
    ax.bar(x + 0.19, [sc[n]["grouped_ap"] for n in names], width=0.34,
           color=BLUE, label="Split by district")
    for i, n in enumerate(names):
        ax.text(i, max(sc[n]["random_ap"], sc[n]["grouped_ap"]) + 0.015,
                f"+{sc[n]['optimism']:.3f}", ha="center", fontsize=8.5,
                color=INK, fontweight="bold")
    ax.set_xticks(x, short, fontsize=8)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 0.48)
    ax.set_title("Random splits overstate every model")
    ax.legend(loc="upper left", fontsize=8)
    _clean(ax)

    fig.tight_layout()
    fig.savefig(FIGS / "02_leakage_and_splits.png")
    plt.close(fig)


def fig_performance(test: pd.DataFrame, metrics: dict) -> None:
    """Honest performance in districts the model has never seen."""
    y = test.y_true.to_numpy()
    p = test.y_prob.to_numpy()
    age = test.age_years.fillna(test.age_years.median()).to_numpy()
    prev = y.mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # PR
    ax = axes[0]
    for score, colour, name in ((p, BLUE, "Random Forest"),
                                (age, ORANGE, "Age rule")):
        pr, rc, _ = precision_recall_curve(y, score)
        ax.plot(rc, pr, color=colour, linewidth=2,
                label=f"{name} (AP {average_precision_score(y, score):.3f})")
    ax.axhline(prev, color=MUTED, linewidth=1.2, linestyle="--",
               label=f"No model ({prev:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision and recall")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 0.75)
    _clean(ax)

    # ROC
    ax = axes[1]
    for score, colour, name in ((p, BLUE, "Random Forest"),
                                (age, ORANGE, "Age rule")):
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, color=colour, linewidth=2,
                label=f"{name} (AUC {roc_auc_score(y, score):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--",
            label="No model (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("The model does not beat pump age")
    ax.legend(fontsize=8, loc="lower right")
    _clean(ax)

    # Calibration
    ax = axes[2]
    frac, mean_pred = calibration_curve(y, p, n_bins=6, strategy="quantile")
    ax.plot([0, max(mean_pred) * 1.1], [0, max(mean_pred) * 1.1],
            color=MUTED, linewidth=1.2, linestyle="--", label="Perfect")
    ax.plot(mean_pred, frac, color=BLUE, linewidth=2, marker="o",
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="Random Forest")
    ax.set_xlabel("Predicted probability of failure")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Calibration")
    ax.legend(fontsize=8, loc="upper left")
    _clean(ax)

    ho = metrics["held_out_districts"]
    fig.suptitle(
        f"Held-out districts: {ho['n_test']} water points in "
        f"{ho['n_test_districts']} districts the model never saw",
        fontsize=11, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "03_performance.png")
    plt.close(fig)


def fig_inspection(test: pd.DataFrame) -> None:
    """The only question a district water officer actually asks."""
    y = test.y_true.to_numpy()
    p = test.y_prob.to_numpy()
    order = np.argsort(-p)
    ys = y[order]
    n, total = len(y), int(y.sum())

    visited = np.arange(1, n + 1) / n * 100
    found = np.cumsum(ys) / total * 100

    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.plot(visited, found, color=BLUE, linewidth=2.2,
            label="Visit in model-ranked order")
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1.4, linestyle="--",
            label="Visit in arbitrary order")

    for pct in (10, 20):
        k = max(1, int(round(pct / 100 * n)))
        f = ys[:k].sum() / total * 100
        ax.plot([pct, pct], [0, f], color=GRID, linewidth=1)
        ax.plot(pct, f, marker="o", markersize=7, color=BLUE,
                markeredgecolor=SURFACE, markeredgewidth=1.6)
        ax.annotate(f"visit {pct}% of points,\nfind {f:.0f}% of the failures",
                    xy=(pct, f), xytext=(pct + 6, f - 11), fontsize=8.5,
                    color=INK)

    ax.set_xlabel("Share of the district's water points inspected (%)")
    ax.set_ylabel("Share of broken water points found (%)")
    ax.set_title("What the ranking is worth to a repair crew")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "04_inspection_curve.png")
    plt.close(fig)


def fig_importance_and_age(imp: pd.DataFrame, model_table: pd.DataFrame) -> None:
    """Where the little signal there is actually comes from."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    top = imp.head(8).iloc[::-1]
    ax.barh(np.arange(len(top)), top.importance, xerr=top["std"],
            height=0.62, color=BLUE,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(np.arange(len(top)), top.feature)
    ax.set_xlabel("Drop in PR-AUC when the column is shuffled")
    ax.set_title("Permutation importance, measured on unseen districts")
    ax.axvline(0, color=GRID, linewidth=1)
    _clean(ax)

    ax = axes[1]
    d = model_table.dropna(subset=["age_years"]).copy()
    bins = [0, 5, 10, 15, 20, 25, 100]
    names = ["0-5", "6-10", "11-15", "16-20", "21-25", "25+"]
    d["band"] = pd.cut(d.age_years, bins=bins, labels=names, right=True)
    grp = d.groupby("band", observed=True).agg(
        rate=("is_non_functional", "mean"), n=("is_non_functional", "size"))
    ax.plot(np.arange(len(grp)), grp.rate * 100, color=BLUE, linewidth=2.2,
            marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.6)
    ax.set_xticks(np.arange(len(grp)), grp.index)
    ax.set_xlabel("Age of the water point at inspection (years)")
    ax.set_ylabel("Non-functional (%)")
    ax.set_title("Older pumps do fail more, and that is most of the signal")
    # Sample sizes sit on the baseline rather than beside each marker, where
    # they collided with the line.
    for i, cnt in enumerate(grp.n):
        ax.annotate(f"n={cnt:,}", xy=(i, 0), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)
    ax.set_ylim(0, max(grp.rate * 100) * 1.25)
    _clean(ax)

    fig.tight_layout()
    fig.savefig(FIGS / "05_importance_and_age.png")
    plt.close(fig)


def fig_map(raw: pd.DataFrame) -> None:
    """Where the water points are, and which of them were dry on the visit."""
    d = raw[raw.status_id.isin(["Yes", "No"])].copy()
    d["y"] = (d.status_id == "No").astype(int)
    rates = d.groupby("dataset_title")["y"].sum()
    d = d[d.dataset_title.isin(rates[rates > 0].index)]

    fig, ax = plt.subplots(figsize=(8, 6.4))
    ok = d[d.y == 0]
    bad = d[d.y == 1]
    ax.scatter(ok.lon_deg, ok.lat_deg, s=7, color=BLUE, alpha=0.55,
               linewidths=0, label=f"Working ({len(ok):,})")
    ax.scatter(bad.lon_deg, bad.lat_deg, s=16, color=RED, alpha=0.9,
               linewidths=0.4, edgecolors=SURFACE,
               label=f"Non-functional ({len(bad):,})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Zambia: the 4,026 water points used in this study")
    # Upper left is the emptiest corner of the country on this projection.
    ax.legend(loc="upper left", fontsize=9, markerscale=1.6)
    ax.set_aspect("equal", adjustable="datalim")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "06_map.png")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "raw" / "wpdx_water_points_zmb.csv",
                      skiprows=[1], low_memory=False)
    model_table = pd.read_csv(ROOT / "data" / "processed" /
                              "water_points_model_table.csv")
    test = pd.read_csv(REPORTS / "test_predictions.csv")
    imp = pd.read_csv(REPORTS / "permutation_importance.csv")
    with open(REPORTS / "metrics.json") as f:
        metrics = json.load(f)

    fig_data_integrity(raw)
    fig_leakage_and_splits(metrics)
    fig_performance(test, metrics)
    fig_inspection(test)
    fig_importance_and_age(imp, model_table)
    fig_map(raw)

    for p in sorted(FIGS.glob("*.png")):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
