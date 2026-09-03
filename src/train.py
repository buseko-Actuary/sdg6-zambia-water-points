"""
Model training and honest evaluation.

Four things here are deliberate, and they are the point of the project:

1. Every transform lives inside a Pipeline, so imputation and encoding are
   fitted on training folds only and can never see the validation fold.
2. Validation is grouped by district. Water points in the same district share
   an installer, a water table and a maintenance team, so a random split lets
   the model recognise a neighbourhood it has already seen. The number that
   matters is performance in a district nobody has visited. Both numbers are
   reported, because the gap between them is a finding.
3. The model is compared against the rule a planner would use without it
   (visit the oldest pumps first), scored on identical folds. A model that
   cannot beat that rule is not worth deploying.
4. The final number is a ranking decision, not an accuracy: given a crew that
   can reach N water points, how many broken ones does the ranking find?

Coordinates are excluded on the evidence of src/ablation.py, which was run on
training districts only. The held-out districts are scored exactly once, at the
end of this script.

Run:  python src/train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import (
    CATEGORICAL_FEATURES,
    GROUP,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "water_points_model_table.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

N_SPLITS = 5

# Latitude and longitude are dropped: see src/ablation.py. Removing them did
# not hurt performance in an unseen district, which is the expected result if
# the model was using them to memorise neighbourhoods rather than to learn
# anything that travels.
MODEL_NUMERIC = [c for c in NUMERIC_FEATURES if c not in ("lat_deg", "lon_deg")]
MODEL_FEATURES = MODEL_NUMERIC + CATEGORICAL_FEATURES


def build_preprocessor(scale: bool, numeric=None) -> ColumnTransformer:
    """Impute and encode. Fitted per fold, never on the full dataset.

    add_indicator keeps the fact that a value was missing. A water point with
    no recorded installation year is not an average water point, and that
    absence may say something about how well the point is documented and
    looked after.
    """
    numeric = numeric or MODEL_NUMERIC
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                # min_frequency folds rare categories into one bucket so a
                # single water point cannot create its own column.
                ("encode", OneHotEncoder(handle_unknown="ignore",
                                         min_frequency=10, sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Baseline (predict prevalence)": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", DummyClassifier(strategy="prior")),
        ]),
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                min_samples_leaf=30, l2_regularization=1.0,
                class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def cross_validate(model, X, y, groups, splitter) -> dict:
    """Out-of-fold scores, plus the age rule scored on identical folds."""
    oof = np.full(len(y), np.nan)
    fold_ap, fold_auc = [], []
    rule_ap, rule_auc = [], []

    grouped = isinstance(splitter, StratifiedGroupKFold)
    split_args = (X, y, groups) if grouped else (X, y)

    for train_idx, val_idx in splitter.split(*split_args):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = m.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = p

        y_val = y.iloc[val_idx]
        if y_val.nunique() < 2:
            continue  # a fold with one class present cannot be scored
        fold_ap.append(average_precision_score(y_val, p))
        fold_auc.append(roc_auc_score(y_val, p))

        # The no-model rule, on exactly the same rows.
        med = X.iloc[train_idx]["age_years"].median()
        rule = X.iloc[val_idx]["age_years"].fillna(med)
        rule_ap.append(average_precision_score(y_val, rule))
        rule_auc.append(roc_auc_score(y_val, rule))

    return {
        "ap_mean": float(np.mean(fold_ap)), "ap_std": float(np.std(fold_ap)),
        "auc_mean": float(np.mean(fold_auc)), "auc_std": float(np.std(fold_auc)),
        "rule_ap": float(np.mean(rule_ap)), "rule_auc": float(np.mean(rule_auc)),
        "oof": oof,
    }


def inspection_curve(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Turn a probability into the only question a water officer asks.

    If a crew can reach the top k% of a district's water points, how many of
    the broken ones do they find, and how much better is that than turning up
    in an arbitrary order?
    """
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    total_broken = int(y_true.sum())
    prevalence = y_true.mean()

    rows = []
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = max(1, int(round(frac * len(y_true))))
        found = int(y_sorted[:k].sum())
        precision = found / k
        rows.append({
            "visit_top_pct": int(frac * 100),
            "points_visited": k,
            "broken_found": found,
            "of_total_broken": total_broken,
            "precision_at_k": round(precision, 3),
            "recall_at_k": round(found / total_broken, 3) if total_broken else np.nan,
            "lift_vs_random": round(precision / prevalence, 2) if prevalence else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(y_true, y_prob, metric, n=2000, seed=RANDOM_STATE):
    """Percentile interval. The test set is 669 points with 83 failures, so a
    point estimate on its own would overstate how much we know."""
    rng = np.random.default_rng(seed)
    stats = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if y_true[s].sum() in (0, len(s)):
            continue
        stats.append(metric(y_true[s], y_prob[s]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_csv(PROCESSED)
    X = df[MODEL_FEATURES]
    y = df[TARGET]
    groups = df[GROUP]

    print(f"{len(df):,} water points | {y.sum():,} non-functional "
          f"({y.mean() * 100:.1f}%) | {groups.nunique()} districts")
    print(f"{len(MODEL_FEATURES)} features (coordinates excluded, see ablation.py)\n")

    # ------------------------------------------------------------------
    # Hold out whole districts, never individual points
    # ------------------------------------------------------------------
    outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=RANDOM_STATE)
    train_idx, test_idx = next(outer.split(X, y, groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    g_tr, g_te = groups.iloc[train_idx], groups.iloc[test_idx]

    print(f"Train: {len(X_tr):,} points, {g_tr.nunique()} districts "
          f"({y_tr.mean() * 100:.1f}% non-functional)")
    print(f"Test:  {len(X_te):,} points, {g_te.nunique()} unseen districts "
          f"({y_te.mean() * 100:.1f}% non-functional)\n")

    # ------------------------------------------------------------------
    # Model selection on training districts only
    # ------------------------------------------------------------------
    grouped_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                      random_state=RANDOM_STATE)
    results, oof_store = {}, {}

    print("Cross-validation, grouped by district:")
    print(f"{'Model':<32s} {'PR-AUC':>18s} {'ROC-AUC':>18s}")
    for name, model in candidate_models().items():
        r = cross_validate(model, X_tr, y_tr, g_tr, grouped_cv)
        results[name] = {k: v for k, v in r.items() if k != "oof"}
        oof_store[name] = r["oof"]
        print(f"{name:<32s} {r['ap_mean']:.3f} +/- {r['ap_std']:.3f}   "
              f"  {r['auc_mean']:.3f} +/- {r['auc_std']:.3f}")

    rule = results["Random Forest"]
    print(f"{'No model: visit oldest first':<32s} {rule['rule_ap']:.3f}"
          f"                {rule['rule_auc']:.3f}")
    print("   (scored on identical folds, so this is a like-for-like comparison)")

    # ------------------------------------------------------------------
    # Why the grouping matters
    # ------------------------------------------------------------------
    print("\nSame models, random split of points instead of districts:")
    random_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=RANDOM_STATE)
    split_comparison = {}
    for name, model in candidate_models().items():
        if name.startswith("Baseline"):
            continue
        r = cross_validate(model, X_tr, y_tr, g_tr, random_cv)
        gap = r["ap_mean"] - results[name]["ap_mean"]
        split_comparison[name] = {
            "random_ap": r["ap_mean"],
            "grouped_ap": results[name]["ap_mean"],
            "optimism": gap,
        }
        print(f"{name:<32s} random {r['ap_mean']:.3f}  vs  district "
              f"{results[name]['ap_mean']:.3f}   overstated by {gap:+.3f}")

    # ------------------------------------------------------------------
    # What leakage would have bought us
    # ------------------------------------------------------------------
    print("\nLeakage demonstration, rehab_priority added back as a feature:")
    raw = pd.read_csv(ROOT / "data" / "raw" / "wpdx_water_points_zmb.csv",
                      skiprows=[1], low_memory=False)
    # Join on wpdx_id, not position: the processed table has been filtered.
    flag_by_id = raw.set_index("wpdx_id")["rehab_priority"].notna().astype(int)
    leak_flag = df.iloc[train_idx]["wpdx_id"].map(flag_by_id).fillna(0).astype(int)
    leak_frame = X_tr.assign(rehab_priority_present=leak_flag.to_numpy())

    leak_model = Pipeline([
        ("prep", build_preprocessor(
            scale=False, numeric=MODEL_NUMERIC + ["rehab_priority_present"])),
        ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                       class_weight="balanced_subsample",
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    leak_r = cross_validate(leak_model, leak_frame, y_tr, g_tr, grouped_cv)
    print(f"  WITH  rehab_priority: PR-AUC {leak_r['ap_mean']:.3f}  "
          f"ROC-AUC {leak_r['auc_mean']:.3f}   looks excellent, predicts nothing")
    print(f"  WITHOUT it:           PR-AUC {results['Random Forest']['ap_mean']:.3f}  "
          f"ROC-AUC {results['Random Forest']['auc_mean']:.3f}   honest")

    # ------------------------------------------------------------------
    # Final model, scored once on unseen districts
    # ------------------------------------------------------------------
    best_name = max((n for n in results if not n.startswith("Baseline")),
                    key=lambda n: results[n]["ap_mean"])
    print(f"\nSelected on grouped CV: {best_name}")

    best = candidate_models()[best_name]
    best.fit(X_tr, y_tr)
    p_te = best.predict_proba(X_te)[:, 1]
    yte = y_te.to_numpy()

    ap_lo, ap_hi = bootstrap_ci(yte, p_te, average_precision_score)
    auc_lo, auc_hi = bootstrap_ci(yte, p_te, roc_auc_score)

    med_age = X_tr["age_years"].median()
    rule_te = X_te["age_years"].fillna(med_age).to_numpy()

    test_metrics = {
        "model": best_name,
        "n_test": int(len(yte)),
        "n_test_districts": int(g_te.nunique()),
        "n_broken": int(yte.sum()),
        "prevalence": float(yte.mean()),
        "pr_auc": float(average_precision_score(yte, p_te)),
        "pr_auc_ci95": [ap_lo, ap_hi],
        "roc_auc": float(roc_auc_score(yte, p_te)),
        "roc_auc_ci95": [auc_lo, auc_hi],
        "brier": float(brier_score_loss(yte, p_te)),
        "age_rule_pr_auc": float(average_precision_score(yte, rule_te)),
        "age_rule_roc_auc": float(roc_auc_score(yte, rule_te)),
    }
    print(f"\nHeld-out districts ({test_metrics['n_test']} points, "
          f"{test_metrics['n_broken']} broken, prevalence "
          f"{test_metrics['prevalence']:.3f}):")
    print(f"  PR-AUC  {test_metrics['pr_auc']:.3f}  95% CI "
          f"[{ap_lo:.3f}, {ap_hi:.3f}]")
    print(f"  ROC-AUC {test_metrics['roc_auc']:.3f}  95% CI "
          f"[{auc_lo:.3f}, {auc_hi:.3f}]")
    print(f"  Brier   {test_metrics['brier']:.3f}")
    print(f"  Age rule on the same points: PR-AUC "
          f"{test_metrics['age_rule_pr_auc']:.3f}, ROC-AUC "
          f"{test_metrics['age_rule_roc_auc']:.3f}")

    # ------------------------------------------------------------------
    # The operational question
    # ------------------------------------------------------------------
    curve = inspection_curve(yte, p_te)
    print("\nIf a repair crew can only reach part of the district:")
    print(curve.to_string(index=False))
    curve.to_csv(REPORTS / "inspection_curve.csv", index=False)

    # ------------------------------------------------------------------
    # Which features earn their place (permutation, not impurity)
    # ------------------------------------------------------------------
    perm = permutation_importance(best, X_te, y_te, n_repeats=30,
                                  random_state=RANDOM_STATE,
                                  scoring="average_precision", n_jobs=-1)
    imp = (pd.DataFrame({"feature": MODEL_FEATURES,
                         "importance": perm.importances_mean,
                         "std": perm.importances_std})
           .sort_values("importance", ascending=False))
    print("\nPermutation importance on held-out districts (drop in PR-AUC):")
    print(imp.head(8).to_string(index=False))
    imp.to_csv(REPORTS / "permutation_importance.csv", index=False)

    # ------------------------------------------------------------------
    # Save everything the report needs
    # ------------------------------------------------------------------
    # compress=3 takes a 500-tree forest from ~10 MB to well under 2 MB, which
    # keeps the repository clone-able without an LFS pointer.
    joblib.dump(best, MODELS / "water_point_model.joblib", compress=3)
    pd.DataFrame({"y_true": yte, "y_prob": p_te,
                  "age_years": X_te["age_years"].to_numpy(),
                  "district": g_te.to_numpy()}
                 ).to_csv(REPORTS / "test_predictions.csv", index=False)
    pd.DataFrame({"y_true": y_tr.to_numpy(),
                  **{f"oof_{n}": v for n, v in oof_store.items()}}
                 ).to_csv(REPORTS / "oof_predictions.csv", index=False)

    with open(REPORTS / "metrics.json", "w") as f:
        json.dump({
            "dataset": {
                "n_rows": int(len(df)), "n_broken": int(y.sum()),
                "prevalence": float(y.mean()),
                "n_districts": int(groups.nunique()),
                "n_features": len(MODEL_FEATURES),
            },
            "cv_grouped_by_district": results,
            "split_comparison": split_comparison,
            "leakage_demo": {
                "with_rehab_priority_ap": leak_r["ap_mean"],
                "with_rehab_priority_auc": leak_r["auc_mean"],
                "without_ap": results["Random Forest"]["ap_mean"],
                "without_auc": results["Random Forest"]["auc_mean"],
            },
            "held_out_districts": test_metrics,
            "inspection_curve": curve.to_dict(orient="records"),
        }, f, indent=2)

    print(f"\nSaved model   {MODELS / 'water_point_model.joblib'}")
    print(f"Saved metrics {REPORTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
