"""
Feature ablation, scored the same way the final model is scored.

Question: does giving the model exact coordinates help it, or does it just let
it memorise neighbourhoods that do not exist in a district it has never seen?

Everything here runs on the training districts only. The held-out districts are
not touched, so nothing in this file can tune a decision using the test set.

Run:  python src/ablation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import (
    CATEGORICAL_FEATURES,
    GROUP,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)

ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5

FEATURE_SETS = {
    "A. Everything": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    "B. No coordinates": (
        [c for c in NUMERIC_FEATURES if c not in ("lat_deg", "lon_deg")],
        CATEGORICAL_FEATURES,
    ),
    "C. No coordinates, no province": (
        [c for c in NUMERIC_FEATURES if c not in ("lat_deg", "lon_deg")],
        [c for c in CATEGORICAL_FEATURES if c != "clean_adm1"],
    ),
    "D. Asset only (age, tech, source)": (
        ["age_years"],
        ["water_source_clean", "water_tech_clean", "water_source_category",
         "water_tech_category"],
    ),
    "E. Age alone": (["age_years"], []),
}


def make_pipeline(num, cat, kind):
    steps_num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if kind == "lr":
        steps_num.append(("scale", StandardScaler()))

    transformers = []
    if num:
        transformers.append(("num", Pipeline(steps_num), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                     sparse_output=False)),
        ]), cat))

    prep = ColumnTransformer(transformers, remainder="drop")

    if kind == "lr":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=RANDOM_STATE)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                     class_weight="balanced_subsample",
                                     random_state=RANDOM_STATE, n_jobs=-1)
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=30, l2_regularization=1.0,
            class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("prep", prep), ("clf", clf)])


def grouped_score(model, X, y, groups):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)
    aps, aucs = [], []
    for tr, va in cv.split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        if y.iloc[va].nunique() > 1:
            aps.append(average_precision_score(y.iloc[va], p))
            aucs.append(roc_auc_score(y.iloc[va], p))
    return float(np.mean(aps)), float(np.std(aps)), float(np.mean(aucs))


def age_rule_grouped(X, y, groups):
    """Score the no-model rule on identical folds, so the comparison is fair."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)
    aps, aucs = [], []
    for tr, va in cv.split(X, y, groups):
        med = X.iloc[tr]["age_years"].median()
        s = X.iloc[va]["age_years"].fillna(med)
        if y.iloc[va].nunique() > 1:
            aps.append(average_precision_score(y.iloc[va], s))
            aucs.append(roc_auc_score(y.iloc[va], s))
    return float(np.mean(aps)), float(np.std(aps)), float(np.mean(aucs))


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "water_points_model_table.csv")
    X_all = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_all = df[TARGET]
    g_all = df[GROUP]

    outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=RANDOM_STATE)
    train_idx, _ = next(outer.split(X_all, y_all, g_all))
    X, y, g = X_all.iloc[train_idx], y_all.iloc[train_idx], g_all.iloc[train_idx]

    print(f"Ablation on {len(X):,} training points in {g.nunique()} districts, "
          f"{y.sum()} non-functional ({y.mean()*100:.1f}%)")
    print("Scored by 5-fold cross-validation grouped by district.\n")

    ap, sd, auc = age_rule_grouped(X, y, g)
    print(f"{'No model: rank by age':<38s} {'-':>6s}  PR-AUC {ap:.3f} +/- {sd:.3f}"
          f"   ROC-AUC {auc:.3f}")
    print()

    rows = []
    for set_name, (num, cat) in FEATURE_SETS.items():
        for kind, label in (("lr", "LogReg"), ("rf", "RF"), ("gb", "GBoost")):
            model = make_pipeline(num, cat, kind)
            ap, sd, auc = grouped_score(model, X, y, g)
            rows.append({"features": set_name, "model": label,
                         "pr_auc": ap, "pr_auc_std": sd, "roc_auc": auc,
                         "n_features": len(num) + len(cat)})
            print(f"{set_name:<38s} {label:>6s}  PR-AUC {ap:.3f} +/- {sd:.3f}"
                  f"   ROC-AUC {auc:.3f}")
        print()

    out = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    out.to_csv(ROOT / "reports" / "ablation.csv", index=False)
    print("Best by PR-AUC:")
    print(out.head(5).to_string(index=False))
    print(f"\nWrote {ROOT / 'reports' / 'ablation.csv'}")


if __name__ == "__main__":
    main()
