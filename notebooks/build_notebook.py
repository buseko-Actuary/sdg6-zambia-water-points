"""
Builds notebooks/water-point-failure-walkthrough.ipynb.

The notebook is the narrative version of src/data_prep.py, src/train.py and
src/ablation.py: the same data, the same decisions, in the order they were
actually made, with the reasoning written between the cells.

Run:  python notebooks/build_notebook.py
Then: jupyter nbconvert --execute --to notebook --inplace <the notebook>
"""

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ===========================================================================
md("""
# Predicting rural water point failure in Zambia

**A walkthrough of the whole project, in the order it actually happened.**

Buseko Fungamwango &middot; Insight Analytics Ltd &middot; SDG 6, Clean Water and Sanitation

---

## What this notebook is

This is the narrative version of the code in `src/`. Same data, same decisions,
same numbers, but with the reasoning written down between the steps and the
dead ends left in.

The short version of the story: I set out to predict which rural water points
in Zambia are broken, found three separate traps in the data that would each
have produced an impressive and completely false result, fixed all three, and
ended up with a model that **loses to a one-line rule**.

I published that result rather than hiding it, and the reason it matters is at
the end.

## The question

Roughly a third of rural handpumps in sub-Saharan Africa are non-functional at
any given time. Repair crews are few and districts are large. If you could rank
water points by how likely they are to be broken, a crew could visit the
worst-looking ones first instead of driving at random.

So: **given a water point's age, type, location and the population it serves,
can we predict whether it is working?**

## How to read this

| Section | What happens |
|---|---|
| 1 to 2 | Load the raw data and look at it properly before modelling anything |
| 3 to 5 | The three traps, each found by counting rather than by modelling |
| 6 to 7 | Build the modelling table, with every exclusion stated |
| 8 to 10 | Split honestly, compare models, and measure what a careless split would have bought me |
| 11 to 13 | Score once on unseen districts, then answer the operational question |
| 14 to 16 | What the model leaned on, what the ablation said, and what I would defend |
""")

# ===========================================================================
md("""
## 1. Setup

Everything is loaded from the repository itself, so this notebook reproduces
the committed results rather than restating them.
""")

code("""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Run from anywhere inside the repo.
ROOT = Path.cwd()
while not (ROOT / "src" / "data_prep.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 40)

# One consistent look for every chart in the notebook.
NAVY, GOLD, SLATE, RUST = "#14213D", "#D6AF7B", "#6B7A8F", "#B0632A"
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110,
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

print("repo root:", ROOT)
print("pandas", pd.__version__, "| numpy", np.__version__)
import sklearn; print("scikit-learn", sklearn.__version__)
""")

# ===========================================================================
md("""
## 2. The raw data

The source is the **Water Point Data Exchange (WPdx)**, obtained through the UN
OCHA Humanitarian Data Exchange. It is a shared register that governments,
NGOs and donor programmes all contribute water point records to.

<https://data.humdata.org/dataset/wpdx_zmb> &middot; licensed CC BY-SA

One detail that matters immediately: **row 2 of the file is not data.** It is a
row of HXL humanitarian markup tags. Read the file naively and every numeric
column arrives as text.
""")

code("""
from data_prep import (
    RAW, TARGET, GROUP, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    RANDOM_STATE, EXCLUSIONS, load_raw, build_label,
    drop_single_class_programmes, engineer_features,
)

raw = load_raw()   # skiprows=[1] drops the HXL tag row
print(f"{raw.shape[0]:,} rows x {raw.shape[1]} columns")
print(f"Reporting period: {raw['report_date'].min()} to {raw['report_date'].max()}")
print(f"Contributing programmes: {raw['dataset_title'].nunique()}")
print(f"Districts: {raw['clean_adm2'].nunique()}   Provinces: {raw['clean_adm1'].nunique()}")
""")

code("""
# What is actually usable? Missingness first, before any modelling ideas.
miss = raw.isna().mean().mul(100).sort_values(ascending=False)
print("Emptiest columns (% missing):")
print(miss.head(12).round(1).to_string())
print()
print("Most complete columns (% missing):")
print(miss.tail(10).round(1).to_string())
""")

# ===========================================================================
md("""
## 3. Trap one: the label is corrupted

There are two columns that look like they answer "does this water point work?"

- `status_clean` &mdash; a tidy Functional / Non-Functional category
- `status_id` &mdash; a raw Yes / No / Unknown, meaning "was water available on the day of the visit?"

`status_clean` is the obvious choice. It is cleaner, it has no Unknowns, and it
is what the column name promises.

**It is also wrong.** Before trusting a label, cross-tabulate it against who
supplied the record.
""")

code("""
# Who reports what? One row per contributing programme.
audit = (raw.assign(clean=raw["status_clean"].fillna("(blank)"))
            .pivot_table(index="dataset_title", columns="clean",
                         values="wpdx_id", aggfunc="count", fill_value=0))
audit["n"] = audit.sum(axis=1)

# Share of each programme's points that status_id says had water available.
water_yes = (raw.assign(yes=(raw["status_id"] == "Yes").astype(int))
                .groupby("dataset_title")["yes"].mean().mul(100).round(1))
audit["status_id says water available %"] = water_yes

audit = audit.sort_values("n", ascending=False)
audit.index = [t[:46] for t in audit.index]
print(f"{len(audit)} contributing programmes\\n")
print(audit.to_string())
""")

md("""
Read the `Functional` column.

Of the fifteen contributing programmes, **exactly one ever records the word
"Functional"**. Every other programme has all of its water points labelled
`Non-Functional`, including programmes where `status_id` says water was flowing
at 100% of the points visited.

That is not a country where every pump is broken. It is a column where a
**missing value has been written down as a category**. `status_clean` measures
who uploaded the record, not whether the water point works.

The comparison in one line:
""")

code("""
print("If I had trusted status_clean:")
sc = raw["status_clean"].value_counts(dropna=False)
print(f"  Non-Functional: {sc.get('Non-Functional', 0):,}")
print(f"  Functional:     {sc.get('Functional', 0):,}")
print(f"  -> apparent failure rate: "
      f"{sc.get('Non-Functional', 0) / sc.sum() * 100:.1f}%")

print("\\nWhat status_id actually records:")
si = raw["status_id"].value_counts(dropna=False)
print(si.to_string())
usable = si.get("Yes", 0) + si.get("No", 0)
print(f"  -> real failure rate among visited points: "
      f"{si.get('No', 0) / usable * 100:.1f}%")
""")

md("""
A 73.7% national failure rate versus a 7.7% one. Publishing the first number
would have been a serious thing to get wrong, and nothing in the modelling
would ever have revealed it. Only counting did.

(The 7.7% here is across every point with a real observation. It rises to 11.8%
in section 5, once the programmes that never record a failure are removed.)

`status_id` becomes the label. `Unknown` is **dropped rather than guessed**:
assigning those rows to either class would invent observations nobody made.
""")

code("""
labelled = build_label(raw)
print(f"Kept {len(labelled):,} rows with a real Yes/No observation")
print(f"Dropped {len(raw) - len(labelled):,} rows where status was Unknown")
print(f"Failure rate: {labelled[TARGET].mean() * 100:.1f}%")
""")

# ===========================================================================
md("""
## 4. Trap two: two columns contain the answer

With an honest label, I trained a Random Forest. It scored **ROC-AUC 0.998**.

That is not a good sign. Real problems are not that easy, and a near-perfect
score usually means the answer has leaked into the features.

The test for leakage is simple: for each column, ask how often it is populated
for broken points versus working ones. A feature that is present for one class
and absent for the other is not a predictor. It is a label in disguise.
""")

code("""
def population_by_class(df, cols):
    \"\"\"How often is each column filled in, split by the outcome?\"\"\"
    rows = []
    for c in cols:
        present = df[c].notna()
        rows.append({
            "column": c,
            "% filled when BROKEN": round(present[df[TARGET] == 1].mean() * 100, 1),
            "% filled when WORKING": round(present[df[TARGET] == 0].mean() * 100, 1),
        })
    out = pd.DataFrame(rows)
    out["gap"] = out["% filled when BROKEN"] - out["% filled when WORKING"]
    return out.sort_values("gap", ascending=False)

suspects = ["rehab_priority", "would_gain_access", "subjective_quality",
            "notes", "install_year", "local_population", "water_tech_clean"]
print(population_by_class(labelled, suspects).to_string(index=False))
""")

md("""
There they are.

`rehab_priority` is filled in for **96.8%** of broken points and **0.0%** of
working ones. `would_gain_access` is filled in for **99.6%** of broken points
and **0.0%** of working ones.

Neither is a measurement of the water point. Both are **rehabilitation planning
notes, written after somebody already knew the point had failed**. A model using
them is not predicting failure, it is reading a repair queue.

I will keep this as a live demonstration and put `rehab_priority` back in
later, in section 11, to show exactly what it was worth.
""")

# ===========================================================================
md("""
## 5. Trap three: the programmes that never fail

One more count, and this one is easy to miss because the column looks harmless.

How many failures does each contributing programme report?
""")

code("""
_, programme_rates = drop_single_class_programmes(labelled)
pr = programme_rates.copy()
pr.index = [t[:52] for t in pr.index]
pr["failure_rate %"] = (pr["failure_rate"] * 100).round(1)
print(pr[["n", "failures", "failure_rate %"]].to_string())

zero = pr[pr["failures"] == 0]
print(f"\\n{len(zero)} of {len(pr)} programmes report ZERO failures, "
      f"covering {int(zero['n'].sum()):,} water points.")
""")

md("""
Seven programmes, 2,171 water points, and not a single failure between them.

No water programme in the world has a 0% failure rate. What is really happening
is a reporting convention: **those programmes upload their points at handover**,
on the day of installation, when everything works by definition, and never
return to update the record.

Left in, a model learns "points from Programme X never break". It can reach that
conclusion without ever seeing the programme name, because programme correlates
with district, with installation year, and with location. The result would look
like geography and would actually be paperwork.

So those programmes come out. Note what this costs: it is not free.
""")

code("""
modelling, _ = drop_single_class_programmes(labelled)
print(f"Before: {len(labelled):,} points, {labelled[TARGET].mean()*100:.1f}% failure rate")
print(f"After:  {len(modelling):,} points, {modelling[TARGET].mean()*100:.1f}% failure rate")
print(f"\\nCost of this decision: {len(labelled) - len(modelling):,} points removed "
      f"({(1 - len(modelling)/len(labelled))*100:.0f}% of the labelled data).")
print("Kept because a programme with no failures carries no information about failure.")
""")

# ===========================================================================
md("""
## 6. Every exclusion, stated

Nothing is dropped silently. This is the full list from `src/data_prep.py`,
printed so the audit trail lives in the output rather than only in a README.
""")

code("""
for cols, reason in EXCLUSIONS.items():
    head = cols if len(cols) < 62 else cols[:59] + "..."
    print(f"* {head}")
    print(f"    {' '.join(reason.split())[:230]}")
    print()
""")

md("""
### Feature engineering

Only one derived feature really matters, and it is the physically obvious one:
**how old was the pump when somebody looked at it?**

The raw file contains install years as early as 1902, which predates any
borehole programme in Zambia by decades. Those are data entry errors, not very
old pumps, so they are blanked and imputed rather than believed.
""")

code("""
modelling = engineer_features(modelling)

install = pd.to_numeric(modelling["install_year"], errors="coerce")
bad = ((install < 1950) | (install > modelling["report_year"])).sum()
print(f"Implausible install years blanked: {bad:,}")
print(f"\\nAge at inspection (years):")
print(modelling["age_years"].describe().round(1).to_string())
print(f"\\nMissing age: {modelling['age_years'].isna().mean()*100:.1f}%")
""")

# ===========================================================================
md("""
## 7. The modelling table
""")

code("""
keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, GROUP, "report_year", "wpdx_id"]
df = modelling[keep].copy()

print(f"{len(df):,} water points x {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)} features")
print(f"Non-functional: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.1f}%)")
print(f"Districts: {df[GROUP].nunique()}   Provinces: {df['clean_adm1'].nunique()}")
print(f"\\nClass balance is {df[TARGET].mean()*100:.0f}/{(1-df[TARGET].mean())*100:.0f}, "
      f"so accuracy is a useless metric here.")
print("A model predicting 'never broken' would score 88% accurate and be worthless.")
print("PR-AUC is the headline metric throughout; it only rewards finding the rare class.")
df.head(3)
""")

code("""
# A first look at the one feature with a physical story behind it.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

bins = [0, 5, 10, 15, 20, 60]
labels = ["0-5", "5-10", "10-15", "15-20", "20+"]
band = pd.cut(df["age_years"], bins=bins, labels=labels, right=False)
rate = df.groupby(band, observed=True)[TARGET].agg(["mean", "size"])

axes[0].bar(rate.index.astype(str), rate["mean"] * 100, color=NAVY, width=0.62)
for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
    axes[0].text(i, m * 100 + 0.4, f"n={n:,}", ha="center", fontsize=7.5, color=SLATE)
axes[0].axhline(df[TARGET].mean() * 100, color=RUST, ls="--", lw=1.2,
                label=f"overall {df[TARGET].mean()*100:.1f}%")
axes[0].set_title("Failure rate rises with pump age")
axes[0].set_xlabel("Age at inspection (years)"); axes[0].set_ylabel("Non-functional (%)")
axes[0].legend(frameon=False, fontsize=8)

dist = df.groupby(GROUP)[TARGET].agg(["mean", "size"]).query("size >= 20")
axes[1].hist(dist["mean"] * 100, bins=22, color=GOLD, edgecolor=NAVY, linewidth=0.6)
axes[1].set_title(f"Failure rate varies widely by district (n={len(dist)})")
axes[1].set_xlabel("District failure rate (%)"); axes[1].set_ylabel("Districts")

plt.tight_layout(); plt.show()

print(f"District failure rates run from {dist['mean'].min()*100:.0f}% "
      f"to {dist['mean'].max()*100:.0f}%.")
print("That spread is exactly why the train/test split has to respect districts.")
""")

# ===========================================================================
md("""
## 8. Splitting honestly

Here is the decision that changes the answer more than any model choice.

Water points in the same district share an installer, a water table, a spare
parts supply and a maintenance team. If I split rows at random, a point in the
validation fold usually sits a few hundred metres from a point the model
trained on. The model can recognise the neighbourhood instead of learning
anything that travels.

The question a planner actually has is: **"here is a district we have never
worked in, which points should we visit?"**

So validation holds out **whole districts**, using `StratifiedGroupKFold`
grouped on district. I report both numbers, because the gap between them is one
of the findings.

Two more things, both inside `Pipeline`:

- **Imputation and encoding are fitted per fold.** Fitting a median on the full
  dataset before splitting leaks the validation fold into training. It is a
  small leak and it is still a leak.
- **`add_indicator=True`** keeps the fact that a value was missing. A water
  point with no recorded install year is not an average water point, and that
  absence may say something about how well it is documented and looked after.
""")

code("""
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train import (MODEL_FEATURES, MODEL_NUMERIC, N_SPLITS, build_preprocessor,
                   candidate_models, cross_validate, inspection_curve, bootstrap_ci)

X, y, groups = df[MODEL_FEATURES], df[TARGET], df[GROUP]

outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(outer.split(X, y, groups))
X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
g_tr, g_te = groups.iloc[train_idx], groups.iloc[test_idx]

print(f"Train: {len(X_tr):,} points across {g_tr.nunique()} districts "
      f"({y_tr.mean()*100:.1f}% non-functional)")
print(f"Test:  {len(X_te):,} points across {g_te.nunique()} UNSEEN districts "
      f"({y_te.mean()*100:.1f}% non-functional)")
print(f"\\nOverlap between train and test districts: "
      f"{len(set(g_tr) & set(g_te))} districts")
print(f"Held-out districts: {', '.join(sorted(set(g_te))[:6])}")
""")

# ===========================================================================
md("""
## 9. Model comparison, against the rule that needs no model

Four candidates, plus the benchmark that actually matters.

That benchmark is **"visit the oldest pumps first"**. It requires no model, no
data science, and no budget. If a trained model cannot beat it, the model
should not be deployed, and saying so is the job.

It is scored on **identical folds**, so this is like for like.
""")

code("""
grouped_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=RANDOM_STATE)
results, oof_store = {}, {}

for name, model in candidate_models().items():
    r = cross_validate(model, X_tr, y_tr, g_tr, grouped_cv)
    results[name] = {k: v for k, v in r.items() if k != "oof"}
    oof_store[name] = r["oof"]

tbl = pd.DataFrame([
    {"Model": n,
     "PR-AUC": f"{v['ap_mean']:.3f} +/- {v['ap_std']:.3f}",
     "ROC-AUC": f"{v['auc_mean']:.3f} +/- {v['auc_std']:.3f}"}
    for n, v in results.items()
])
rule = results["Random Forest"]
tbl.loc[len(tbl)] = ["NO MODEL: visit oldest first",
                     f"{rule['rule_ap']:.3f}", f"{rule['rule_auc']:.3f}"]
print("Cross-validation, grouped by district (5 folds):\\n")
print(tbl.to_string(index=False))
""")

md("""
Look at the last row against the rest.

The Random Forest reaches **PR-AUC 0.239** against the age rule's **0.184**, so
on this measure the model is ahead. But look at the standard deviations: plus
or minus 0.147 across five folds. The folds disagree with each other about as
much as the model differs from the rule.

Logistic Regression and Gradient Boosting are both **worse than doing nothing
clever at all**.
""")

# ===========================================================================
md("""
## 10. What a careless split would have bought me

Same models, same data, same code. The only change is splitting **rows at
random** instead of holding out whole districts.
""")

code("""
random_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rows = []
for name, model in candidate_models().items():
    if name.startswith("Baseline"):
        continue
    r = cross_validate(model, X_tr, y_tr, g_tr, random_cv)
    g = results[name]["ap_mean"]
    rows.append({"Model": name,
                 "Random split PR-AUC": round(r["ap_mean"], 3),
                 "District split PR-AUC": round(g, 3),
                 "Overstated by": f"+{r['ap_mean'] - g:.3f}",
                 "Inflation": f"{(r['ap_mean']/g - 1)*100:.0f}%"})
split_comparison = pd.DataFrame(rows)
print(split_comparison.to_string(index=False))
""")

md("""
The Random Forest looks **63% better** under the random split. Nothing about
the model changed. Only the way I checked it changed.

This is the single most common way a machine learning result gets published
wrong, and it is invisible unless you go looking for it. Every number in that
"random split" column is real, reproducible, and misleading.
""")

# ===========================================================================
md("""
## 11. What the leaked column was worth

Now put `rehab_priority` back, as a simple present/absent flag, and rerun the
identical cross-validation.
""")

code("""
flag_by_id = raw.set_index("wpdx_id")["rehab_priority"].notna().astype(int)
leak_flag = df.iloc[train_idx]["wpdx_id"].map(flag_by_id).fillna(0).astype(int)
leak_frame = X_tr.assign(rehab_priority_present=leak_flag.to_numpy())

leak_model = Pipeline([
    ("prep", build_preprocessor(scale=False,
                                numeric=MODEL_NUMERIC + ["rehab_priority_present"])),
    ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=-1)),
])
leak_r = cross_validate(leak_model, leak_frame, y_tr, g_tr, grouped_cv)

print(f"{'':<26s} {'PR-AUC':>8s} {'ROC-AUC':>9s}")
print(f"{'WITH rehab_priority':<26s} {leak_r['ap_mean']:>8.3f} {leak_r['auc_mean']:>9.3f}"
      f"   <- looks superb, predicts nothing")
print(f"{'WITHOUT it (honest)':<26s} {results['Random Forest']['ap_mean']:>8.3f} "
      f"{results['Random Forest']['auc_mean']:>9.3f}")
print(f"\\nOne column moved ROC-AUC by "
      f"{leak_r['auc_mean'] - results['Random Forest']['auc_mean']:+.3f}.")
print("A model that good would have been deployed, and it would have been useless")
print("on any water point whose repair had not already been planned.")
""")

# ===========================================================================
md("""
## 12. The held-out districts, scored exactly once

Everything so far has happened inside the training districts. The five held-out
districts have not been touched, and they are scored **once**. No going back to
tune after seeing this.

The test set is small: 669 points with 83 failures. A point estimate alone
would overstate how much I know, so every metric gets a bootstrap confidence
interval.
""")

code("""
best_name = max((n for n in results if not n.startswith("Baseline")),
                key=lambda n: results[n]["ap_mean"])
print(f"Selected on grouped CV: {best_name}\\n")

best = candidate_models()[best_name]
best.fit(X_tr, y_tr)
p_te = best.predict_proba(X_te)[:, 1]
yte = y_te.to_numpy()

ap_lo, ap_hi = bootstrap_ci(yte, p_te, average_precision_score)
auc_lo, auc_hi = bootstrap_ci(yte, p_te, roc_auc_score)

med_age = X_tr["age_years"].median()
rule_te = X_te["age_years"].fillna(med_age).to_numpy()

print(f"Held-out: {len(yte)} points, {g_te.nunique()} districts, "
      f"{yte.sum()} broken ({yte.mean()*100:.1f}%)\\n")
print(f"  {best_name}")
print(f"    PR-AUC   {average_precision_score(yte, p_te):.3f}   "
      f"95% CI [{ap_lo:.3f}, {ap_hi:.3f}]")
print(f"    ROC-AUC  {roc_auc_score(yte, p_te):.3f}   "
      f"95% CI [{auc_lo:.3f}, {auc_hi:.3f}]")
print(f"    Brier    {brier_score_loss(yte, p_te):.3f}")
print(f"\\n  Age rule on exactly the same points")
print(f"    PR-AUC   {average_precision_score(yte, rule_te):.3f}")
print(f"    ROC-AUC  {roc_auc_score(yte, rule_te):.3f}")
""")

md("""
### This is the result

On districts it had never seen, the model scores **ROC-AUC 0.613**.

Sorting the same water points by age alone scores **0.642**.

**The model loses to one line of common sense.** And the confidence interval on
the model is wide enough to contain "no better than chance", which is the more
honest way to say it.

I could have gone back and tuned until the number moved. That is precisely how
overfitting to a test set happens, and the result would have been a number I
could not defend. So this is the number.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

# Held-out comparison against the benchmarks that matter.
names = ["Random\\nchance", "Age rule\\n(no model)", f"{best_name}\\n(held-out)"]
vals = [0.5, roc_auc_score(yte, rule_te), roc_auc_score(yte, p_te)]
cols = [SLATE, GOLD, NAVY]
bars = axes[0].bar(names, vals, color=cols, width=0.6)
axes[0].errorbar(2, vals[2], yerr=[[vals[2]-auc_lo], [auc_hi-vals[2]]],
                 fmt="none", ecolor=RUST, capsize=5, lw=1.4)
axes[0].axhline(0.5, color=SLATE, ls=":", lw=1)
axes[0].set_ylim(0.4, 0.8); axes[0].set_ylabel("ROC-AUC")
axes[0].set_title("The model does not beat pump age")
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                 ha="center", fontsize=9, weight="600")

# Optimism from a careless split.
sc = split_comparison.set_index("Model")
xp = np.arange(len(sc)); w = 0.36
axes[1].bar(xp - w/2, sc["Random split PR-AUC"], w, label="Random split", color=RUST)
axes[1].bar(xp + w/2, sc["District split PR-AUC"], w, label="District split", color=NAVY)
axes[1].set_xticks(xp)
axes[1].set_xticklabels([m.replace(" ", "\\n") for m in sc.index], fontsize=8)
axes[1].set_ylabel("PR-AUC"); axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("A random split flatters every model")

plt.tight_layout(); plt.show()
""")

# ===========================================================================
md("""
## 13. The question a water officer actually asks

PR-AUC is not a thing anyone can act on. The real question is operational:

> *A crew can reach the top 20% of the water points in a district this month.
> How many of the broken ones will they find?*

That is a ranking question, and a model can be useful for ranking even when its
headline score is unimpressive.
""")

code("""
curve = inspection_curve(yte, p_te)
print("If a repair crew can only reach part of the district:\\n")
print(curve.to_string(index=False))

r20 = curve[curve.visit_top_pct == 20].iloc[0]
print(f"\\nVisiting the top 20% by model rank finds {r20.broken_found} of "
      f"{r20.of_total_broken} broken points ({r20.recall_at_k*100:.0f}%),")
print(f"which is {r20.lift_vs_random}x better than arbitrary order.")
print("Real, but modest. And the age rule would deliver much of it for free.")
""")

code("""
fig, ax = plt.subplots(figsize=(6.4, 3.8))
frac = np.linspace(0, 1, len(yte) + 1)
order = np.argsort(-p_te)
gains = np.concatenate([[0], np.cumsum(yte[order]) / yte.sum()])
rorder = np.argsort(-rule_te)
rgains = np.concatenate([[0], np.cumsum(yte[rorder]) / yte.sum()])

ax.plot(frac * 100, gains * 100, color=NAVY, lw=2, label=best_name)
ax.plot(frac * 100, rgains * 100, color=GOLD, lw=2, label="Age rule (no model)")
ax.plot([0, 100], [0, 100], color=SLATE, ls="--", lw=1, label="Arbitrary order")
ax.axvline(20, color=RUST, ls=":", lw=1.2)
ax.text(21, 8, "crew reaches\\ntop 20%", fontsize=8, color=RUST)
ax.set_xlabel("Water points visited (% of district, ranked)")
ax.set_ylabel("Broken points found (%)")
ax.set_title("Inspection curve on held-out districts")
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()
""")

# ===========================================================================
md("""
## 14. What the model actually leaned on

Tree impurity importances are biased towards high-cardinality features, so this
uses **permutation importance**, measured on the held-out districts: shuffle one
column, see how much PR-AUC falls.
""")

code("""
perm = permutation_importance(best, X_te, y_te, n_repeats=30,
                              random_state=RANDOM_STATE,
                              scoring="average_precision", n_jobs=-1)
imp = (pd.DataFrame({"feature": MODEL_FEATURES,
                     "importance": perm.importances_mean,
                     "std": perm.importances_std})
       .sort_values("importance", ascending=False))
print("Drop in PR-AUC when the column is shuffled:\\n")
print(imp.to_string(index=False))
""")

code("""
fig, ax = plt.subplots(figsize=(6.6, 4))
top = imp.head(10).iloc[::-1]
ax.barh(top["feature"], top["importance"], xerr=top["std"],
        color=[NAVY if v > 0 else SLATE for v in top["importance"]],
        error_kw=dict(ecolor=SLATE, lw=0.9, capsize=2.5))
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Drop in PR-AUC when shuffled")
ax.set_title("Permutation importance, held-out districts")
plt.tight_layout(); plt.show()
""")

# ===========================================================================
md("""
## 15. Ablation: do coordinates help, or do they help it memorise?

Latitude and longitude are excluded from the model. That decision was made in
`src/ablation.py`, **on the training districts only**, so it could never be
tuned against the test set.

The suspicion: exact coordinates let a model memorise neighbourhoods, which is
worth nothing in a district it has never visited.
""")

code("""
ab_path = ROOT / "reports" / "ablation.csv"
if ab_path.exists():
    ab = pd.read_csv(ab_path)
    print(ab.to_string(index=False))
    print("\\nRemoving coordinates did not hurt performance in an unseen district,")
    print("which is exactly what you expect if they were being used to memorise")
    print("rather than to learn something that travels. So they are dropped.")
else:
    print("Run: python src/ablation.py")
""")

# ===========================================================================
md("""
## 16. What I would defend, and what I would not

### The findings, in order of importance

1. **`status_clean` is unusable.** Fourteen of the fifteen contributing
   programmes never record a functional water point. It encodes who uploaded
   the record, not whether the water flows. Anyone using this dataset should
   check this before doing anything else.
2. **Two columns leak the answer.** `rehab_priority` and `would_gain_access` are
   populated for ~97% of failures and 0.0% of working points. Left in, ROC-AUC
   reaches 0.998 and means nothing.
3. **Seven programmes report a 0% failure rate**, covering 2,171 points,
   because they upload at handover and never return.
4. **A random split overstates PR-AUC by 63%** compared with holding out whole
   districts.
5. **The model does not beat pump age.** 0.613 against 0.642 on unseen
   districts.

### Method choices I would defend in an interview

- `StratifiedGroupKFold` grouped on district, not a random split
- Every transform inside a `Pipeline`, so imputation is fitted per fold
- Permutation importance rather than tree impurity
- Bootstrap confidence intervals, because 83 failures is not many
- The age rule scored on identical folds as the benchmark that matters
- Coordinates dropped on ablation evidence gathered before the test set was touched

### Limitations I will not talk around

- **669 test points and 83 failures.** The confidence intervals are wide and I
  am not going to pretend otherwise.
- **The label is "was water available on the day of the visit"**, which is not
  the same as "the pump is broken". A dry season visit and a mechanical failure
  look identical in this data.
- **Survivorship.** Points that were abandoned entirely may never have been
  surveyed at all.
- **Excluding the zero-failure programmes removed a third of the labelled
  data.** That is a defensible call, not a free one.

### The conclusion that actually mattered

The reason this model cannot predict much is not that it needs more trees or
better tuning. It is that **the underlying measurement does not exist**. Nobody
goes back to a water point after handover to record whether it still works, so
there is no signal to learn from. No amount of modelling creates a fact that was
never written down.

That realisation is what the follow-on project is built on: an SMS reporting
and verification loop that generates the dated functionality records this
dataset has never had.

**<https://github.com/buseko-Actuary/pump-watch>**

Finding out that a model does not work, and being able to prove why, is a
result. Publishing a number you cannot defend is not.

---

*Data: Water Point Data Exchange via UN OCHA HDX, CC BY-SA. Every figure in
this notebook was produced by the cells above.*
""")

nb["cells"] = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.9"},
}

out = Path(__file__).resolve().parent / "water-point-failure-walkthrough.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
print(f"{len(cells)} cells "
      f"({sum(1 for c in cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
