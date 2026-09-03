"""
Data preparation for the Zambia water point functionality model.

Reads the raw WPdx export, resolves the label, removes columns that would leak
the answer, and writes a modelling table to data/processed/.

Every exclusion below is deliberate and is explained in EXCLUSIONS. Nothing is
dropped silently.

Run:  python src/data_prep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wpdx_water_points_zmb.csv"
PROCESSED = ROOT / "data" / "processed"

RANDOM_STATE = 42

# Reasons a column never reaches the model. Printed at run time so the audit
# trail lives in the output, not only in the README.
EXCLUSIONS = {
    "status_clean": (
        "CORRUPTED LABEL. Only one of the 15 contributing programmes ever "
        "records 'Functional'. Every other programme has all of its points "
        "labelled 'Non-Functional', including programmes where status_id says "
        "water is available for 100% of rows. It is a missing value encoded as "
        "a category, so it measures who uploaded the record, not whether the "
        "water point works."
    ),
    "rehab_priority": (
        "TARGET LEAKAGE. Populated for 96.8% of non-functional points and 0.0% "
        "of functional ones. It is a rehabilitation queue built after the "
        "status was known."
    ),
    "would_gain_access": (
        "TARGET LEAKAGE. Populated for 99.6% of non-functional points and 0.0% "
        "of functional ones."
    ),
    "subjective_quality": (
        "LEAKAGE RISK. Recorded by the same enumerator on the same visit that "
        "produced the status. 64% missing and not independently verifiable."
    ),
    "assigned_population,pressure,criticality,usage_cap": (
        "WPdx service-allocation outputs. These are computed by WPdx rather "
        "than observed in the field, and the allocation logic may already know "
        "which points are working. Excluded as a conservative choice; see the "
        "limitations section of the README."
    ),
    "source,dataset_title,created_timestamp,activity_id,wpdx_id,scheme_id,installer,rehabilitator": (
        "PROGRAMME IDENTITY. Failure rates run from 0% to 16% across "
        "programmes for reporting reasons, not physical ones. Including these "
        "lets the model predict failure from who filed the record. The task is "
        "to predict from the characteristics of the water point."
    ),
    "days_since_report,staleness": (
        "Both are functions of the upload date, so they encode programme "
        "vintage rather than anything about the water point."
    ),
    "notes": "Free text, 81% missing, and may quote the status directly.",
    "prediction_yes_0y,prediction_yes_2y,prediction_no_0y,prediction_no_2y,"
    "predicted_status_0y,predicted_status_2y,predicted_category": (
        "WPdx's own model output. Entirely empty in the Zambia export, but "
        "dropped explicitly so a future refresh cannot quietly reintroduce it."
    ),
    "management_clean,pay_clean,facility_type,clean_adm3,clean_adm4,rehab_year,"
    "fecal_coliform_presence,fecal_coliform_value": (
        "Constant or almost entirely empty in the Zambia export, so they carry "
        "no usable signal."
    ),
}

NUMERIC_FEATURES = [
    "age_years",
    "local_population",
    "distance_to_city",
    "distance_to_town",
    "distance_to_primary",
    "distance_to_secondary",
    "distance_to_tertiary",
    "lat_deg",
    "lon_deg",
]

CATEGORICAL_FEATURES = [
    "water_source_clean",
    "water_tech_clean",
    "water_source_category",
    "water_tech_category",
    "clean_adm1",
    "is_urban",
]

TARGET = "is_non_functional"
GROUP = "clean_adm2"

# The earliest plausible installation year. The raw file contains 1902, which
# predates any borehole programme in the country by decades.
MIN_INSTALL_YEAR = 1950


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Read the WPdx export. Row 2 of the file is HXL tags, not data."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See the README for the download command."
        )
    return pd.read_csv(path, skiprows=[1], low_memory=False)


def build_label(df: pd.DataFrame) -> pd.DataFrame:
    """status_id is the field-observed answer: was water available on the visit?

    'Unknown' is dropped rather than guessed. Treating it as either class would
    invent 446 observations that nobody made.
    """
    out = df[df["status_id"].isin(["Yes", "No"])].copy()
    out[TARGET] = (out["status_id"] == "No").astype(int)
    return out


def drop_single_class_programmes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove programmes that never record a failure.

    Seven of the fourteen programmes that carry a usable label report a 0%
    failure rate, covering 2,171 water points. That is a reporting convention:
    those
    programmes upload points at handover, when everything works by definition.
    A programme with no failures carries no information about failure, and
    leaving it in lets any model infer risk from programme identity through
    correlated features such as district or installation year.
    """
    rates = (
        df.groupby("dataset_title")[TARGET]
        .agg(n="size", failures="sum")
        .assign(failure_rate=lambda d: (d.failures / d.n).round(4))
        .sort_values("n", ascending=False)
    )
    keep = rates.index[rates["failures"] > 0]
    return df[df["dataset_title"].isin(keep)].copy(), rates


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the handful of features that are not already in the file."""
    out = df.copy()
    out["report_year"] = pd.to_datetime(out["report_date"], errors="coerce").dt.year

    install = pd.to_numeric(out["install_year"], errors="coerce")
    # An install year before 1950 or after the visit is a data entry error, not
    # a very old pump. Blanked so it is imputed rather than believed.
    implausible = (install < MIN_INSTALL_YEAR) | (install > out["report_year"])
    out["install_year_clean"] = install.where(~implausible)

    # Age at the time of inspection. The single most physically meaningful
    # feature available: pumps wear out.
    out["age_years"] = out["report_year"] - out["install_year_clean"]

    out["is_urban"] = out["is_urban"].astype("string")
    return out


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    print(f"Raw export:                {raw.shape[0]:>6,} rows x {raw.shape[1]} columns")

    labelled = build_label(raw)
    dropped_unknown = len(raw) - len(labelled)
    print(f"Dropped status 'Unknown':  {dropped_unknown:>6,} rows")

    modelling, programme_rates = drop_single_class_programmes(labelled)
    print(f"Dropped 0%-failure progs:  {len(labelled) - len(modelling):>6,} rows")

    modelling = engineer_features(modelling)

    # wpdx_id is kept as a row identifier only. It is never a feature; it makes
    # the processed table traceable back to the raw export, which the leakage
    # demonstration in train.py relies on.
    keep_cols = (NUMERIC_FEATURES + CATEGORICAL_FEATURES
                 + [TARGET, GROUP, "report_year", "wpdx_id"])
    final = modelling[keep_cols].copy()

    print(f"\nModelling table:           {final.shape[0]:,} rows x "
          f"{len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)} features")
    print(f"Non-functional:            {final[TARGET].sum():,} "
          f"({final[TARGET].mean() * 100:.1f}%)")
    print(f"Districts (group unit):    {final[GROUP].nunique()}")
    print(f"Provinces:                 {final['clean_adm1'].nunique()}")

    print("\nMissingness in retained features:")
    miss = final[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().mean().mul(100)
    for col, pct in miss.sort_values(ascending=False).items():
        if pct > 0:
            print(f"  {col:<24s} {pct:5.1f}%")

    print("\nColumns deliberately excluded:")
    for cols, reason in EXCLUSIONS.items():
        head = cols if len(cols) < 60 else cols[:57] + "..."
        print(f"  {head}\n      {reason[:110]}...")

    final.to_csv(PROCESSED / "water_points_model_table.csv", index=False)
    programme_rates.to_csv(PROCESSED / "programme_failure_rates.csv")
    print(f"\nWrote {PROCESSED / 'water_points_model_table.csv'}")
    print(f"Wrote {PROCESSED / 'programme_failure_rates.csv'}")


if __name__ == "__main__":
    main()
