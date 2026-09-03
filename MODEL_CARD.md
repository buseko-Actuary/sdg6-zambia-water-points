# Model card: rural water point failure ranking, Zambia

## Model details

- **Task:** binary classification, used as a ranking. Estimates the probability
  that a rural water point is non-functional.
- **Model:** scikit-learn `Pipeline`: median imputation with missingness
  indicators and constant-fill one-hot encoding, then a `RandomForestClassifier`
  (500 trees, `min_samples_leaf=5`, `class_weight="balanced_subsample"`).
- **Selected by:** 5-fold `StratifiedGroupKFold` grouped on district, on
  training districts only.
- **Features (13):** age at inspection, local population within 1 km, distances
  to city, town, primary, secondary and tertiary road, water source, water
  technology, source and technology category, province, urban/rural flag.
- **Version:** 1.0, 3 September 2026. Seed 42 throughout.

## Intended use

**Intended:** ordering maintenance and inspection visits when a district has
more water points than it can reach. The output is a priority list.

**Users:** district water officers, WASH programme staff, NGO maintenance
planners, and researchers working on rural water reliability.

## Out of scope

Do not use this model to:

- **Decide funding or resource allocation for a named community.** ROC-AUC is
  0.613 with a lower 95% bound of 0.543. It is close enough to chance that an
  individual prediction carries very little information.
- **Report a functionality rate.** It has not been validated for prevalence
  estimation, and `class_weight="balanced"` makes the raw probabilities too
  high.
- **Replace an inspection.** It orders visits; it does not substitute for one.
- **Generalise beyond rural Zambia.** It was trained on Zambian records only,
  and was not tested anywhere else.
- **Assess urban or piped networks.** WPdx covers rural point sources.

## Training data

4,026 water points across 61 districts and 10 provinces, from the Water Point
Data Exchange via UN OCHA HDX, recorded 2012 to 2022. 476 (11.8%) were
non-functional.

Derived from a 6,643-row export by:

- dropping 446 records whose status was `Unknown`;
- dropping 2,171 records from six contributing programmes that report a 0%
  failure rate, because programme identity would otherwise predict the target;
- excluding `status_clean` (corrupted: only one of fifteen programmes ever
  records `Functional`), `rehab_priority` and `would_gain_access` (present for
  ~97% of failures and 0% of working points), programme-identity columns, and
  four WPdx service-allocation columns excluded as a precaution.

Full reasoning in `src/data_prep.py`.

## Evaluation

Five entire districts held out, 669 water points, 83 non-functional. These
districts appear in no training or selection step.

| Metric | Value | 95% CI (bootstrap, 2000 resamples) |
|---|---|---|
| PR-AUC | 0.200 | [0.144, 0.284] |
| ROC-AUC | 0.613 | [0.543, 0.678] |
| Brier score | 0.170 | |

Baselines on the same points: random guessing gives PR-AUC 0.124 and ROC-AUC
0.500. **Sorting by pump age gives PR-AUC 0.191 and ROC-AUC 0.642**, so the
model does not beat the simplest available heuristic on ROC-AUC.

Operationally, inspecting the top-ranked 20% of a district finds 37% of its
broken water points, about 1.9 times an arbitrary visiting order.

## Ethical considerations

- **Reporting bias is in the data and partly in the model.** Contributing
  organisations survey where they work. Districts and communities that no NGO
  has surveyed are absent entirely, and they are plausibly the least served.
  A model built on this data can only ever describe the surveyed subset.
- **A low score is not evidence that a water point works.** If this ranking
  were used to *skip* inspections rather than to order them, the communities
  systematically pushed to the bottom would be the ones least able to complain.
- **The data is about communities, not individuals.** It contains no personal
  data. Coordinates identify infrastructure, not households.
- **Age drives the ranking**, so a model-ordered queue will tend to favour
  older installations. Where older water points serve particular areas, this
  reproduces that pattern. It should be checked against equity criteria before
  any operational use.

## Caveats

- Poorly calibrated: predicted probabilities are systematically too high.
  Use the ordering, not the numbers.
- One observation per water point; no repair history, water table depth, tariff
  or committee data. Those are the likely drivers of failure and are absent.
- 2012 to 2022 records are pooled as one cohort with no time-aware validation.
- 476 positives across 61 districts gives fold-to-fold PR-AUC standard
  deviations up to 0.15. Differences between model families in this repository
  are not statistically distinguishable.
