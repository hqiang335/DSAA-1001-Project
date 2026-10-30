# Chicago Food Inspections

The **final version** contains executed data-quality analysis, exploratory figures, and the team's Modeling Plan. No predictive models have been trained; the validation and model-comparison procedures remain proposed work.

- [Final version](data_analysis_final_version.ipynb): the submission notebook, with complete findings, limitations, and modelling implications.
- [Chinese working notebook](data_analysis_final_中文版.ipynb): the retained discussion version, including the team's Modeling Plan.
- [Data snapshot](data/food_inspections_snapshot.zip) and [metadata](data/snapshot_metadata.json): the fixed data used to reproduce the analysis.

## Data and Definitions

The source is [Chicago Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5), downloaded on October 12, 2026. The requested range was January 1, 2016–October 30, 2026; the actual snapshot contains **197,882 records and 17 columns, from January 4, 2016 to October 9, 2026**. The filename's end date does not imply data coverage through that date.

- Each row represents an inspection. Failure rate is `Fail / (Fail + Pass + Pass w/ Conditions)`; other outcomes are not treated as passes.
- Identity-based EDA uses valid license, normalized address, DBA, and AKA as an establishment proxy. This is not a verified business identity.
- Historical EDA uses dates strictly before each current inspection; same-day records do not borrow outcomes from one another. Current `Violations` and its derived features are excluded from predictors.
- EDA includes 2026. These explored records cannot be described as a completely unseen final test. Associations are not evidence of causality or validated predictive gains.
- The team's Modeling Plan proposes valid `License #` as the primary history key, whereas the EDA uses the more conservative composite proxy. This identity-rule difference requires resolution before implementing the model.

## Notebook Sections

| Section | Purpose |
|---|---|
| 1. Data Source and Overview | Read the snapshot and verify dimensions, actual date coverage, and key fields. |
| 2. Data Quality Analysis | Examine missing Facility Type, Risk, and AKA separately; review category formatting, identity ambiguity, geography, target outcomes, leakage, and brand-name variants. Summarize problem, evidence, possible reason, and effect on modelling. |
| 3.1 Group Differences | Compare Facility Type, Inspection Type, and Risk failure rates and their relevance as candidate features. |
| 3.2 Repeated Inspections | Count distinct observed dates per establishment proxy and assess history coverage. |
| 3.3 Brand Candidates | Compare within-brand proxy distributions and weighting choices for six name-matched brands, with coverage limitations. |
| 3.4 Temporal Change | Present annual counts and monthly volume/rates, discuss the 2018 reform and pandemic context, and motivate temporal validation. |
| 3.5 Inspection History | Examine historical failure rates, previous outcomes alongside current inspection processes, and inspection intervals. |
| 3.6 Spatial Distribution | Compare establishment coverage, inspection-weighted rates, and equal-proxy rates; propose testing coordinates beyond ZIP. |
| 4. Modeling Plan | Define the target, Logistic Regression baseline, historical features, temporal validation, feature ablation, and Macro-F1 evaluation. |

## Figure Register

Numbering matches the notebook. Figure 11 contains three panels.

| Figure | Meaning and interpretation |
|---|---|
| 1. Facility Type | Failure rates for the eight largest facility categories. Differences motivate candidate features but do not establish predictive contribution. |
| 2. Inspection Type | Failure rates for the eight largest inspection categories. Inspection reasons and processes are associated with outcomes. |
| 3. Risk | Failure rates for three regulatory classifications. Risk is not a historical hygiene-performance score. |
| 4. Repeated Inspection Dates | Distribution of distinct inspection dates per proxy, showing repeated observation and potential history coverage. |
| 5. Within-Brand Variation | Blue dots are proxy failure rates; orange diamonds are inspection-weighted means; hollow circles are equal-proxy means. Requires at least 15 labelled records per proxy and eight eligible proxies per brand. Brand averages do not represent every branch. |
| 6. Monthly Inspection Volume | All inspection records by month, marking the 2018 reform and 2020–2021. Temporal coincidence does not establish causality. |
| 7. Monthly Failure Rate | Monthly failure rates and the inspection-weighted reference for the plotted period. Figures 6–7 end in September 2026, excluding incomplete October. |
| 8 (History A). Historical Failure Rate | Current rates by prior failure-rate group, requiring at least five prior labelled inspections. More history does not automatically establish stronger statistical significance. |
| 9 (History B). Previous Outcome and Current Process | Failure rates by previous Fail/non-Fail and current reinspection/non-reinspection, separating historical associations from process composition. |
| 10 (History C). Inspection Interval | Current failure rates across time gaps, with a non-reinspection comparison. The association is not strictly monotonic and does not demonstrate diligence or complacency. |
| 11A. Establishment Coverage | Distinct observed proxies per approximately 2 km cell, on a logarithmic scale; not inspection volume or the density of all operating establishments. |
| 11B. Inspection-Weighted Failure Rate | Failures divided by labelled inspections, giving each inspection equal weight and matching an inspection-level prediction task. |
| 11C. Equal-Proxy Failure Rate | Average of individual proxy rates, assessing sensitivity to frequent inspections; not a corrected true-risk estimate. |

Maps 11B/C use identical cells, records, and colour scales. Each displayed cell requires at least 100 labelled inspections and 30 proxies. Grey means insufficient support; white means no observations. Neither means zero risk. The notebook retains the 2 km three-panel comparison.

## Reproduce the Analysis

Use Python 3.12 and run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m zipfile -e data/food_inspections_snapshot.zip data/
jupyter lab
```

Open `data_analysis_final_version.ipynb`, select a Python kernel, and run all cells. It reads the extracted CSV through a relative path and does not download live data. Extraction requires approximately 206 MiB. The metadata includes the original CSV SHA-256; a fresh portal download may contain revised records and produce different results.

The final version has 56 executed code cells and 11 numbered figures. Its numerical analysis and figures are retained from the Chinese working notebook; narrative and the interval sample-size note are translated. HTML previews and supporting research files remain local and are not included in this repository.
