# 🛡️ Shield Dashboard: Predicting Academic Staff Reductions in Public Universities

**An Interpretable Machine-Learning Early-Warning System Using Longitudinal IPEDS Data**

[![Live App](https://img.shields.io/badge/streamlit-live%20app-FF4B4B?logo=streamlit&logoColor=white)](https://academic-staff-safeguard.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Model](https://img.shields.io/badge/model-Random%20Forest-2E8B57)

**Authors:** Jason Avalos · Luigi Salemi · Jee Hun Hwang \
Master of Science in Applied Data Science, Shiley Marcos School of Engineering — University of San Diego

**Live Dashboard:** [academic-staff-safeguard.streamlit.app](https://academic-staff-safeguard.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Working Hypothesis](#working-hypothesis)
- [Data](#data)
- [Methodology](#methodology)
- [Modeling](#modeling)
- [Results](#results)
- [Key Findings](#key-findings)
- [The Shield Dashboard](#the-shield-dashboard)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Recommended Next Steps](#recommended-next-steps)
- [Acknowledgments](#acknowledgments)
- [Selected References](#selected-references)
- [License](#license)

---

## Overview

Public four-year universities are experiencing sustained fiscal instability — declining enrollments, unpredictable state and federal funding, and rising operational costs. Most institutions rely on retrospective accounting reports that only surface distress *after* instructional quality and staffing continuity have already been compromised.

This project builds an interpretable early-warning system that predicts, 18–24 months in advance, whether a public four-year institution is at elevated risk of a significant reduction in instructional full-time-equivalent (FTE) academic staff. It uses an eleven-year longitudinal panel (2014–2024) built from IPEDS survey data and a Random Forest classifier trained under a leakage-safe, chronological validation scheme, paired with a Streamlit decision-support tool called the **Shield Dashboard**.

**Primary users:** Provosts, CFOs, institutional research teams, and HR workforce planners who need to move from reactive, after-the-fact staffing cuts to proactive, evidence-based workforce planning.

## Problem Statement

Academic advisors, program coordinators, financial aid specialists, student affairs professionals, and instructional support staff are typically the first roles cut during financial hardship — despite being the backbone of student retention, persistence, and graduation outcomes. IPEDS provides rich longitudinal financial, staffing, and enrollment data, but it is almost never used *operationally* to forecast which institutions are about to reduce academic staff. This project closes that gap by treating year-over-year staffing reduction — not institutional closure — as the prediction target, and by delivering per-institution, SHAP/impurity-based explanations rather than a single opaque risk score.

## Working Hypothesis

> An institution's historical ratio of **Institutional Support** (administrative overhead) to **Instructional Spend** reliably predicts whether it will enter a high- or low-risk category for significant reductions in core academic FTE staff during future economic downturns.

**Product Promise:** The Shield Dashboard delivers an interpretable, data-driven early-warning system that empowers university leadership to anticipate workforce vulnerability 18–24 months in advance, enabling strategic cost optimization rather than reactionary layoffs.

## Data

- **Source:** U.S. Department of Education's National Center for Education Statistics (NCES), via the Integrated Postsecondary Education Data System (IPEDS).
- **Coverage:** 2014–2024, public four-year institutions only.
- **Components used:** Institutional Characteristics, Finance, Human Resources, and Fall Enrollment (EF) survey files.
- **Scale:** Several hundred raw variables per institution-year, reduced through cleaning, leakage removal, and collinearity screening to a smaller set of engineered predictors.

### Data Cleaning Highlights

- Standardized inconsistent column headers and naming conventions across annual files (including the provisional 2023/2024 formats).
- Converted IPEDS' period-based missing-value markers to proper nulls.
- Dropped columns with >50% missingness (unrecoverable) and any variable calculated from future years (leakage risk).
- Median imputation for remaining numeric gaps (chosen over mean due to heavy right-skew in financial variables — mean total revenue ≈ $433M vs. median ≈ $151M).
- Correlation filter (>0.95 pairwise) + iterative VIF screen (threshold = 10) to remove redundant predictors, with a protected set of hypothesis-relevant variables retained regardless of screening thresholds.
- All preprocessing (imputation, scaling, encoding, collinearity screening) fit on the training partition only, then applied unchanged to validation/test — preventing information from later years leaking into earlier predictions.

## Methodology

The project follows a **seven-stage longitudinal supervised-learning pipeline** designed specifically to prevent target leakage:

1. Combine raw IPEDS survey files into a single institution-year panel
2. Clean the data and align annual file structures/schemas
3. Construct a forward-looking classification target
4. Engineer features from institutional finance, staffing, and enrollment patterns
5. Chronologically split into training, validation, and test periods
6. Fit all preprocessing exclusively on the training data
7. Train and evaluate classification models

### Target Variable

Binary indicator: does an institution experience a **decline of ≥5% in instructional FTE staff over a two-year window** (year *t* vs. year *t+2*)? A two-year horizon was chosen over a one-year target because it captures gradual declines (e.g., 3% + 3% across two separate years) that a single-year threshold would miss, and it aligns with the dashboard's stated 18–24-month early-warning purpose.

### Feature Engineering

Three groups of engineered predictors:

1. **Fiscal structure:** tuition dependence, state appropriation share, operating margin, instruction share of expenses, revenue per staff FTE
2. **Trend/momentum:** year-over-year changes in revenue, state funding, expenses, and staffing
3. **Enrollment:** YoY and multi-year enrollment trends, retention trends, per-student financial measures, student-to-faculty ratio, graduate enrollment share (raw enrollment counts added little value on their own — trend/ratio framing is what mattered)

### Chronological Partitioning

| Split | Years |
|---|---|
| Train | 2014–2020 |
| Validation | 2021 |
| Test | 2022 |

*(2023–2024 predictor years can't yet be labeled since the target looks two years ahead.)*

## Modeling

Seven classifiers were trained and compared: **Logistic Regression** (baseline), **Penalized Logistic Regression**, **Random Forest**, **XGBoost**, **LightGBM**, **Linear SVM**, and **k-Nearest Neighbors**. Resampling (SMOTE), ensembling, and PCA dimensionality reduction were tested but did not improve performance under temporal validation, so they were excluded from the final pipeline.

**Evaluation metric:** PR-AUC (precision-recall AUC) was the primary metric, since staffing reductions are rare (~18–25% of institution-years) and PR-AUC handles class imbalance far better than ROC-AUC alone. Performance is reported as **lift over the base rate** (1× = no skill).

## Results

### Model Comparison (Validation/Test)

| Model | Valid PR-AUC | Test PR-AUC | Test ROC-AUC | Test PR-Lift |
|---|---|---|---|---|
| LightGBM | 0.442 | 0.313 | 0.694 | 1.75× |
| XGBoost | 0.446 | 0.344 | 0.696 | 1.77× |
| Linear SVM | 0.398 | 0.269 | 0.614 | 1.49× |
| **Random Forest (selected)** | **0.454** | **0.342** | **0.694** | **1.91×** |
| Penalized Logistic Regression | 0.347 | 0.252 | 0.631 | 1.32× |
| Logistic Regression | 0.326 | 0.234 | 0.633 | 1.29× |
| k-Nearest Neighbors | 0.336 | 0.240 | 0.594 | 1.25× |

Random Forest, XGBoost, and LightGBM were statistically indistinguishable on paired folds (p = 0.117). **Random Forest was selected** for its stable impurity-based feature ranking and low sensitivity to hyperparameter choices (<0.006 PR-AUC movement across 100–800 trees) — a useful property for a model that needs periodic retraining.

### Final Production Model

Refit on all data through the 2021 validation year and evaluated on the fully held-out 2022 test year:

- **PR-AUC:** 0.390 (base rate: 0.181)
- **PR-Lift:** 2.15× over random ranking
- **ROC-AUC:** 0.69

In practical terms, using the model's top-ranked institutions identifies at-risk institutions at roughly **twice the rate of chance** — enough to narrow a shortlist for closer institutional review, but not strong enough to justify fully automated staffing decisions.

## Key Findings

- **The working hypothesis was only partially supported.** The Institutional Support / Instructional Spend ratio was *not* among the top predictors. Instead, the three strongest predictors by impurity reduction were all **enrollment-trajectory features** (`ENROLL_YOY`, `ENROLL_UG_YOY`, `ENROLL_3Y_TREND`), followed by retention, staffing momentum, and expense growth.
- This means the model behaves more like a **demographic early-warning system** than a financial-structure audit tool — consistent with Grawe's (2018) "demographic cliff" projections.
- The broader premise — that historical financial/operational data carry predictive signal for staffing outcomes — *is* supported: the model's single highest-risk 2022 prediction (P = 0.81) correctly flagged an institution that went on to cut staff, driven by the same variables dominating the global feature ranking (falling enrollment, retention far below average, contracting staffing).
- Median instructional spending declined 3.3% (nominal) from 2014–2024 while median institutional support spending grew 18.2% over the same period — administrative-intensity trends are real, they just weren't the leading predictive signal in this specification.

## The Shield Dashboard

A Streamlit application that turns model outputs into a decision-support tool for non-technical stakeholders. Model training happens offline; the app loads saved artifacts (final Random Forest model, institution-level risk scores, comparison/tuning results, feature importance, evaluation metrics) for fast, reproducible use.

**🔗 Live app:** https://academic-staff-safeguard.streamlit.app/

**Sections:**
1. **Overview** — model performance and key EDA findings
2. **Model Comparison** — why Random Forest was selected, across 7 models and 3 evaluation splits
3. **Hyperparameter Tuning** — selected settings and tuning results
4. **Random Forest Interpretation** — feature importance + highest-risk institution profile
5. **Risk Scorer** — select an institution, adjust financial/enrollment variables (e.g., simulate a funding cut), and see predicted risk update in real time

<!-- Optional: add a screenshot once you have the file path in your repo
![Shield Dashboard Risk Scorer](docs/images/risk-scorer.png)
-->

The dashboard is intended as a **screening and prioritization tool, not an automated decision-maker** — risk scores are meant to guide further review, not determine staffing actions on their own.

## Repository Structure

<!-- PLACEHOLDER: update this tree to match your actual repo layout -->

```
.
├── data/
│   ├── raw/                  # Raw IPEDS survey files (not tracked — see Data section)
│   └── processed/            # Cleaned institution-year panel
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── dashboard/
│   └── app.py                # Shield Dashboard (Streamlit)
├── models/
│   └── random_forest_final.pkl
├── requirements.txt
└── README.md
```

## Getting Started

<!-- PLACEHOLDER: adjust commands/paths to match your actual entry point and dependency file -->

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Shield Dashboard locally
streamlit run dashboard/app.py
```

## Limitations

- IPEDS financial variables have substantial missingness (especially F1C fields and Carnegie classification codes) and are reported with a 1–2 year lag.
- The model uses only publicly reported financial, enrollment, and staffing data — it cannot capture leadership changes, program closures, or internal governance decisions.
- Scope is limited to **public four-year institutions**; results should not be extended to private institutions or community colleges without re-estimation.
- PR-AUC plateaued in the mid-0.30s across every model family tested (resampling, ensembling, and PCA did not move it), suggesting the ceiling reflects available features rather than model choice.
- Chronological evaluation covers only two out-of-time years (2021, 2022), both following an unusually volatile 2018–2019 staffing-reduction spike — periodic retraining is recommended as the underlying data distribution shifts.
- The two-year prediction horizon means the most recent years can't yet be evaluated; dashboard stress-testing outputs should be read as directional scenarios, not precise forecasts.

## Recommended Next Steps

1. **Retest the administrative-intensity hypothesis** with a lagged specification and an interaction with institutional size — administrative decisions plausibly take longer to reach staffing outcomes than enrollment shocks do.
2. **Expand features to address the PR-AUC ceiling**, prioritizing external data in the same family as the current leading signal (enrollment): regional demographic projections, competing local enrollment pressure, and state appropriation context.

## Acknowledgments

The authors thank **Dr. Ebrahim Tarshizi** for guidance and feedback throughout the development of this project, including input on research framing, modeling approach, and drafts of the report.

This project made use of **Claude (Anthropic)** to help build and debug the data-integration and cleaning pipeline (reconciling annual IPEDS files, aligning schemas, decoding missing-value markers, constructing and validating the forward-looking target), generate starter code for figures, and support structuring/copyediting of the written report. All code, analyses, and written content produced with LLM assistance were reviewed, tested, and edited by the authors, who take full responsibility for the content of this work.

## Selected References

- Grawe, N. D. (2018). *Demographics and the demand for higher education.* Johns Hopkins University Press.
- Kelchen, R., Ritter, D., & Webber, D. (2025). *Predicting college closures and financial distress* (FEDS No. 2025-003). Board of Governors of the Federal Reserve System. https://doi.org/10.17016/FEDS.2025.003
- Leslie, L. L., & Rhoades, G. (1995). Rising administrative costs: Seeking explanations. *The Journal of Higher Education, 66*(2), 187–212.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30.* https://doi.org/10.48550/arXiv.1705.07874
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence, 1*(5), 206–215.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.

*Full reference list available in the accompanying paper.*
