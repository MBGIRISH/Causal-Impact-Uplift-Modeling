# Causal Impact & Uplift Modeling for Marketing Campaign Optimization

> A production-grade pipeline for estimating causal treatment effects and identifying customers most likely to respond to marketing campaigns.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Business Problem](#business-problem)
- [Methodology](#methodology)
- [Technical Implementation](#technical-implementation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Deliverables](#deliverables)
- [Technical Notes](#technical-notes)
- [References](#references)

---

## 🎯 Executive Summary

This project implements a comprehensive causal inference and uplift modeling pipeline to optimize marketing campaign targeting. By combining rigorous statistical methods (Propensity Score Matching, Inverse Propensity Weighting) with machine learning-based uplift modeling, we identify which customers are truly influenced by marketing campaigns versus those who would purchase regardless.

**Key Results:**
- ✅ **Average Treatment Effect**: 6.0 percentage point increase in conversion rate
- ✅ **Uplift Model Performance**: Qini coefficient of 74.05 (substantial improvement over random targeting)
- ✅ **Business Impact**: Identified persuadable segments and quantified cost savings through optimized targeting
- ✅ **Actionable Insights**: Four-quadrant user segmentation with clear targeting recommendations

---

## 💼 Business Problem

### Context

An e-commerce company ran a marketing campaign (email/discount offers) where some users received the campaign (treatment group) and others did not (control group). The critical business question is not "Who will purchase?" but rather:

1. **What is the true incremental impact** of the campaign on purchase behavior?
2. **Which users are positively influenced** by the campaign (should be targeted)?
3. **Which users would purchase anyway** (wasteful to target)?
4. **Which users are negatively affected** (should be excluded)?

### Challenge

Standard machine learning approaches predict purchase probability but fail to distinguish between:
- Users who purchase **because** of the campaign (incremental value)
- Users who purchase **regardless** of the campaign (no incremental value)

This leads to suboptimal targeting, wasted marketing spend, and potentially negative customer experiences.

### Solution

We implement a two-stage approach:
1. **Causal Inference**: Estimate average treatment effects using propensity score methods to account for confounding
2. **Uplift Modeling**: Predict individual treatment effects to enable personalized targeting decisions

---

## 🔬 Methodology

### 1. Causal Inference Framework

#### Propensity Score Matching (PSM)
- Estimates propensity scores (probability of treatment assignment) using logistic regression on pre-treatment covariates
- Creates balanced strata by matching treated and control units with similar propensity scores
- Computes Average Treatment Effect (ATE) within each stratum and aggregates

**Note**: The implementation uses stratification on propensity score bins rather than true 1:1 matching. This is faster and often works just as well for large datasets.

#### Inverse Propensity Weighting (IPW)
- Uses propensity scores to create inverse probability weights
- Re-weights observations to create a pseudo-population where treatment assignment is independent of covariates
- Estimates ATE using stabilized weights with truncation to reduce variance

#### Validation
- **Covariate Balance Diagnostics**: Standardized mean differences before/after adjustment
- **Common Support**: Ensures overlap in propensity score distributions
- **Sensitivity Analysis**: Compares multiple estimation methods for robustness

### 2. Uplift Modeling

#### Two-Model Approach
- Trains separate models:
  - Model 1: `P(Y=1 | T=1, X)` on treated users
  - Model 2: `P(Y=1 | T=0, X)` on control users
- Individual Treatment Effect (ITE): `τ(X) = P(Y=1 | T=1, X) - P(Y=1 | T=0, X)`

#### Evaluation Metrics
- **Uplift Curve**: Cumulative incremental conversions vs. fraction of population targeted
- **Qini Curve**: Uplift curve minus random targeting baseline
- **Qini Coefficient**: Area under Qini curve (higher = better targeting)

### 3. Business Segmentation

Users are classified into four quadrants based on baseline probability and predicted uplift:

| Segment | Baseline | Uplift | Action |
|---------|----------|--------|--------|
| **Persuadables** | Low | High | **TARGET** - Primary audience for campaigns |
| **Sure-things** | High | Low | **AVOID** - Would purchase anyway, wasteful spend |
| **Lost causes** | Low | Low | **DEFER** - Unlikely to respond, low priority |
| **Do-not-disturb** | Any | Negative | **EXCLUDE** - Campaign hurts them, monitor for churn |

---

## 🛠 Technical Implementation

### Architecture

The project follows a modular, production-ready architecture:

```
Causal Impact & Uplift Modeling/
├── src/                          # Core modules
│   ├── preprocessing.py          # Data loading, splitting, balance diagnostics
│   ├── causal_models.py          # Propensity scores, PSM, IPW
│   ├── uplift_models.py          # Two-model uplift, segmentation
│   └── evaluation.py             # Uplift/Qini curves, metrics
├── notebooks/                    # Analysis pipeline
│   ├── 01_eda_treatment_vs_control.ipynb
│   ├── 02_causal_inference.ipynb
│   ├── 03_uplift_modeling.ipynb
│   └── 04_business_insights.ipynb
├── data/                         # Dataset (64,000 users)
│   └── marketing_campaign.csv
├── results/                      # Outputs (figures, predictions)
└── requirements.txt
```

### Key Design Decisions

1. **Modular Code**: Reusable functions enable easy integration into production pipelines
2. **Robust Imports**: Handles both package-style and notebook-style imports
3. **Automatic Feature Detection**: Infers numeric vs. categorical features from data types
4. **Stratified Splitting**: Preserves treatment-outcome distribution in train/test sets
5. **Comprehensive Diagnostics**: Balance checks, common support validation, sensitivity analysis

### Dependencies

- **Core**: pandas, numpy, scikit-learn, statsmodels
- **Visualization**: matplotlib, seaborn
- **Causal ML**: causalml (optional, for advanced uplift models)
- **Jupyter**: jupyter, ipykernel, ipywidgets

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone repository
git clone <repository-url>
cd "Causal Impact & Uplift Modeling"

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install CausalML for advanced uplift models
pip install causalml
```

### Running the Analysis

**Option 1: Jupyter Notebooks (Recommended for Exploration)**

```bash
# Start Jupyter
jupyter notebook

# Run notebooks in sequential order:
# 1. 01_eda_treatment_vs_control.ipynb      - Exploratory analysis
# 2. 02_causal_inference.ipynb              - Causal effect estimation
# 3. 03_uplift_modeling.ipynb              - Individual treatment effects
# 4. 04_business_insights.ipynb             - Segmentation & ROI analysis
```

**Option 2: Programmatic Usage**

```python
from src.preprocessing import ColumnConfig, load_marketing_data, stratified_train_test_split
from src.causal_models import fit_propensity_score_model, ps_stratified_ate, ipw_ate
from src.uplift_models import train_two_model_uplift, predict_two_model_uplift
from src.evaluation import uplift_and_qini

# Load data
column_config = ColumnConfig(
    user_id_col="user_id",
    treatment_col="treatment",
    outcome_col="purchase",
)
df, column_config = load_marketing_data("data/marketing_campaign.csv", column_config)

# Train/test split
df_train, df_test = stratified_train_test_split(df, column_config, test_size=0.3)

# Causal inference
ps_result = fit_propensity_score_model(df_train, df_test, column_config)
psm_ate, _, _, _ = ps_stratified_ate(ps_result.train, column_config, ps_col="ps")
ate_ipw, _ = ipw_ate(ps_result.train, column_config, ps_col="ps")

# Uplift modeling
from sklearn.ensemble import GradientBoostingClassifier
two_model_result, feature_cols = train_two_model_uplift(
    df_train, column_config,
    base_model_cls=GradientBoostingClassifier,
    model_kwargs={"n_estimators": 100, "max_depth": 5, "random_state": 42}
)
df_test_pred = predict_two_model_uplift(df_test, two_model_result, feature_cols)

# Evaluation
results = uplift_and_qini(
    df_test_pred[column_config.outcome_col].values,
    df_test_pred[column_config.treatment_col].values,
    df_test_pred["uplift"].values
)
print(f"Qini Coefficient: {results['qini_coef']:.4f}")
```

### Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'preprocessing'"
- **Solution**: Make sure you've activated the virtual environment and the notebook is using the correct kernel

**Issue**: "FileNotFoundError: data/marketing_campaign.csv"
- **Solution**: Verify the file exists: `ls -la data/marketing_campaign.csv`

**Issue**: Kernel not found in Jupyter
- **Solution**: Install ipykernel: `pip install ipykernel`
- Then register: `python -m ipykernel install --user --name=causal-uplift --display-name "Python (Causal Impact)"`

---

## 📊 Dataset

### Overview

- **Source**: Kaggle Marketing Campaign Dataset
- **Size**: 64,000 users
- **Treatment Rate**: 66.7% (42,694 treated, 21,306 control)
- **Overall Conversion**: 14.7%

### Variables

**Core:**
- `user_id`: Unique identifier
- `treatment`: Binary (1 = received offer, 0 = no offer)
- `purchase`: Binary outcome (1 = converted, 0 = did not convert)

**Pre-treatment Features:**
- **Numeric**: `recency_days`, `total_spent_last_6m`, `used_discount_before`, `used_bogo_before`, `is_referral_user`
- **Categorical**: `zip_code` (Urban/Suburban/Rural), `channel` (Web/Phone/Multichannel)

### Data Quality

- No missing values
- Balanced treatment assignment across key segments
- Sufficient sample size for robust causal inference

---

## 📈 Key Findings

### Causal Effect Estimates

| Method | ATE | Interpretation |
|--------|-----|----------------|
| Naive (biased) | 6.1 pp | Unadjusted difference (confounded) |
| PSM-stratified | **6.0 pp** | Adjusted for observed confounders |
| IPW (stabilized) | **5.9 pp** | Weighted estimate, robust to model specification |

**Conclusion**: The campaign increases conversion rate by approximately **6 percentage points** after adjusting for confounding factors.

### Uplift Model Performance

- **Qini Coefficient**: 74.05 (substantial improvement over random targeting)
- **Uplift Distribution**: Heterogeneous treatment effects across user segments
- **Targeting Efficiency**: Top 30% of users by predicted uplift capture majority of incremental conversions

### Business Impact

**Recommended Strategy**: Target top 30% of users by predicted uplift

- **Incremental Revenue**: Significant increase vs. random targeting
- **Cost Savings**: Reduced marketing spend by excluding non-responsive segments
- **ROI Improvement**: Optimized targeting strategy maximizes return on marketing investment

### User Segmentation

- **Persuadables**: High uplift, low baseline → Primary targeting audience
- **Sure-things**: Low uplift, high baseline → Exclude to reduce wasteful spend
- **Lost causes**: Low uplift, low baseline → Low priority, defer targeting
- **Do-not-disturb**: Negative uplift → Exclude and monitor for churn risk

---

## 📦 Deliverables

### Analysis Notebooks

1. **EDA & Validation** (`01_eda_treatment_vs_control.ipynb`)
   - Treatment-control balance analysis
   - Confounder identification
   - Baseline (naive) effect estimates

2. **Causal Inference** (`02_causal_inference.ipynb`)
   - Propensity score modeling
   - PSM and IPW ATE estimation
   - Balance diagnostics and sensitivity checks

3. **Uplift Modeling** (`03_uplift_modeling.ipynb`)
   - Individual treatment effect predictions
   - Qini and uplift curve evaluation
   - Conditional average treatment effects by segment

4. **Business Insights** (`04_business_insights.ipynb`)
   - Four-quadrant user segmentation
   - Targeting strategy comparison
   - ROI analysis and executive recommendations

### Outputs

- **Causal Estimates**: ATE from multiple methods with confidence intervals
- **Uplift Predictions**: Individual treatment effects for each user
- **Segmentation**: Four-quadrant classification with targeting recommendations
- **Business Metrics**: ROI, cost savings, optimal targeting thresholds
- **Visualizations**: Balance plots, uplift curves, Qini curves, segmentation scatter plots

### Code Quality

- **Modular Design**: Reusable, well-documented functions
- **Type Hints**: Full type annotations for maintainability
- **Error Handling**: Robust imports and data validation
- **Documentation**: Comprehensive docstrings and inline comments

---

## 🔧 Technical Notes

### Assumptions

- **No Unmeasured Confounding**: All confounders are observed and included in the model
- **Stable Unit Treatment Value Assumption (SUTVA)**: No interference between units
- **Positivity**: All units have non-zero probability of receiving treatment
- **Ignorability**: Treatment assignment is independent of potential outcomes conditional on covariates

### Limitations

- Propensity score models assume correct specification (logistic regression)
- Uplift models require sufficient sample size in both treated and control groups
- Results are conditional on observed covariates; unmeasured confounding may bias estimates

### Future Enhancements

- **Advanced Uplift Models**: Causal forests, meta-learners (T-learner, X-learner)
- **Time-varying Treatments**: Extend to longitudinal campaign analysis
- **Heterogeneous Effects**: Deep dive into treatment effect modifiers
- **Production Deployment**: API endpoints for real-time targeting decisions

### Implementation Notes

- The dataset used here is from a real marketing campaign, but some preprocessing was done to anonymize and structure the data appropriately
- CausalML is optional - the two-model approach works well on its own
- For production use, consider adding more robust error handling and logging

---

## 📚 References

### Academic Papers

1. Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.

2. Radcliffe, N. J., & Surry, P. D. (2011). Real-world uplift modelling with significance-based uplift trees. *Portland: Stochastic Solutions*.

3. Gutierrez, P., & Gérardy, J. Y. (2017). Causal inference and uplift modelling: A review of the literature. *International Conference on Predictive Applications and APIs*, 1-13.

### Software Libraries

- **scikit-learn**: Machine learning models and preprocessing
- **statsmodels**: Statistical modeling and inference
- **causalml**: Advanced causal inference and uplift modeling (optional)
- **pandas/numpy**: Data manipulation and numerical computing

---

## 📄 License

This project is provided for educational and business use. Please ensure compliance with data usage agreements when working with proprietary datasets.

---

## 👤 Author

Data Science Team

**Project Status**: ✅ Complete and tested  
**Last Updated**: January 2025

---

## 🙏 Acknowledgments

This project demonstrates best practices in causal inference and uplift modeling for marketing analytics. The methodology combines rigorous statistical methods with practical business applications.
