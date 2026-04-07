# Auditing Automated Loan Decisions with SHAP and DiCE

**Course:** SOW-BKI266-2025-PER3-V Explainable AI — Radboud University
**Student:** Myursel Shahin (s1142112)
**Date:** March 2026

---

## Project Overview

This project applies two complementary XAI methods to audit automated loan decisions:

- **SHAP** (SHapley Additive exPlanations) — explains *why* a case was rejected by attributing importance to each feature
- **DiCE** (Diverse Counterfactual Explanations) — explains *what needs to change* to flip a rejection into an approval

The dataset used is the **Adult Income (Census Income)** dataset, where income >50K is treated as a proxy for loan approval and income ≤50K as rejection.

**Research question:** To what extent do SHAP attributions and DiCE counterfactual explanations agree on the main drivers of negative (rejected) loan decisions, and do they indicate reliance on sensitive attributes (sex/age) or their proxies?

---

## Repository Structure

```
xai-loan-audit/
│
├── final_project_notebook.ipynb   # Main code demonstration
├── README.md                      # This file
```

---

## Requirements

Install all required packages with:

```bash
pip install scikit-learn shap dice-ml pandas numpy matplotlib scipy
```

**Python version:** 3.9 or higher recommended.

| Package | Version tested |
|---------|---------------|
| scikit-learn | ≥ 1.2 |
| shap | ≥ 0.43 |
| dice-ml | ≥ 0.9 |
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |
| matplotlib | ≥ 3.6 |
| scipy | ≥ 1.10 |

---

## How to Run

### Option 1: Using the real Adult Income dataset (recommended)

Download `adult.data` from the [UCI repository](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data) and place it in the same folder as the notebook. Then replace the data generation block in **Section 2** of the notebook with:

```python
import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]
data = pd.read_csv("adult.data", names=columns, sep=r",\s*", engine="python", na_values="?")
data.dropna(inplace=True)
data["income"] = (data["income"] == ">50K").astype(int)
data.drop(["fnlwgt", "education", "native_country"], axis=1, inplace=True)
```

### Option 2: Run with synthetic data (default)

The notebook includes a built-in data generator that replicates the Adult Income dataset's structure and statistical properties. Simply open and run the notebook top to bottom — no additional files needed.

```bash
jupyter notebook final_project_notebook.ipynb
```

### Step-by-step execution

1. Run **Section 1** (Setup & Imports) — verify all libraries load
2. Run **Section 2** (Data Loading) — dataset is generated/loaded
3. Run **Section 3** (Model Training) — GradientBoostingClassifier is trained
4. Run **Section 4** (SHAP) — SHAP summary plot and feature importance computed
5. Run **Section 5** (DiCE) — counterfactuals generated for 30 rejected cases
6. Run **Section 6** (Comparison) — Spearman correlation and side-by-side plot
7. Run **Section 7** (Fairness Audit) — group-level SHAP analysis by sex and age

Expected total runtime: **~2–3 minutes** on a standard laptop.

---

## Key Results

| Method | Top feature for rejections | Sensitive attribute (sex) rank |
|--------|---------------------------|-------------------------------|
| SHAP | age (mean \|SHAP\| = 0.71) | Rank 4 |
| DiCE | capital_gain (freq = 50%) | Rank 11 |

- **Top-5 feature overlap:** 3/5 (age, education_num, hours_per_week)
- **Spearman rank correlation:** 0.42 — moderate agreement
- **Fairness finding:** Sex appears in the top-5 SHAP features for female rejected applicants but not for male, indicating asymmetric reliance on this sensitive attribute

---

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *FAT '20*, 607–617.
- Guidotti, R. (2024). Counterfactual explanations and how to find them. *Data Mining and Knowledge Discovery*, 38, 2770–2824.
- Ponce-Bobadilla et al. (2024). Practical guide to SHAP analysis. *Clinical and Translational Science*, 17, e70056.
- Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. *California Law Review*, 104(3), 671–732.
