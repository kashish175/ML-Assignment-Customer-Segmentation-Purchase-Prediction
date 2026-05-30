#  Customer Segmentation & Purchase Prediction
### Machine Learning Assignment 

- **Student Name:** Kashish Agrawal
- **Student ID:** BITSoM_BA_25071489
- **Email:** kashisha049@gmail.com

##  Project Overview

This project simulates the role of a **Data Analyst at an e-commerce company**. The business goal is to:

1. **Segment customers** into meaningful behavioral groups using unsupervised learning (K-Means Clustering) to enable personalized marketing campaigns.
2. **Predict high-value customers** using a supervised classification model (Random Forest) to forecast future spending behavior.
3. **Derive actionable business recommendations** grounded in the analytical findings.

> **Dataset:** 5,025 customer transaction records with 13 features covering demographics, purchase history, and engagement metrics.

---

##  Repository Structure

```
├── Kashish_Agrawal_ML_Assignment.ipynb   # Main Jupyter Notebook (all code + outputs)
├── Project_Data.csv                       # Customer transaction dataset
├── ML_Assignment_Problem_Statement.pdf    # Assignment brief
└── README.md                              # This file
```

---


##  Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | Categorical | Unique identifier per customer |
| `age` | Numerical | Customer age in years (18–75) |
| `gender` | Categorical | Male / Female / Other |
| `city_tier` | Categorical | Tier 1 / Tier 2 / Tier 3 |
| `membership_type` | Categorical | Silver / Gold / Platinum |
| `total_spend` | Numerical | Total cumulative spend (₹11 – ₹27,621) |
| `num_transactions` | Numerical | Number of purchases made |
| `avg_transaction_value` | Numerical | Average spend per transaction |
| `days_since_last_purchase` | Numerical | Recency metric |
| `num_visits` | Numerical | Website / app visit count |
| `product_categories_purchased` | Numerical | Distinct categories bought |
| `discount_used` | Numerical | Total discount availed |
| `high_value_customer` | Binary Target | 1 = High Value, 0 = Low Value |

**Target Distribution:** 3,147 Low-Value (62.5%) · 1,878 High-Value (37.5%)

---

##  Tech Stack

- **Language:** Python 3.x
- **Notebook:** Jupyter Notebook
- **Libraries:**
  - `pandas`, `numpy` — Data manipulation
  - `matplotlib`, `seaborn` — Visualization
  - `scikit-learn` — ML modeling (KMeans, RandomForest, GridSearchCV, StandardScaler)

---

##  Project Walkthrough

### Part 1 — Data Exploration & Preprocessing
- Loaded dataset and inspected shape, types, and summary statistics
- Identified and handled missing values:
  - Numerical columns (`age`, `days_since_last_purchase`, `num_visits`) → filled with **median**
  - Categorical columns (`gender`) → filled with **mode**
- Removed duplicate records based on `customer_id`
- Exploratory visualizations:
  - Histogram of `total_spend` — reveals strong right-skewed distribution
  - Boxplot of `total_spend` by `membership_type` — Platinum members spend significantly more

---

### Part 2 — Customer Segmentation (K-Means Clustering)
- Selected 6 numerical features: `age`, `total_spend`, `num_transactions`, `avg_transaction_value`, `days_since_last_purchase`, `num_visits`
- Applied **StandardScaler** to normalize features (prevents high-range features from dominating distance calculations)
- Used the **Elbow Method** (K=2 to 8) to determine optimal number of clusters → **K = 3**

#### Customer Segments Identified

| Segment | Label | Size | Avg Age | Avg Spend | Avg Transactions |
|---------|-------|------|---------|-----------|-----------------|
| 0 | Mid-Tier Regulars | 2,101 | ~39 yrs | ₹2,931 | 15 |
| 1 | High-Value Loyalists | 1,145 | ~50 yrs | ₹8,461 | 25 |
| 2 | Young Budget Explorers | ~1,779 | ~28 yrs | ₹1,002 | 8 |

---

### Part 3 — Predictive Modeling (Random Forest Classifier)

**Feature Engineering:**
- `spend_per_visit` = `total_spend` / `num_visits` → spend efficiency metric
- `discount_percentage` = `discount_used` / `total_spend` → price sensitivity proxy

**Data Preparation:**
- One-Hot Encoding for `gender`, `city_tier`, `membership_type`
- 80/20 stratified train-test split (`random_state=42`)
- StandardScaler fitted **only on training data** (no data leakage)

**Model:** Random Forest Classifier (`n_estimators=100`)

#### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **99.40%** |
| Precision | ~99% |
| Recall | **99.20%** |
| F1-Score | **0.9920** |

> Recall is the most business-critical metric here — missing a high-value customer (false negative) costs more than a false positive.

---

### Part 4 — Optimization & Business Insights

**Hyperparameter Tuning (GridSearchCV, cv=3):**

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
```

Performance remained stable post-tuning (~99.40% accuracy, F1 ~0.9920), indicating the default Random Forest was already near-optimal for this dataset.

**Top 8 Features by Importance:**

1. `total_spend` — Primary driver of customer value
2. `discount_used` — Price sensitivity / promotional responsiveness
3. `avg_transaction_value` — Spend efficiency per purchase
4. `num_transactions` — Purchase frequency
5. `spend_per_visit` — Engineered feature: value per visit
6. `days_since_last_purchase` — Recency / churn risk
7. `num_visits` — Engagement frequency
8. `age` — Demographic signal

---

##  Business Recommendations

### 1.  Target Young Budget Explorers (Cluster 2) — From Clustering
Implement a **Habit-Formation Campaign** using push notifications and time-sensitive flash sales for the youngest, lowest-frequency segment (~28 yrs avg age). Converting this group from low-frequency to moderate-frequency shoppers is the most scalable path to growing CLV.

### 2.  Automate VIP Fast-Tracking — From Model Predictions
Deploy the classifier in the **live CRM pipeline**. When a new customer is predicted as High-Value early in their lifecycle, automatically unlock Gold Tier benefits (free shipping, priority support). The model's ~99% precision makes this cost-effective — nearly zero benefit wasted on non-profitable customers.

### 3.  Maximize Average Order Value (AOV) — From Feature Importance
Shift merchandising strategy toward **bundle pricing and threshold discounts** (e.g., "Spend ₹500 more to save 10%"). Feature importance analysis shows `avg_transaction_value` and `spend_per_visit` outrank visit frequency as predictors of customer value — meaning value-per-transaction matters more than visit count.

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/customer-segmentation-ml.git
cd customer-segmentation-ml
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook Kashish_Agrawal_ML_Assignment.ipynb
```

> **Note:** Run all cells from top to bottom. The notebook is designed to execute sequentially. Make sure `Project_Data.csv` is in the same directory as the notebook.

---

## Key Results Summary

```
Dataset         : 5,025 customers · 13 features
Missing Values  : Handled (median/mode imputation)
Clusters        : 3 (K-Means, K determined via Elbow Method)
Model           : Random Forest Classifier
Accuracy        : 99.40%
F1-Score        : 0.9920
Top Predictor   : total_spend
```

---
