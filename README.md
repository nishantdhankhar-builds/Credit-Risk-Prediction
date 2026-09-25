# Credit Risk Prediction

Predicting whether a borrower will experience serious financial distress (90+ days delinquent) in the next two years, using the "Give Me Some Credit" dataset from Kaggle.
Credit scoring is a real problem banks and lenders deal with, and this dataset is a well-known benchmark for it — so it felt like a good way to practice classification while working on something with an obvious business angle.

# Dataset

150,000 records, 10 features (income, debt ratio, credit utilization, past delinquencies, etc.)
Target: SeriousDlqin2yrs — whether the person defaulted/was seriously delinquent within 2 years
Class imbalance: ~93.3% non-defaulters vs ~6.7% defaulters

# Procedure

1. Cleaning — MonthlyIncome had ~20% missing values, filled with the median. NumberOfDependents had a small number missing, filled with 0. Dropped a handful of rows with clearly bad age values (e.g. age = 0).
2. Handling imbalance — since defaulters are a small minority, I used class_weight='balanced' in both models instead of leaving them to default to predicting "no default" for almost everyone.
3. Models — trained a Logistic Regression as a baseline (simple, interpretable, easy to explain to a non-technical audience) and a Random Forest as a comparison.
4. Evaluation — used AUC-ROC instead of plain accuracy, since accuracy is misleading on an imbalanced dataset like this one.

# Results
AUC Values:

Logistic Regression - 0.800
Random Forest - 0.838

# What actually predicts default?
(Feature importance from the Random Forest)

1. Revolving credit utilization (~27%) — by far the strongest signal
2. Debt ratio (~14%)
3. Age (~12%)
4. Monthly income (~12%)
5. Number of times 90 days late previously (~9%)

The interesting takeaway to me was that how much of their available credit someone is already using matters more than their income or even their past late payments. It makes intuitive sense — someone maxing out their credit lines is already showing financial strain, even before they've missed a payment. A lender leaning more on utilization data (which updates in near real-time) rather than income alone could probably catch risk earlier.

# Tools

Python, pandas, scikit-learn (Logistic Regression, Random Forest), Google Colab.

# Limitations / what I'd do next if I had more time

1. No hyperparameter tuning — used default settings for both models, so there's likely more performance on the table.
2. No cross-validation, just a single train/test split.
3. Could try feature engineering (e.g., combining the three "days late" columns into one) or a gradient boosting model (XGBoost/LightGBM) to push the AUC higher.
4. Would be worth checking for outliers more rigorously (a few known extreme values in this dataset, like age or debt ratio, can skew results if not handled carefully).
