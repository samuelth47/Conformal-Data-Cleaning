# Conformal Data Cleaning

CMPE 257 (Machine Learning), Section 02, San José State University
Samuel Haileselassie, Injung Kim, Michael Cheung

A simplified re-implementation of **Conformal Data Cleaning (CDC)** from
[Jäger & Bießmann, "From Data Imputation to Data Cleaning – Automated Cleaning of Tabular Data Improves Downstream Predictive Performance" (AISTATS 2024)](https://proceedings.mlr.press/v238/jager24a/jager24a.pdf).

## What it does

CDC trains one model per column that predicts that column's value from the other columns.
Split conformal prediction then calibrates how far an observed value may sit from the
prediction. Cells outside the calibrated range are flagged as errors and replaced with the
model's prediction (median for numeric columns, most likely label for categorical columns).

We inject controlled errors into the test set, clean them, and compare downstream
classifier F1 on the original, corrupted, and cleaned data.


## Dataset

OpenML Airlines (id 42493): 26,969 rows, 7 features, binary target (flight delayed or not).
The data has no missing values, so every error we study is an injected one.
Split: 20,575 fit / 1,000 calibration / 5,394 test.

## Method

- **Column models:** random forest / extra trees for categorical columns, gradient boosting
  (quantile regression) for numeric columns. No one-hot encoding before fitting.
- **Conformal step:** conformalized quantile regression for numeric columns and
  class-conditional conformal prediction for categorical columns. Flagging is skipped when
  a prediction set is empty.
- **Confidence levels:** 0.5, 0.8, 0.999 (alpha = 0.5, 0.2, 0.001).
- **Corruptions (test set only):** Gaussian noise, random scaling (x10 / x100 / x1000),
  swapped airport columns, categorical shift. Error fractions: 1%, 5%, 10%, 30%, 50%.
- **Downstream model:** SGD classifier with one-hot encoding and grid search; metric is F1.
- **Baseline:** the same column models without the conformal calibration step.
- **Seeds:** 3 (0, 1, 2). Reported numbers are means over seeds.

## How to run

1. Open `conformal_data_cleaning.ipynb` in Google Colab (or locally with Python 3).
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
3. Run all cells. The notebook downloads the data from OpenML, runs seeds 0 to 2, and
   writes `results_seed{0,1,2}.csv` and `improvement.png` to `RESULTS_DIR`
   (`/content/results` by default; change it for local runs).
