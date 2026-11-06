# Company Bankruptcy Prediction — Classification Portfolio Project

## Objective
Compare six supervised classifiers for predicting the `Bankrupt?` label from company financial indicators. Bankruptcy is rare (about 3% of observations), so overall accuracy alone is misleading.

## Models
Logistic Regression, K-Nearest Neighbors, Decision Tree, Multi-Layer Perceptron, Random Forest, and Gradient Boosting.

## Method
1. Load `dataset/Dataset.csv` in the Colab notebook (the notebook currently expects the user to upload `Dataset.csv`).
2. Split data into stratified 80% training and 20% test partitions using `random_state=42`.
3. Learn the 0.90 correlation feature-removal threshold **only on training features**, then apply the same retained columns to the test partition.
4. Apply SMOTE **only to the training partition** to generate synthetic minority-class examples.
5. Train the classifiers and evaluate on the untouched, imbalanced test partition.

## Evaluation: what matters
- **Recall (bankruptcy class):** proportion of bankrupt companies detected.
- **Precision:** proportion of predicted bankruptcies that are actual bankruptcies.
- **F1:** balances precision and recall.
- **PR-AUC:** useful for comparing rare-event classifiers across thresholds.
- **Confusion matrix:** makes missed bankruptcies and false alarms visible.

### Historical results — not yet revalidated
The previous notebook outputs (before the training-only correlation-selection correction) reported the following for Random Forest:

| Metric | Historical value |
|---|---:|
| Accuracy | 0.96 |
| Precision | 0.41 |
| F1 | 0.47 |
| True positives | 24 |
| False negatives | 20 |
| False positives | 34 |
| True negatives | 1286 |
| Bankruptcy recall | 24/44 = 54.5% |

**These are historical outputs, not validated results of the revised notebook.** The revised notebook's stale outputs have been cleared; it must be rerun with the source dataset before any new performance claim is made.

## Business interpretation
The model is a possible **screening aid**, not an automated bankruptcy decision. False negatives can leave risky companies undetected; false positives create unnecessary investigations. Decision thresholds should reflect the cost of each error, and model probabilities should be calibrated and monitored before any operational use.

## Reproduction and limitations
Open `Company_bankruptcy_prediction.ipynb` in Google Colab and upload the source `Dataset.csv` when prompted. The dataset is not documented here with a verified public source or license. Install compatible versions of pandas, numpy, scikit-learn and imbalanced-learn if missing.

**Outstanding validation:** confirm dataset provenance, execute the revised notebook end-to-end, compare metrics across models (including recall and PR-AUC), inspect threshold trade-offs, and verify environment requirements. No end-to-end rerun has been performed in this GitHub review.

## Portfolio progression
AI-PM-CT Project 02: classification after NYC Taxi regression, before NLP and LLM projects.

## November 2026: training-only cross-validation (preliminary)
Using the supplied 6,819-row dataset, we reserved a stratified 20% test split (1,364 rows, including 44 bankrupt companies). **The following scores are 3-fold cross-validation means on the 80% training partition only**; they are not final test results. SMOTE was applied inside each training fold, never to validation folds. All 95 features were retained for this initial six-model comparison.

| Model | SMOTE | Recall | Precision | F1 | Average precision (PR-AUC summary) |
|---|---|---:|---:|---:|---:|
| Logistic Regression | No | .227 | .520 | .316 | .342 |
| Logistic Regression | Yes | .749 | .184 | .296 | .322 |
| KNN | No | .159 | .690 | .256 | .227 |
| KNN | Yes | .693 | .173 | .276 | .171 |
| Decision Tree | No | .346 | .318 | .330 | .131 |
| Decision Tree | Yes | .431 | .241 | .307 | .123 |
| MLP | No | .130 | .383 | .184 | .268 |
| MLP | Yes | .449 | .302 | .358 | .296 |
| Random Forest | No | .154 | .638 | .246 | .408 |
| Random Forest | Yes | .460 | .327 | .382 | .362 |
| Histogram-based Gradient Boosting | No | .199 | .581 | .296 | .453 |
| Histogram-based Gradient Boosting | Yes | .426 | .401 | .413 | .412 |

**Important:** The new validation uses `HistGradientBoostingClassifier`, which is not the same estimator as the original notebook's `GradientBoostingClassifier`. This is an exploratory alternative, not a direct replication. For scaled models, standardization is learned inside each fold. These are default-threshold metrics, not tuned deployment metrics. The original notebook has not been rerun end-to-end; no final model or threshold has been approved. A reproducible validation script and subsequent held-out evaluation are still needed. Dataset redistribution rights have not been verified, so the uploaded CSV is not committed.


## Reproducible six-model validation (6 November 2026)
Run `python validate_models.py "/path/to/Dataset.csv"`. The source CSV is **not committed**. The script performs a stratified 80/20 split (random_state=42), 3-fold stratified cross-validation on the training portion, scaling where appropriate, and SMOTE only within training folds. It selects the highest mean cross-validation F1 (average precision as tie-breaker), refits on all training records, then evaluates once on the held-out 20%.

**Training-only 3-fold cross-validation, with SMOTE:**

| Classifier | Recall | F1 | Average precision |
|---|---:|---:|---:|
| Logistic Regression | 0.749 | 0.296 | 0.322 |
| KNN | 0.693 | 0.276 | 0.171 |
| Decision Tree | 0.431 | 0.307 | 0.123 |
| MLP | 0.449 | 0.358 | 0.296 |
| Random Forest | 0.460 | 0.382 | 0.362 |
| Histogram Gradient Boosting | 0.426 | **0.413** | **0.412** |

**Selected model:** Histogram Gradient Boosting with SMOTE (highest cross-validated F1 across all tested configurations).

**Final held-out evaluation (1,364 records, including 44 bankrupt):**

| Metric | Result |
|---|---:|
| Accuracy | 0.9611 |
| Precision (bankrupt) | 0.4237 |
| Recall (bankrupt) | 0.5682 |
| F1 (bankrupt) | 0.4854 |
| Average precision | 0.5220 |
| True negatives / false positives | 1286 / 34 |
| False negatives / true positives | 19 / 25 |

**Interpretation:** 25 of 44 bankrupt companies were detected; 19 were missed. This is an experimental screening model, not a production-grade credit or bankruptcy decision system. A different threshold or business cost objective may change the preferred model.

**Caveats:** This script uses all 95 input features rather than the notebook's 74-feature correlation-reduced path. The histogram-based boosting estimator differs from the notebook's original gradient boosting implementation. The existing notebook has not been fully rerun; its historical metrics should not be confused with these separately validated script results. Dataset provenance, dependency pinning, threshold calibration, external validation and monitoring remain outstanding.

## Public dataset and attribution

The course-provided dataset has 6,819 rows, 95 financial features and a `Bankrupt?` target. Its structure matches the [UCI Taiwanese Bankruptcy Prediction dataset](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction), which is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: UCI Machine Learning Repository and the original dataset creators; see the UCI record for the recommended citation.

**Before merging this change:** verify the course CSV is equivalent to the UCI-distributed version and the CSV has been uploaded at the repository root as `Dataset - Company Bankruptcy.csv`. The notebook loads it from the public `main` branch without a manual Colab upload. Dataset equivalence to the UCI source and redistribution attribution should still be independently verified.
