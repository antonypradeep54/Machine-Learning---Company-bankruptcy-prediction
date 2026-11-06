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
