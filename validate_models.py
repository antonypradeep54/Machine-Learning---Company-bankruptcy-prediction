"""Reproducible training-only model selection and one final holdout evaluation.
Usage: python validate_models.py /path/to/Dataset.csv
Dependencies: pandas numpy scikit-learn imbalanced-learn
"""
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def main(path):
    df = pd.read_csv(path)
    X, y = df.drop(columns=["Bankrupt?"]), df["Bankrupt?"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=500, random_state=42), True),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
        "MLP": (MLPClassifier(hidden_layer_sizes=(50,), max_iter=120, early_stopping=True, random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42), False),
        "Histogram Gradient Boosting": (HistGradientBoostingClassifier(max_iter=100, random_state=42), False),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []
    for name, (model, needs_scaling) in models.items():
        for smote in (False, True):
            steps = []
            if needs_scaling:
                steps.append(("scale", StandardScaler()))
            if smote:
                steps.append(("smote", SMOTE(random_state=42)))
            steps.append(("model", model))
            pipeline = Pipeline(steps)
            scores = cross_validate(
                pipeline, X_train, y_train, cv=cv, n_jobs=1,
                scoring={"recall": "recall", "precision": "precision", "f1": "f1", "ap": "average_precision"},
                error_score="raise",
            )
            row = {"model": name, "smote": smote}
            row.update({metric: float(np.mean(scores["test_" + metric]))
                        for metric in ("recall", "precision", "f1", "ap")})
            results.append(row)
    report = pd.DataFrame(results).sort_values(["f1", "ap"], ascending=False)
    print("TRAINING-ONLY CROSS-VALIDATION")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # F1 is the predeclared selection metric; average precision breaks ties.
    winner = report.iloc[0]
    name, smote = winner["model"], bool(winner["smote"])
    model, needs_scaling = models[name]
    steps = []
    if needs_scaling:
        steps.append(("scale", StandardScaler()))
    if smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", model))
    final_model = Pipeline(steps)
    final_model.fit(X_train, y_train)
    prediction = final_model.predict(X_test)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    print("\nSELECTED:", name, "SMOTE:", smote)
    print("FINAL HOLDOUT METRICS")
    for key, val in {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction),
        "f1": f1_score(y_test, prediction),
        "average_precision": average_precision_score(y_test, probabilities),
    }.items():
        print(f"{key}: {val:.4f}")
    print("confusion_matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, prediction))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python validate_models.py /path/to/Dataset.csv")
    main(sys.argv[1])
