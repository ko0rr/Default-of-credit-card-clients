
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

for folder in [RESULTS, FIGURES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)


# 1. Load data

df = pd.read_csv(ROOT / "data" / "clients_prepared.csv")

# Undocumented education codes -> Other/Unknown
df["EDUCATION"] = df["EDUCATION"].replace({
    0: 4,
    5: 4,
    6: 4,
})

# Undocumented marriage code -> Other/Unknown
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})


# 2. Feature engineering

pay_cols = ["PAY_0", "PAY_2", "PAY_3",
            "PAY_4", "PAY_5", "PAY_6"]

bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
payment_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

df["late_months"] = (df[pay_cols] > 0).sum(axis=1)
df["max_delay"] = df[pay_cols].max(axis=1)
df["avg_bill"] = df[bill_cols].mean(axis=1)
df["avg_payment"] = df[payment_cols].mean(axis=1)
df["bill_to_limit"] = df["BILL_AMT1"] / df["LIMIT_BAL"]


# 3. Features and target

categorical = ["SEX", "EDUCATION", "MARRIAGE"]

numerical = [
    "LIMIT_BAL",
    "AGE",
    *pay_cols,
    *bill_cols,
    *payment_cols,
    "late_months",
    "max_delay",
    "avg_bill",
    "avg_payment",
    "bill_to_limit",
]

X = df[categorical + numerical]
y = df["default"]
ids = df["ID"]


# 4. Train / validation / test split

X_train_full, X_test, y_train_full, y_test, id_train_full, id_test = (
    train_test_split(
        X, y, ids,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    stratify=y_train_full,
    random_state=42,
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# 5. Build pipelines

def make_model(drop_first=False):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first" if drop_first else None,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical),
        ("cat", categorical_pipeline, categorical),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
        )),
    ])


original_model = make_model(drop_first=False)
improved_model = make_model(drop_first=True)

dummy_model = DummyClassifier(strategy="prior")


# 6. Evaluation function

def evaluate(model, X_eval, y_eval, name):
    probabilities = model.predict_proba(X_eval)[:, 1]

    return {
        "model": name,
        "roc_auc": roc_auc_score(y_eval, probabilities),
        "average_precision": average_precision_score(
            y_eval, probabilities
        ),
        "brier": brier_score_loss(y_eval, probabilities),
        "log_loss": log_loss(y_eval, probabilities),
    }


# 7. Compare models on validation data

models = {
    "Dummy": dummy_model,
    "Original": original_model,
    "Improved": improved_model,
}

validation_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    validation_results.append(
        evaluate(model, X_val, y_val, name)
    )

comparison = pd.DataFrame(validation_results)

print("\nValidation comparison:")
print(comparison.to_string(index=False))

comparison.to_csv(
    RESULTS / "validation_comparison.csv",
    index=False,
)


# 8. Refit improved model on the full training set

final_model = make_model(drop_first=True)
final_model.fit(X_train_full, y_train_full)

final_dummy = DummyClassifier(strategy="prior")
final_dummy.fit(X_train_full, y_train_full)

test_results = pd.DataFrame([
    evaluate(final_dummy, X_test, y_test, "Dummy"),
    evaluate(final_model, X_test, y_test, "Improved"),
])

print("\nFinal test metrics:")
print(test_results.to_string(index=False))

test_results.to_csv(
    RESULTS / "final_test_metrics.csv",
    index=False,
)

y_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0,
))


# 9. Save predictions

predictions = pd.DataFrame({
    "ID": id_test.to_numpy(),
    "actual_default": y_test.to_numpy(),
    "predicted_pd": y_proba,
    "LIMIT_BAL": X_test["LIMIT_BAL"].to_numpy(),
})

predictions["risk_decile"] = (
    pd.qcut(
        predictions["predicted_pd"],
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1
)

predictions.to_csv(
    RESULTS / "test_predictions.csv",
    index=False,
)

segments = (
    predictions.groupby("risk_decile")
    .agg(
        clients=("ID", "size"),
        predicted_pd=("predicted_pd", "mean"),
        observed_default=("actual_default", "mean"),
        avg_limit=("LIMIT_BAL", "mean"),
    )
    .reset_index()
)

segments.to_csv(
    RESULTS / "risk_segments.csv",
    index=False,
)

print("\nRisk segments:")
print(segments.to_string(index=False))


# 10. ROC curve

fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr, tpr,
    label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}",
)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "05_roc_curve.png", dpi=160)
plt.close()


# 11. Calibration curve

observed, predicted = calibration_curve(
    y_test,
    y_proba,
    n_bins=10,
    strategy="quantile",
)

plt.figure(figsize=(7, 5))
plt.plot(predicted, observed, "o-", label="Model")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Default Rate")
plt.title("Probability Calibration")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "07_calibration.png", dpi=160)
plt.close()


# 12. Risk deciles

plt.figure(figsize=(8, 5))
plt.plot(
    segments["risk_decile"],
    segments["predicted_pd"] * 100,
    "o-",
    label="Predicted",
)
plt.plot(
    segments["risk_decile"],
    segments["observed_default"] * 100,
    "o-",
    label="Observed",
)

plt.xlabel("Risk Decile")
plt.ylabel("Default Rate (%)")
plt.title("Risk Segmentation")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "08_risk_segments.png", dpi=160)
plt.close()


# 13. Feature coefficients

feature_names = (
    final_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    final_model.named_steps["classifier"].coef_[0]
)

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

coef_df = coef_df.sort_values(
    "abs_coefficient",
    ascending=False,
)

coef_df.to_csv(
    RESULTS / "model_coefficients.csv",
    index=False,
)

print("\nTop coefficients:")
print(coef_df.head(15).to_string(index=False))


# 14. Save model

joblib.dump(
    final_model,
    MODELS / "credit_risk_model.joblib",
)