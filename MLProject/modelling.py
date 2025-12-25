import joblib
import time
import dagshub
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

# =========================
# LOAD DATA
# =========================
X_train = joblib.load("german_credit_data_preprocessing/X_train.joblib")
X_test = joblib.load("german_credit_data_preprocessing/X_test.joblib")
y_train = joblib.load("german_credit_data_preprocessing/y_train.joblib")
y_test = joblib.load("german_credit_data_preprocessing/y_test.joblib")

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

input_example = X_train[:5, :]
# =========================
# MLFLOW SETUP
# =========================
USERNAME = "rafeeedah"
REPO_NAME = "german-credit-models"
TOKEN = os.getenv("MLFLOW_TRACKING_PASSWORD") # Mengambil dari secrets.DAGSHUB_TOKEN

# Setup Tracking Remote secara manual
mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")
mlflow.set_experiment("German Credit Scoring and Serving Best Model")

# =========================
# MODEL CONFIGS
# =========================
models = {
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=2000),
        "search_space": {
            "C": Real(1e-3, 10.0, prior="log-uniform"),
            "penalty": Categorical(["l2"]),
            "solver": Categorical(["lbfgs"]),
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42),
        "search_space": {
            "n_estimators": Integer(100, 500),
            "max_depth": Integer(3, 20),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 5),
        },
    },
    "SVC": {
        "estimator": SVC(probability=True),
        "search_space": {
            "C": Real(1e-3, 10.0, prior="log-uniform"),
            "gamma": Real(1e-4, 1.0, prior="log-uniform"),
            "kernel": Categorical(["rbf"]),
        },
    },
}

all_runs = []

# =========================
# TRAIN, TUNE & LOG
# =========================
artifact_dir = "artifacts"
os.makedirs(artifact_dir, exist_ok=True)

for model_name, config in models.items():
    print(f"\n{'='*70}")
    print(f"Training {model_name}...")
    print(f"{'='*70}")

    with mlflow.start_run(run_name=f"{model_name}_BayesSearch") as run:

        bayes_search = BayesSearchCV(
            estimator=config["estimator"],
            search_spaces=config["search_space"],
            n_iter=20,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            random_state=42,
        )

        bayes_search.fit(X_train, y_train)

        best_model = bayes_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # ---- METRICS ----
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("test_score", acc)  # For consistency

        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        # ---- PARAMETERS ----
        for param, value in bayes_search.best_params_.items():
            mlflow.log_param(param, value)

        # ---- LOG MODEL (all models) ----
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=f"{model_name}GermanCreditModel"
        )

        # ---- ARTIFACTS (for all models) ----
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"{model_name} Confusion Matrix")
        cm_file = os.path.join(artifact_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_file)
        mlflow.log_artifact(cm_file)
        plt.close()

        # Classification Report
        report = classification_report(y_test, y_pred)
        report_file = os.path.join(artifact_dir, f"{model_name}_classification_report.txt")
        with open(report_file, "w") as f:
            f.write(f"{model_name} Classification Report\n")
            f.write("="*50 + "\n")
            f.write(report)
        mlflow.log_artifact(report_file)

        # Store run info for comparison
        all_runs.append({
            "model_name": model_name,
            "run_id": run.info.run_id,
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        print(f"✓ {model_name} training completed")
        print(f"  Run ID: {run.info.run_id}")

# =========================
# FIND BEST MODEL
# =========================
print("\n" + "="*70)
print("Comparing all models...")
print("="*70)

# Sort by F1 score (you can change this to accuracy or roc_auc)
all_runs_sorted = sorted(all_runs, key=lambda x: x['f1_score'], reverse=True)

print("\nModel Performance Summary:")
print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'ROC AUC':<12}")
print("-"*70)
for run_info in all_runs_sorted:
    print(f"{run_info['model_name']:<25} {run_info['accuracy']:<12.4f} {run_info['f1_score']:<12.4f} {run_info['roc_auc']:<12.4f}")

best_run = all_runs_sorted[0]

print("\n" + "="*70)
print("BEST MODEL SELECTED")
print("="*70)
print(f"Model: {best_run['model_name']}")
print(f"Run ID: {best_run['run_id']}")
print(f"Accuracy: {best_run['accuracy']:.4f}")
print(f"F1 Score: {best_run['f1_score']:.4f}")
print(f"ROC AUC: {best_run['roc_auc']:.4f}")
print("="*70)

# Save best model info for Docker build
with open("best_model_info.txt", "w") as f:
    f.write(f"model_name={best_run['model_name']}\n")
    f.write(f"run_id={best_run['run_id']}\n")
    f.write(f"accuracy={best_run['accuracy']:.4f}\n")
    f.write(f"f1_score={best_run['f1_score']:.4f}\n")
    f.write(f"roc_auc={best_run['roc_auc']:.4f}\n")
    f.write(f"registered_model_name={best_run['model_name']}GermanCreditModel\n")

print("\n✓ Best model info saved to best_model_info.txt")
print("\nTraining complete! All models logged to MLflow.")