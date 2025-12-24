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

# =========================
# MLFLOW SETUP
# =========================
USERNAME = "rafeeedah"
REPO_NAME = "german-credit-prediction"
TOKEN = os.getenv("MLFLOW_TRACKING_PASSWORD") # Mengambil dari secrets.DAGSHUB_TOKEN

# Setup Tracking Remote secara manual
mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")

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

# =========================
# TRAIN, TUNE & LOG
# =========================
for model_name, config in models.items():

    with mlflow.start_run(run_name=f"{model_name}_BayesSearch"):

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

        # ---- PARAMETERS ----
        for param, value in bayes_search.best_params_.items():
            mlflow.log_param(param, value)

        # ---- MODEL ----
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # =========================
        # ARTIFACT 1: Confusion Matrix
        # =========================
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"{model_name} Confusion Matrix")

        cm_file = f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_file)
        plt.close()

        mlflow.log_artifact(cm_file)
        time.sleep(3)
        os.remove(cm_file)

        # =========================
        # ARTIFACT 2: Classification Report
        # =========================
        report = classification_report(y_test, y_pred)
        report_file = f"{model_name}_classification_report.txt"

        with open(report_file, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_file)
        time.sleep(3)
        os.remove(report_file)

        print(f"{model_name} tuning completed")


