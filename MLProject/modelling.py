import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# =========================
# LOAD DATA
# =========================
X_train = joblib.load("MLProject/german_credit_data_preprocessing/X_train.joblib")
X_test = joblib.load("MLProject/german_credit_data_preprocessing/X_test.joblib")
y_train = joblib.load("MLProject/german_credit_data_preprocessing/y_train.joblib")
y_test = joblib.load("MLProject/german_credit_data_preprocessing/y_test.joblib")

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# =========================
# MLFLOW SETUP
# =========================
mlflow.set_experiment("German Credit Scoring - Autolog")
mlflow.sklearn.autolog()

# =========================
# MODELS (NO TUNING)
# =========================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVC": SVC()
}

# =========================
# TRAIN MODELS
# =========================
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        model.score(X_test, y_test)