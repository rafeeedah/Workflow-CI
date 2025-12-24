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
X_train = joblib.load("german_credit_data_preprocessing/X_train.joblib")
X_test = joblib.load("german_credit_data_preprocessing/X_test.joblib")
y_train = joblib.load("german_credit_data_preprocessing/y_train.joblib")
y_test = joblib.load("german_credit_data_preprocessing/y_test.joblib")

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# =========================
# MLFLOW SETUP
# =========================
mlflow.set_experiment("German Credit Scoring - Docker")
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


# =========================
# FIND AND SAVE BEST MODEL INFO
# =========================
print("\n" + "="*50)
print("Finding best model...")
print("="*50)

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("German Credit Scoring - Docker")
runs = client.search_runs(
    experiment.experiment_id, 
    order_by=["metrics.training_accuracy_score DESC"],
    max_results=1
)

best_run = runs[0]
best_model_name = best_run.data.tags.get('mlflow.runName', 'unknown')
best_run_id = best_run.info.run_id
best_accuracy = best_run.data.metrics.get('training_accuracy_score', 0.0)

print(f"\nBest Model: {best_model_name}")
print(f"Run ID: {best_run_id}")
print(f"Accuracy: {best_accuracy:.4f}")
print("="*50)

# Save to file for GitHub Actions
with open("best_model_info.txt", "w") as f:
    f.write(f"model_name={best_model_name}\n")
    f.write(f"run_id={best_run_id}\n")
    f.write(f"accuracy={best_accuracy}\n")

print("\n✓ Best model info saved to best_model_info.txt")