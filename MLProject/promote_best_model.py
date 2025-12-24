# MLProject/promote_best_model.py

import mlflow
from mlflow.tracking import MlflowClient

# Konfigurasi
EXPERIMENT_NAME = "German Credit Scoring - Tuning All Models"
MODEL_NAME = "GermanCreditModel" # Nama yang akan digunakan di MLflow Registry
METRIC_TO_MONITOR = "f1_score"

def promote_best_model():
    # Setup DagsHub Tracking
    # mlfow tracking uri harus sudah diset di GitHub Actions
    client = MlflowClient()

    # 1. Cari Experiment ID
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
        return

    # 2. Cari Run Terbaik (Metrik Tertinggi)
    # Filter run yang sukses, urutkan berdasarkan F1 Score (DESC)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRIC_TO_MONITOR} DESC"],
        max_results=1, # Ambil 1 run terbaik
    )

    if not runs:
        print("Tidak ditemukan run yang selesai.")
        return

    best_run = runs[0]
    best_f1 = best_run.data.metrics.get(METRIC_TO_MONITOR)
    model_uri = f"runs:/{best_run.info.run_id}/model"

    print(f"Model Terbaik ditemukan: Run ID {best_run.info.run_id} dengan {METRIC_TO_MONITOR}={best_f1:.4f}")

    # 3. Register Model Terbaik ke Registry
    try:
        # Register versi baru dari model terbaik
        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        version = result.version
        print(f"Model versi {version} berhasil didaftarkan.")

        # 4. Promosikan Model Baru ke Status 'Production'
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production"
        )
        print(f"Model {MODEL_NAME} versi {version} dipromosikan ke 'Production'.")

    except Exception as e:
        print(f"Gagal mendaftarkan atau mempromosikan model: {e}")

if __name__ == "__main__":
    promote_best_model()