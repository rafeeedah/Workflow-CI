# MLProject/promote_best_model.py (VERSI TANPA STAGE)

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "German Credit Scoring - Tuning All Models"
MODEL_NAME = "GermanCreditModel" 
METRIC_TO_MONITOR = "f1_score"

# Alias yang akan digunakan untuk menggantikan 'Production' Stage
# Contoh: kita akan beri alias 'Champion' pada model terbaik
ALIAS_NAME = "Champion" 

def register_best_model_without_stage():
    client = MlflowClient()

    # 1. Cari Experiment ID
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
        return

    # 2. Cari Run Terbaik (Metrik Tertinggi)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRIC_TO_MONITOR} DESC"],
        max_results=1, 
    )

    if not runs:
        print("Tidak ditemukan run yang selesai.")
        return

    best_run = runs[0]
    best_f1 = best_run.data.metrics.get(METRIC_TO_MONITOR)
    
    ARTIFACT_PATH = "model" 
    model_uri = f"runs:/{best_run.info.run_id}/{ARTIFACT_PATH}"

    print(f"Model Terbaik ditemukan: Run ID {best_run.info.run_id} dengan {METRIC_TO_MONITOR}={best_f1:.4f}")
    print(f"Mencoba mendaftarkan model dari URI: {model_uri}")

    # 3. Register Model Terbaik ke Registry (Hanya Mendaftarkan)
    try:
        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        version = result.version
        print(f"Model versi {version} berhasil didaftarkan.")

        # --- LANGKAH PENGGANTI STAGE (OPSIONAL, HANYA ALIASING) ---
        # Memberikan Alias 'Champion' pada versi model terbaik
        # client.set_registered_model_alias(MODEL_NAME, ALIAS_NAME, version)
        # print(f"Model {MODEL_NAME} versi {version} diberi alias '{ALIAS_NAME}'.")
        # ------------------------------------------------------------

        # HAPUS langkah promosi ke stage Production

    except Exception as e:
        print(f"Gagal mendaftarkan model: {e}")
        print(f"Gagal menggunakan URI: {model_uri}")

if __name__ == "__main__":
    register_best_model_without_stage()