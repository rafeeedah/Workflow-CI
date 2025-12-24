# MLProject/promote_best_model.py

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "German Credit Scoring - Tuning All Models"
MODEL_NAME = "GermanCreditModel" 
METRIC_TO_MONITOR = "f1_score"

def promote_best_model():
    client = MlflowClient()

    # Setup Tracking URI agar Client tahu kemana mencari
    # (Di CI, ini sudah diset via $GITHUB_ENV, tapi baiknya diset lagi di Python)
    # mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    # 1. Cari Experiment ID
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
        return

    # 2. Cari Run Terbaik (Metrik Tertinggi)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        # Filter status dan urutkan
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRIC_TO_MONITOR} DESC"],
        max_results=1, 
    )

    if not runs:
        print("Tidak ditemukan run yang selesai.")
        return

    best_run = runs[0]
    best_f1 = best_run.data.metrics.get(METRIC_TO_MONITOR)
    
    # ARTIFACT PATH: SESUAIKAN DENGAN YANG ANDA LOG DI MODELLING.PY
    # Karena Anda menggunakan artifact_path="model", kita gunakan path itu.
    ARTIFACT_PATH = "model" 
    model_uri = f"runs:/{best_run.info.run_id}/{ARTIFACT_PATH}"

    print(f"Model Terbaik ditemukan: Run ID {best_run.info.run_id} dengan {METRIC_TO_MONITOR}={best_f1:.4f}")
    print(f"Mencoba mendaftarkan model dari URI: {model_uri}")

    # 3. Register Model Terbaik ke Registry
    try:
        # Cek apakah artefak 'model' benar-benar ada di run ini (Opsional tapi membantu)
        # artifact_list = client.list_artifacts(best_run.info.run_id)
        # print(f"Daftar Artefak: {[a.path for a in artifact_list]}")

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
        # Jika gagal, coba cetak URI yang gagal untuk debugging
        print(f"Gagal menggunakan URI: {model_uri}")

if __name__ == "__main__":
    promote_best_model()