# IMPORT LIBRARY UTAMA UNTUK ML API SERVICE
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import time
import logging
import numpy as np
import joblib

# PROMETHEUS METRICS
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# =========================================================
# INIT FASTAPI APP
# =========================================================
app = FastAPI()

# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================================================
# LOAD MODEL
# =========================================================
try:
    model = joblib.load("random_forest_model.pkl")
except Exception as e:
    model = None
    logging.error(f"MODEL FAILED TO LOAD: {str(e)}")

# =========================================================
# PROMETHEUS METRICS
# =========================================================

REQUEST_COUNT = Counter(
    "ml_request_total",
    "Total request ke inference API",
    ["endpoint", "method", "status"]
)

ERROR_COUNT = Counter(
    "ml_error_total",
    "Total error pada inference API",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Latency request inference"
)

PREDICTION_TIME = Histogram(
    "ml_prediction_time_seconds",
    "Waktu proses model prediksi"
)

PREDICTION_CLASS = Counter(
    "ml_prediction_class_total",
    "Distribusi kelas hasil prediksi",
    ["predicted_class"]
)

MODEL_STATUS = Gauge(
    "ml_model_status",
    "Status model ML"
)

IN_PROGRESS = Gauge(
    "ml_requests_in_progress",
    "Request yang sedang diproses"
)

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {"message": "ML Inference API Running"}

# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
def health():
    status = 1 if model is not None else 0
    MODEL_STATUS.set(status)

    return {
        "status": "ok" if model is not None else "unavailable",
        "model_ready": model is not None
    }

# =========================================================
# METRICS ENDPOINT
# =========================================================
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# =========================================================
# PREDICTION ENDPOINT
# =========================================================
@app.post("/predict")
async def predict(request: Request):

    IN_PROGRESS.inc()
    start_time = time.time()

    try:
        payload = await request.json()

        if "features" not in payload:
            raise ValueError("Missing 'features' in request body")

        # =====================================================
        # FEATURE SAFETY LAYER (FIX INPUT SIZE 27 FEATURES)
        # =====================================================
        expected_features = 27

        features_input = np.array(payload["features"]).flatten()

        # PAD jika kurang fitur
        if len(features_input) < expected_features:
            features_input = np.pad(
                features_input,
                (0, expected_features - len(features_input)),
                mode="constant"
            )

        # TRUNCATE jika lebih fitur
        elif len(features_input) > expected_features:
            features_input = features_input[:expected_features]

        # RESHAPE FINAL INPUT
        features = features_input.reshape(1, -1)

        if model is None:
            raise ValueError("Model belum berhasil diload")

        # LATENCY + PREDICTION TIME
        with REQUEST_LATENCY.time():
            with PREDICTION_TIME.time():
                prediction = model.predict(features)

        result = str(prediction[0])

        PREDICTION_CLASS.labels(predicted_class=result).inc()

        REQUEST_COUNT.labels(
            endpoint="/predict",
            method="POST",
            status="success"
        ).inc()

        latency = time.time() - start_time

        logging.info(f"PREDICT SUCCESS | RESULT={result} | LATENCY={latency:.4f}")

        return JSONResponse({
            "prediction": result,
            "latency": latency
        })

    except Exception as e:

        ERROR_COUNT.labels(endpoint="/predict").inc()

        REQUEST_COUNT.labels(
            endpoint="/predict",
            method="POST",
            status="error"
        ).inc()

        logging.error(f"PREDICT ERROR: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        IN_PROGRESS.dec()