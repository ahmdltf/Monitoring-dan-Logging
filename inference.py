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
# LOGGING SETUP UNTUK DEBUGGING DAN AUDIT
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================================================
# LOAD MODEL (PASTIKAN FILE ADA DI ROOT PROJECT)
# =========================================================
try:
    model = joblib.load("model.pkl")
    MODEL_READY = True
except Exception as e:
    model = None
    MODEL_READY = False
    logging.error(f"MODEL FAILED TO LOAD: {str(e)}")

# =========================================================
# PROMETHEUS METRICS (ADVANCED + SAFE)
# =========================================================

# TOTAL REQUEST MASUK
REQUEST_COUNT = Counter(
    "ml_request_total",
    "Total request ke inference API",
    ["endpoint", "method", "status"]
)

# ERROR COUNT
ERROR_COUNT = Counter(
    "ml_error_total",
    "Total error pada inference API",
    ["endpoint"]
)

# LATENCY REQUEST
REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Latency request inference"
)

# INFERENCE TIME
PREDICTION_TIME = Histogram(
    "ml_prediction_time_seconds",
    "Waktu proses model prediksi"
)

# DISTRIBUSI HASIL PREDIKSI
PREDICTION_CLASS = Counter(
    "ml_prediction_class_total",
    "Distribusi kelas hasil prediksi",
    ["predicted_class"]
)

# STATUS MODEL (1 READY, 0 NOT READY)
MODEL_STATUS = Gauge(
    "ml_model_status",
    "Status model ML"
)

MODEL_STATUS.set(1 if MODEL_READY else 0)

# REQUEST IN PROGRESS
IN_PROGRESS = Gauge(
    "ml_requests_in_progress",
    "Request yang sedang diproses"
)

# =========================================================
# ROOT ENDPOINT
# =========================================================
@app.get("/")
def root():
    return {
        "message": "ML Inference API Running"
    }

# =========================================================
# HEALTH CHECK ENDPOINT (WAJIB UNTUK MONITORING)
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_READY else "unavailable",
        "model_ready": MODEL_READY
    }

# =========================================================
# PROMETHEUS METRICS ENDPOINT
# =========================================================
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# =========================================================
# MAIN PREDICTION ENDPOINT
# =========================================================
@app.post("/predict")
async def predict(request: Request):

    IN_PROGRESS.inc()
    start_time = time.time()

    try:
        # AMBIL INPUT JSON
        payload = await request.json()

        # VALIDASI INPUT
        if "features" not in payload:
            raise ValueError("Missing 'features' in request body")

        # CONVERT INPUT KE NUMPY ARRAY
        features = np.array(payload["features"]).reshape(1, -1)

        # =====================================================
        # PREDICTION PROCESS
        # =====================================================
        with PREDICTION_TIME.time():
            prediction = model.predict(features)

        result = str(prediction[0])

        # CATAT DISTRIBUSI PREDIKSI
        PREDICTION_CLASS.labels(predicted_class=result).inc()

        # LATENCY TOTAL
        latency = time.time() - start_time

        # METRICS SUCCESS REQUEST
        REQUEST_COUNT.labels(
            endpoint="/predict",
            method="POST",
            status="success"
        ).inc()

        # LOGGING
        logging.info(f"PREDICT SUCCESS | RESULT={result} | LATENCY={latency:.4f}")

        return JSONResponse({
            "prediction": result,
            "latency": latency
        })

    except Exception as e:

        # ERROR METRICS
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