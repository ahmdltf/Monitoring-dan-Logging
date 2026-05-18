# IMPORT LIBRARY UTAMA UNTUK EXPORTER
from flask import Flask, Response
import psutil
import time
import requests

# PROMETHEUS CLIENT UNTUK METRICS
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# INISIALISASI FLASK APP SEBAGAI EXPORTER SERVICE
app = Flask(__name__)

# =========================================================
# METRICS API LAYER
# =========================================================

# TOTAL REQUEST KE MODEL API
REQUEST_TOTAL = Counter(
    "ml_request_total",
    "Total request ke ML inference service"
)

# ERROR COUNT DARI SERVICE
ERROR_TOTAL = Counter(
    "ml_error_total",
    "Total error dari ML service"
)

# LATENCY REQUEST KE MODEL
REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Latency request ke model inference"
)

# =========================================================
# METRICS SISTEM (INFRASTRUCTURE LEVEL)
# =========================================================

# CPU USAGE SISTEM
CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "CPU usage sistem dalam persen"
)

# RAM USAGE SISTEM
RAM_USAGE = Gauge(
    "system_ram_usage_percent",
    "RAM usage sistem dalam persen"
)

# DISK USAGE SISTEM
DISK_USAGE = Gauge(
    "system_disk_usage_percent",
    "Disk usage sistem dalam persen"
)

# NETWORK I/O (BASIC INDICATOR)
NETWORK_SENT = Gauge(
    "system_network_sent_bytes",
    "Total bytes sent"
)

NETWORK_RECV = Gauge(
    "system_network_recv_bytes",
    "Total bytes received"
)

# =========================================================
# MODEL HEALTH STATUS
# =========================================================

MODEL_STATUS = Gauge(
    "ml_model_status",
    "Status model (1 = up, 0 = down)"
)

MODEL_STATUS.set(1)

# =========================================================
# ENDPOINT: UPDATE SYSTEM METRICS
# =========================================================
def update_system_metrics():
    # AMBIL CPU USAGE TANPA BLOCKING
    CPU_USAGE.set(psutil.cpu_percent(interval=None))

    # AMBIL RAM USAGE
    RAM_USAGE.set(psutil.virtual_memory().percent)

    # AMBIL DISK USAGE
    DISK_USAGE.set(psutil.disk_usage('/').percent)

    # AMBIL NETWORK STATISTICS
    net = psutil.net_io_counters()
    NETWORK_SENT.set(net.bytes_sent)
    NETWORK_RECV.set(net.bytes_recv)

# =========================================================
# ENDPOINT: CHECK HEALTH ML SERVICE
# =========================================================
def check_model_health():
    try:
        # CEK KE FASTAPI INFERENCE SERVICE
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)

        if response.status_code == 200:
            MODEL_STATUS.set(1)
        else:
            MODEL_STATUS.set(0)

    except:
        MODEL_STATUS.set(0)

# =========================================================
# ENDPOINT PROMETHEUS SCRAPE
# =========================================================
@app.route("/metrics")
def metrics():

    # UPDATE SYSTEM METRICS SETIAP REQUEST
    update_system_metrics()

    # CEK STATUS MODEL SETIAP SCRAPE
    check_model_health()

    # RETURN FORMAT PROMETHEUS
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# =========================================================
# OPTIONAL: SIMULASI TRACKING REQUEST MANUAL
# (BISA DIPANGGIL DARI INFERENCE JIKA MAU ADVANCED++)
# =========================================================
@app.route("/track_request")
def track_request():

    start = time.time()

    try:
        # SIMULASI REQUEST KE INFERENCE API
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"features": [1, 2, 3, 4]}
        )

        REQUEST_TOTAL.inc()

        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)

        return response.json()

    except Exception as e:

        ERROR_TOTAL.inc()

        return {"error": str(e)}

# =========================================================
# MAIN RUNNER
# =========================================================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8001,
        debug=False
    )