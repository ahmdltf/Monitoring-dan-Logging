from flask import Flask, request, jsonify, Response
import requests
import time
import psutil # Untuk monitoring sistem
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests') # Total request yang diterima
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency') # Waktu respons API
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second') # Throughput

# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage') # Penggunaan CPU
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage') # Penggunaan RAM

# Endpoint untuk Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    # Update metrik sistem setiap kali /metrics diakses
    CPU_USAGE.set(psutil.cpu_percent(interval=1)) # Ambil data CPU usage (persentase)
    RAM_USAGE.set(psutil.virtual_memory().percent) # Ambil data RAM usage (persentase)
   
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint untuk mengakses API model dan mencatat metrik
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc() # Tambah jumlah request
    THROUGHPUT.inc() # Tambah throughput (request per detik)

    # Kirim request ke API model
    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration) # Catat latensi
       
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)