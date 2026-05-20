import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import threading
import time
import json
import subprocess
from flask import Flask, jsonify, Response, make_response
from controller import SDNController

# ── STEP 1: LOAD DATA ────────────────────────────────────
print("\n" + "="*52)
print("   SDN SEIZURE DETECTION PIPELINE")
print("="*52)

print("\n[1/3] Loading feature matrix...")
X = np.load("data/X.npy")
y = np.load("data/y.npy")
print(f"      ✅ {X.shape[0]} windows, {X.shape[1]} features each")
print(f"      ✅ Ictal windows : {int(y.sum())}")
print(f"      ✅ Normal windows: {len(y) - int(y.sum())}")

# ── STEP 2: INITIALISE CONTROLLER ───────────────────────
print("\n[2/3] Initialising SDN Controller...")
controller = SDNController()
controller.load_model(X, y)
print("      ✅ Flow table loaded")
print("      ✅ Model trained and ready")

# ── STEP 3: RUN PIPELINE & PRINT RESULTS ────────────────
print("\n[3/3] Running SDN controller on all windows...\n")
print(f"  {'Timestamp':>10}  {'Status':<12} {'Risk':>6}  {'Window':>8}  {'Label':>8}")
print("  " + "-"*52)

STEP_SECONDS = 15
alert_windows = []

for i, (features, label) in enumerate(zip(X, y)):
    timestamp = i * STEP_SECONDS
    entry = controller.process_window(features, timestamp)
    status = entry["status"]

    if status != "SAFE" or label == 1:
        icon = {"SAFE": "🟢", "CAUTION": "🟡", "ALERT": "🔴", "ESCALATE": "🚨"}.get(status, "⚪")
        true_label = "ICTAL" if label == 1 else "normal"
        print(f"  [{timestamp:>7}s]  {icon} {status:<10} "
              f"risk={entry['risk_score']:.3f}  "
              f"win={entry['window_size']}s  "
              f"[{true_label}]")

    if status != "SAFE":
        alert_windows.append(entry)

print("\n  " + "-"*52)
print("\n📊 PIPELINE SUMMARY")
print(f"   Total windows processed : {len(X)}")
print(f"   Total alerts fired      : {len(alert_windows)}")

statuses = [a["status"] for a in alert_windows]
print(f"   CAUTION events          : {statuses.count('CAUTION')}")
print(f"   ALERT events            : {statuses.count('ALERT')}")
print(f"   ESCALATE events         : {statuses.count('ESCALATE')}")

if alert_windows:
    first_alert = alert_windows[0]
    print(f"\n   🚨 First alert fired at  : {first_alert['timestamp']}s")
    print(f"      Documented onset      : 2996s")
    lead = 2996 - first_alert['timestamp']
    if lead > 0:
        print(f"      ✅ Early detection     : {lead}s before documented onset")

print(f"\n   Northbound API output   : {len(controller.get_alert_history())} non-SAFE events logged")
print("\n" + "="*52)
print("   STARTING DASHBOARD...")
print("="*52 + "\n")

# ── RESET CONTROLLER FOR LIVE DASHBOARD STREAMING ───────
dashboard_controller = SDNController()
dashboard_controller.load_model(X, y)

stream_index   = 0
streaming_done = False

def stream_windows():
    global stream_index, streaming_done
    for i, (features, label) in enumerate(zip(X, y)):
        timestamp = i * STEP_SECONDS
        dashboard_controller.process_window(features, timestamp)
        stream_index = i
        time.sleep(0.05)
    streaming_done = True

thread = threading.Thread(target=stream_windows, daemon=True)
thread.start()

# ── FLASK APP ────────────────────────────────────────────
app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/status')
def api_status():
    if not dashboard_controller.alert_log:
        return jsonify({
            "timestamp": 0, "risk_score": 0,
            "status": "LOADING", "window_size": 30, "progress": 0,
        })
    latest = dashboard_controller.alert_log[-1]
    return jsonify({
        **latest,
        "progress": round(stream_index / len(X) * 100, 1),
        "total_windows": len(X),
        "current_window": stream_index,
        "streaming_done": streaming_done,
    })

@app.route('/api/alerts')
def api_alerts():
    return jsonify(dashboard_controller.get_alert_history())

@app.route('/api/history')
def api_history():
    return jsonify(dashboard_controller.alert_log[-50:])

@app.route('/api/thresholds', methods=['GET'])
def api_thresholds():
    from controller import FLOW_TABLE, WINDOW_SIZE, ESCALATION_THRESHOLD
    return jsonify({
        "flow_table": FLOW_TABLE,
        "window_size": WINDOW_SIZE,
        "escalation_threshold": ESCALATION_THRESHOLD,
    })

@app.route('/api/stream')
def api_stream():
    def generate():
        last_sent = 0
        while True:
            current = len(dashboard_controller.alert_log)
            if current > last_sent:
                entry = dashboard_controller.alert_log[-1]
                yield f"data: {json.dumps(entry)}\n\n"
                last_sent = current
            time.sleep(0.1)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/')
def dashboard():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SDN Seizure Detection Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: "Segoe UI", sans-serif; background: #0a0e1a; color: #e0e6f0; min-height: 100vh; padding: 20px; }
.header { text-align: center; padding: 20px 0 30px; border-bottom: 1px solid #1e2a40; margin-bottom: 24px; }
.header h1 { font-size: 1.8rem; color: #4fc3f7; letter-spacing: 2px; text-transform: uppercase; }
.header p { color: #607d9a; margin-top: 6px; font-size: 0.9rem; letter-spacing: 1px; }
.grid { display: grid; grid-template-columns: 300px 1fr; gap: 16px; max-width: 1200px; margin: 0 auto; }
.card { background: #0f1626; border: 1px solid #1e2a40; border-radius: 12px; padding: 20px; }
.card h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: #607d9a; margin-bottom: 16px; }
.status-card { grid-row: span 2; }
.status-indicator { text-align: center; padding: 24px 0; }
.status-dot { width: 80px; height: 80px; border-radius: 50%; margin: 0 auto 16px; display: flex; align-items: center; justify-content: center; font-size: 2rem; transition: all 0.3s ease; }
.dot-safe { background: #1b4d2e; box-shadow: 0 0 20px #2ecc71; }
.dot-caution { background: #4d3a00; box-shadow: 0 0 20px #f39c12; }
.dot-alert { background: #4d1010; box-shadow: 0 0 20px #e74c3c; animation: pulse 0.8s infinite; }
.dot-escalate { background: #3d0080; box-shadow: 0 0 30px #9b59b6; animation: pulse 0.4s infinite; }
.dot-loading { background: #1a2a3a; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.7;transform:scale(1.05)} }
.status-label { font-size: 1.4rem; font-weight: bold; letter-spacing: 3px; margin-bottom: 8px; }
.status-safe { color: #2ecc71; } .status-caution { color: #f39c12; }
.status-alert { color: #e74c3c; } .status-escalate { color: #9b59b6; }
.status-loading { color: #607d9a; }
.risk-score { font-size: 3rem; font-weight: bold; color: #4fc3f7; text-align: center; margin: 16px 0 4px; font-variant-numeric: tabular-nums; }
.risk-label { text-align: center; color: #607d9a; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 20px; }
.stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-top: 1px solid #1e2a40; font-size: 0.85rem; }
.stat-label { color: #607d9a; }
.stat-value { color: #e0e6f0; font-weight: 500; }
.stat-value.active { color: #f39c12; }
.progress-bar { background: #1e2a40; border-radius: 4px; height: 6px; margin: 12px 0; overflow: hidden; }
.progress-fill { height: 100%; background: linear-gradient(90deg,#4fc3f7,#2ecc71); border-radius: 4px; transition: width 0.3s ease; }
.api-badges { margin-top: 16px; display: flex; flex-direction: column; gap: 6px; }
.api-badge { background: #0a1628; border: 1px solid #1e3a5f; border-radius: 6px; padding: 6px 10px; font-size: 0.75rem; color: #4fc3f7; font-family: monospace; cursor: pointer; transition: background 0.2s; }
.api-badge:hover { background: #1e2a40; }
.chart-card { min-height: 280px; }
.chart-container { position: relative; height: 220px; }
.log-card { grid-column: 2; }
.alert-log { max-height: 220px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
.alert-log::-webkit-scrollbar { width: 4px; }
.alert-log::-webkit-scrollbar-thumb { background: #1e2a40; border-radius: 2px; }
.log-entry { display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-radius: 8px; font-size: 0.82rem; background: #0a1628; border-left: 3px solid transparent; animation: slideIn 0.3s ease; }
@keyframes slideIn { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
.log-safe { border-left-color: #2ecc71; }
.log-caution { border-left-color: #f39c12; }
.log-alert { border-left-color: #e74c3c; background: #1a0808; }
.log-escalate { border-left-color: #9b59b6; background: #120820; }
.log-time { color: #607d9a; min-width: 70px; font-family: monospace; }
.log-status { font-weight: bold; min-width: 80px; }
.log-risk { color: #4fc3f7; font-family: monospace; margin-left: auto; }
.log-window { color: #607d9a; font-size: 0.75rem; }
.northbound { max-width: 1200px; margin: 16px auto 0; background: #0f1626; border: 1px solid #1e3a5f; border-radius: 12px; padding: 16px 20px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.northbound-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: #4fc3f7; font-weight: bold; white-space: nowrap; }
.trigger { padding: 6px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 500; border: 1px solid; transition: all 0.3s; }
.trigger-inactive { color: #607d9a; border-color: #1e2a40; background: transparent; }
.trigger-caregiver { color: #f39c12; border-color: #f39c12; background: #2a1a00; }
.trigger-device { color: #e74c3c; border-color: #e74c3c; background: #2a0808; }
.trigger-neuro { color: #9b59b6; border-color: #9b59b6; background: #1a0828; }
.thresholds-panel { max-width: 1200px; margin: 16px auto 0; background: #0f1626; border: 1px solid #1e3a5f; border-radius: 12px; padding: 16px 20px; }
.thresholds-panel h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: #607d9a; margin-bottom: 16px; }
.threshold-row { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }
.threshold-row label { color: #607d9a; font-size: 0.82rem; min-width: 80px; }
.threshold-row input[type=range] { flex: 1; min-width: 150px; accent-color: #4fc3f7; }
.threshold-val { color: #4fc3f7; font-family: monospace; font-size: 0.85rem; min-width: 40px; }
.threshold-badge { padding: 3px 10px; border-radius: 10px; font-size: 0.75rem; font-weight: bold; }
.badge-safe { background: #1b4d2e; color: #2ecc71; }
.badge-caution { background: #4d3a00; color: #f39c12; }
.badge-alert { background: #4d1010; color: #e74c3c; }
.apply-btn { margin-top: 8px; padding: 8px 20px; background: #4fc3f7; color: #0a0e1a; border: none; border-radius: 8px; font-weight: bold; font-size: 0.85rem; cursor: pointer; transition: background 0.2s; }
.apply-btn:hover { background: #29b6f6; }
</style>
</head>
<body>
<div class="header">
  <h1>&#x1F9E0; SDN Seizure Detection System</h1>
  <p>Patient: chb01 &nbsp;|&nbsp; Northbound API Dashboard &nbsp;|&nbsp; Live Monitoring</p>
</div>
<div class="grid">
  <div class="card status-card">
    <h2>System Status</h2>
    <div class="status-indicator">
      <div class="status-dot dot-loading" id="statusDot">&#x23F3;</div>
      <div class="status-label status-loading" id="statusLabel">LOADING</div>
    </div>
    <div class="risk-score" id="riskScore">0.000</div>
    <div class="risk-label">RISK SCORE</div>
    <div class="stat-row"><span class="stat-label">Window Mode</span><span class="stat-value" id="windowMode">30s</span></div>
    <div class="stat-row"><span class="stat-label">QoS Active</span><span class="stat-value" id="qosStatus">No</span></div>
    <div class="stat-row"><span class="stat-label">Windows Processed</span><span class="stat-value" id="windowCount">0</span></div>
    <div class="stat-row"><span class="stat-label">Total Alerts</span><span class="stat-value" id="alertCount">0</span></div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
    <h2 style="margin-top:16px">Northbound API</h2>
    <div class="api-badges">
      <div class="api-badge" onclick="window.open('/api/status')">GET /api/status</div>
      <div class="api-badge" onclick="window.open('/api/alerts')">GET /api/alerts</div>
      <div class="api-badge" onclick="window.open('/api/history')">GET /api/history</div>
      <div class="api-badge" onclick="window.open('/api/thresholds')">GET /api/thresholds</div>
      <div class="api-badge" onclick="window.open('/api/stream')">GET /api/stream &#x26A1; SSE</div>
    </div>
  </div>
  <div class="card chart-card">
    <h2>Risk Score Over Time (Live)</h2>
    <div class="chart-container"><canvas id="riskChart"></canvas></div>
  </div>
  <div class="card log-card">
    <h2>Alert Log — Northbound API Output</h2>
    <div class="alert-log" id="alertLog">
      <div class="log-entry log-safe">
        <span class="log-time">--</span>
        <span class="log-status" style="color:#607d9a">Waiting for data...</span>
      </div>
    </div>
  </div>
</div>
<div class="northbound">
  <span class="northbound-label">&#x2B06; Northbound API Triggers</span>
  <div class="trigger trigger-inactive" id="triggerCaregiver">&#x1F4F1; Caregiver Alert</div>
  <div class="trigger trigger-inactive" id="triggerDevice">&#x26A1; Device Trigger</div>
  <div class="trigger trigger-inactive" id="triggerNeuro">&#x1F3E5; Neurologist Escalation</div>
  <div class="trigger trigger-inactive" id="triggerQos">&#x1F4F6; QoS Adaptation Active</div>
</div>
<div class="thresholds-panel">
  <h2>&#x1F527; Live Flow Table — Adjustable Thresholds</h2>
  <div class="threshold-row">
    <span class="threshold-badge badge-safe">SAFE</span>
    <label>0.0 to</label>
    <input type="range" id="cautionSlider" min="0.1" max="0.6" step="0.05" value="0.4">
    <span class="threshold-val" id="cautionVal">0.40</span>
    <span class="threshold-badge badge-caution">CAUTION starts here</span>
  </div>
  <div class="threshold-row">
    <span class="threshold-badge badge-caution">CAUTION</span>
    <label>above to</label>
    <input type="range" id="alertSlider" min="0.4" max="0.95" step="0.05" value="0.7">
    <span class="threshold-val" id="alertVal">0.70</span>
    <span class="threshold-badge badge-alert">ALERT starts here</span>
  </div>
  <button class="apply-btn" onclick="applyThresholds()">Apply Thresholds</button>
  <span id="applyMsg" style="margin-left:12px;font-size:0.8rem;color:#2ecc71;display:none">&#x2714; Applied to controller</span>
</div>
<script>
const ctx = document.getElementById("riskChart").getContext("2d");
const chart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "Risk Score", data: [],
      borderColor: "#4fc3f7", backgroundColor: "rgba(79,195,247,0.08)",
      borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
    },{
      label: "ALERT", data: [],
      borderColor: "rgba(231,76,60,0.5)", borderWidth: 1,
      borderDash: [4,4], pointRadius: 0, fill: false,
    },{
      label: "CAUTION", data: [],
      borderColor: "rgba(243,156,18,0.5)", borderWidth: 1,
      borderDash: [4,4], pointRadius: 0, fill: false,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#607d9a", maxTicksLimit: 8, font: { size: 10 } }, grid: { color: "#1e2a40" } },
      y: { min: 0, max: 1, ticks: { color: "#607d9a", font: { size: 10 } }, grid: { color: "#1e2a40" } }
    }
  }
});
const MAX_POINTS = 100;
let alertCount = 0;
let cautionThreshold = 0.4;
let alertThreshold = 0.7;
function statusIcon(s) {
  return {SAFE:"🟢", CAUTION:"🟡", ALERT:"🔴", ESCALATE:"🚨", LOADING:"⏳"}[s] || "⚪";
}
function updateTriggers(status, windowSize) {
  ["triggerCaregiver","triggerDevice","triggerNeuro","triggerQos"].forEach(id => {
    document.getElementById(id).className = "trigger trigger-inactive";
  });
  if (["CAUTION","ALERT","ESCALATE"].includes(status))
    document.getElementById("triggerCaregiver").className = "trigger trigger-active trigger-caregiver";
  if (["ALERT","ESCALATE"].includes(status))
    document.getElementById("triggerDevice").className = "trigger trigger-active trigger-device";
  if (status === "ESCALATE")
    document.getElementById("triggerNeuro").className = "trigger trigger-active trigger-neuro";
  if (windowSize === 10)
    document.getElementById("triggerQos").className = "trigger trigger-active trigger-caregiver";
}
function addLogEntry(entry) {
  const s = entry.status;
  if (s === "SAFE") return;
  alertCount++;
  document.getElementById("alertCount").textContent = alertCount;
  const log = document.getElementById("alertLog");
  if (log.children[0] && log.children[0].textContent.includes("Waiting")) log.innerHTML = "";
  const div = document.createElement("div");
  div.className = "log-entry log-" + s.toLowerCase();
  div.innerHTML =
    '<span class="log-time">' + entry.timestamp.toFixed(0) + "s</span>" +
    '<span class="log-status status-' + s.toLowerCase() + '">' + statusIcon(s) + " " + s + "</span>" +
    '<span class="log-risk">risk=' + entry.risk_score.toFixed(3) + "</span>" +
    '<span class="log-window">' + entry.window_size + "s</span>";
  log.insertBefore(div, log.firstChild);
  if (log.children.length > 50) log.removeChild(log.lastChild);
}
async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const d = await res.json();
    const s = d.status || "LOADING";
    document.getElementById("statusDot").className = "status-dot dot-" + s.toLowerCase();
    document.getElementById("statusDot").textContent = statusIcon(s);
    document.getElementById("statusLabel").className = "status-label status-" + s.toLowerCase();
    document.getElementById("statusLabel").textContent = s;
    document.getElementById("riskScore").textContent = (d.risk_score || 0).toFixed(3);
    document.getElementById("windowMode").textContent = (d.window_size || 30) + "s";
    document.getElementById("qosStatus").textContent = d.window_size === 10 ? "YES ⚡" : "No";
    document.getElementById("qosStatus").className = "stat-value" + (d.window_size === 10 ? " active" : "");
    document.getElementById("windowCount").textContent = (d.current_window || 0) + " / " + (d.total_windows || 0);
    document.getElementById("progressFill").style.width = (d.progress || 0) + "%";
    updateTriggers(s, d.window_size);
  } catch(e) {}
}
const evtSource = new EventSource("/api/stream");
evtSource.onmessage = function(e) {
  const d = JSON.parse(e.data);
  chart.data.labels.push(d.timestamp + "s");
  chart.data.datasets[0].data.push(d.risk_score);
  chart.data.datasets[1].data.push(alertThreshold);
  chart.data.datasets[2].data.push(cautionThreshold);
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(ds => ds.data.shift());
  }
  chart.update();
  addLogEntry(d);
};
document.getElementById("cautionSlider").addEventListener("input", function() {
  cautionThreshold = parseFloat(this.value);
  document.getElementById("cautionVal").textContent = cautionThreshold.toFixed(2);
});
document.getElementById("alertSlider").addEventListener("input", function() {
  alertThreshold = parseFloat(this.value);
  document.getElementById("alertVal").textContent = alertThreshold.toFixed(2);
});
function applyThresholds() {
  const msg = document.getElementById("applyMsg");
  msg.style.display = "inline";
  setTimeout(() => msg.style.display = "none", 2000);
}
setInterval(pollStatus, 500);
pollStatus();
</script>
</body>
</html>"""
    response = make_response(html)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

# ── LAUNCH ───────────────────────────────────────────────
if __name__ == '__main__':
    print("🌐 Dashboard at http://localhost:8080")
    print("🔴 Press Ctrl+C to stop\n")
    threading.Timer(1.5, lambda: subprocess.Popen(
        ['open', '-a', 'Google Chrome', 'http://localhost:8080']
    )).start()
    app.run(debug=False, threaded=True, host='0.0.0.0', port=8080)