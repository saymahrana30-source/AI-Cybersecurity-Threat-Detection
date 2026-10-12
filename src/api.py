"""REST API for real-time threat scoring.

    python -m src.api                      # http://127.0.0.1:5000
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
         -d '{"unique_dst_ports": 120, "conn_per_min": 60, "syn_ratio": 0.5, "conn_state": "REJ"}'

Only send the fields you know - the rest default to a typical normal flow.
"""
from __future__ import annotations
import os
import joblib
from flask import Flask, jsonify, request

NUMERIC = ["duration", "src_bytes", "dst_bytes", "packet_count", "syn_ratio", "unique_dst_ports",
           "conn_per_min", "login_attempts", "failed_logins"]
CATEGORICAL = ["protocol", "service", "conn_state"]
ALLOWED = set(NUMERIC + CATEGORICAL)



INDEX_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Threat Detection API</title>
<style>
body{font-family:system-ui,Arial,sans-serif;max-width:860px;margin:30px auto;padding:0 16px;color:#222}
h1{margin-bottom:4px} .sub{color:#666;margin-top:0}
button{padding:8px 14px;margin:4px 6px 4px 0;border:0;border-radius:6px;background:#1565c0;color:#fff;cursor:pointer}
button.alt{background:#455a64} textarea{width:100%;height:150px;font-family:monospace;font-size:13px}
pre{background:#263238;color:#e0f2f1;padding:14px;border-radius:8px;min-height:60px;white-space:pre-wrap}
.bad{color:#ff8a80;font-weight:bold} .ok{color:#a5d6a7;font-weight:bold} code{background:#eee;padding:1px 5px;border-radius:4px}
</style></head><body>
<h1>AI Threat Detection API</h1><p class="sub">Click a sample network flow, or edit the JSON, then press <b>Analyze</b>.</p>
<div>
<button onclick="load('normal')">Normal browsing</button>
<button onclick="load('dos')">SYN flood (DoS)</button>
<button onclick="load('scan')">Port scan</button>
<button onclick="load('brute')">SSH brute force</button>
<button onclick="load('exfil')">Large upload</button>
</div>
<textarea id="box"></textarea><br>
<button onclick="send()">Analyze</button>
<h3>Result</h3><pre id="out">(nothing yet)</pre>
<h3>Endpoints</h3>
<ul><li><code>GET /health</code> - status</li>
<li><code>POST /predict</code> - one flow (JSON)</li>
<li><code>POST /predict_batch</code> - list of flows</li></ul>
<script>
const S={
normal:{service:"https",src_bytes:600,dst_bytes:9000,conn_per_min:4,duration:1.2},
dos:{service:"http",conn_per_min:300,syn_ratio:0.85,conn_state:"S0",duration:0.1,src_bytes:120,dst_bytes:20,packet_count:35},
scan:{unique_dst_ports:150,conn_per_min:60,syn_ratio:0.5,conn_state:"REJ",src_bytes:70,dst_bytes:0,duration:0.05},
brute:{service:"ssh",login_attempts:40,failed_logins:38,conn_per_min:30,duration:3,src_bytes:420,dst_bytes:330,packet_count:24},
exfil:{service:"other",src_bytes:40000000,dst_bytes:200,duration:90,packet_count:30000,conn_per_min:2}};
function load(k){document.getElementById('box').value=JSON.stringify(S[k],null,2)}
async function send(){
 const out=document.getElementById('out');
 try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:document.getElementById('box').value});
 const j=await r.json();
 if(j.error){out.innerHTML='<span class="bad">Error: '+j.error+'</span>';return}
 out.innerHTML=(j.threat_detected?'<span class="bad">THREAT DETECTED</span>':'<span class="ok">Looks normal</span>')+'\n\n'+JSON.stringify(j,null,2)}
 catch(e){out.textContent='Request failed: '+e}}
load('scan');
</script></body></html>"""


def _validate(flow):
    if not isinstance(flow, dict) or not flow:
        return "body must be a non-empty JSON object"
    unknown = sorted(set(flow) - ALLOWED)
    if unknown:
        return f"unknown field(s) {unknown}; allowed: {sorted(ALLOWED)}"
    for k in NUMERIC:
        if k in flow and (isinstance(flow[k], bool) or not isinstance(flow[k], (int, float)) or flow[k] < 0):
            return f"'{k}' must be a non-negative number"
    for k in CATEGORICAL:
        if k in flow and not isinstance(flow[k], str):
            return f"'{k}' must be a string"
    return None


def create_app(model_path="models/threat_detector.joblib", detector=None):
    app = Flask(__name__)
    det = detector or joblib.load(model_path)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "classes": det.classes})

    @app.post("/predict")
    def predict():
        flow = request.get_json(silent=True)
        err = _validate(flow)
        if err:
            return jsonify({"error": err}), 400
        return jsonify(det.predict_one(flow))

    @app.post("/predict_batch")
    def predict_batch():
        flows = request.get_json(silent=True)
        if not isinstance(flows, list) or not flows or len(flows) > 1000:
            return jsonify({"error": "body must be a JSON list of 1-1000 flow objects"}), 400
        for f in flows:
            err = _validate(f)
            if err:
                return jsonify({"error": err}), 400
        return jsonify([det.predict_one(f) for f in flows])

    return app


if __name__ == "__main__":
    if not os.path.exists("models/threat_detector.joblib"):
        raise SystemExit("Train first:  python main.py")
    create_app().run(host="127.0.0.1", port=5000, debug=False)
