"""Create proof images from REAL runs:  python tools/make_screenshots.py
 (the API screenshot starts the Flask server on localhost:5057 and sends real requests to it)

 * runs the project's commands, saves their output to terminal_logs/*.txt
 * renders each as a terminal-style PNG in screenshots/terminal/
 * renders key source excerpts as PNGs in screenshots/code/
Needs: pygments (pip install pygments).  Fonts: DejaVu Sans Mono (falls back to PIL default).
"""
import ast
import os
import subprocess
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf",
              "/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Courier New.ttf"]


def font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def window(body: Image.Image, title: str, bg=(30, 31, 38)) -> Image.Image:
    bar_h, pad = 34, 10
    w, h = body.width + 2 * pad, body.height + bar_h + 2 * pad
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, bar_h], fill=(52, 54, 66))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + i * 22, 11, 24 + i * 22, 23], fill=c)
    d.text((w // 2, 17), title, font=font(14), fill=(200, 200, 210), anchor="mm")
    img.paste(body, (pad, bar_h + pad))
    return img


def line_color(s):
    if s.startswith("$"):
        return (130, 220, 130)
    if s.startswith("#"):
        return (120, 125, 145)
    if "FAIL" in s or "Traceback" in s:
        return (255, 110, 110)
    if "PASS" in s or "SUCCESS" in s or " passed" in s or s.strip() == "OK" or s.startswith("A* expands"):
        return (120, 230, 140)
    if s.startswith("[t="):
        return (240, 200, 110)
    if set(s.strip()) <= set("=-") and s.strip():
        return (110, 115, 130)
    return (225, 227, 235)


def terminal_png(cmd: str, output: str, path: str):
    f = font(14)
    lines = ([f"$ {cmd}"] if cmd else []) + output.rstrip().splitlines()
    cw = f.getlength("M")
    width = int(min(max(len(l) for l in lines), 215) * cw) + 10
    body = Image.new("RGB", (width, 20 * len(lines) + 6), (30, 31, 38))
    d = ImageDraw.Draw(body)
    for i, l in enumerate(lines):
        d.text((4, 3 + 20 * i), l[:215], font=f, fill=line_color(l))
    window(body, "Terminal - " + os.path.basename(ROOT)).save(path)


def run(cmd: str):
    r = subprocess.run(cmd.replace("python ", f'"{PY}" ', 1), shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).replace("\r", "")


TERMINALS = [
    ("01_install_check", "python -c \"import sklearn, pandas, numpy, flask; print('OK', 'scikit-learn', sklearn.__version__, 'pandas', pandas.__version__)\""),
    ("02_dataset_and_preprocessing", "python explore.py"),
    ("03_tests", "python run_tests.py"),
    ("04_train_evaluate_detect", "python main.py"),
    ("05_predict_samples", "python predict.py"),
]

CODE = [  # (file, qualified function name or None for the whole file, output name)
    ("src/simulator.py", ["_gen_attack_raw"], "01_attack_simulator"),
    ("src/preprocessing.py", ["build_preprocessor"], "02_preprocessing_pipeline"),
    ("src/models.py", ["fit_isolation_forest"], "03_isolation_forest"),
    ("src/detector.py", ["ThreatDetector.predict"], "04_threat_detector"),
    ("src/alerts.py", ["generate_alerts"], "05_alert_generation"),
    ("src/api.py", ["create_app"], "06_flask_api"),
    ("src/pipeline.py", ["train_and_evaluate"], "07_training_and_evaluation"),
    ("tests/test_alerts.py", None, "08_alert_tests"),
]


def extract(path, names):
    src = open(path, encoding="utf-8").read()
    if names is None:
        return src.strip("\n"), 1
    tree, wanted, chunks, first = ast.parse(src), names[0].split("."), [], None
    nodes = tree.body
    for part in wanted:
        node = next(n for n in nodes if getattr(n, "name", None) == part)
        nodes = getattr(node, "body", [])
    lines = src.splitlines()
    return textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno])), node.lineno


def code_png(path, names, out):
    from pygments import highlight
    from pygments.formatters import ImageFormatter
    from pygments.lexers import PythonLexer
    code, start = extract(path, names)
    fmt = ImageFormatter(font_name="DejaVu Sans Mono", font_size=15, style="monokai", line_numbers=True,
                         line_number_start=start if names is None else 1, image_pad=10, line_pad=3)
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(highlight(code, PythonLexer(), fmt))
    body = Image.open(tmp).convert("RGB")
    window(body, path + (f"  -  {names[0]}()" if names else ""), bg=(39, 40, 34)).save(out)
    os.remove(tmp)



def api_demo():
    """Start the real API server, send real requests, return text for a terminal screenshot."""
    import json
    import time
    import urllib.request
    env = dict(os.environ, PYTHONPATH=ROOT)
    srv = subprocess.Popen([PY, "-c", "from src.api import create_app; create_app().run(port=5057)"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    base = "http://127.0.0.1:5057"
    reqs = [("GET", "/health", None),
            ("POST", "/predict", {"service": "https", "src_bytes": 600, "dst_bytes": 9000, "conn_per_min": 4}),
            ("POST", "/predict", {"unique_dst_ports": 150, "conn_per_min": 60, "syn_ratio": 0.5, "conn_state": "REJ",
                                  "src_bytes": 70, "dst_bytes": 0, "duration": 0.05}),
            ("POST", "/predict", {"conn_per_min": "lots"})]
    out = ["# terminal 1:  python -m src.api   ->  running on http://127.0.0.1:5000", "# terminal 2:"]
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(base + "/health"); break
            except Exception:
                time.sleep(0.5)
        for method, path, body in reqs:
            data = json.dumps(body).encode() if body is not None else None
            cmd = f"curl -s {'-X POST ' if method == 'POST' else ''}localhost:5000{path}" + (
                f" -H 'Content-Type: application/json' -d '{json.dumps(body, separators=(',', ':'))}'" if body is not None else "")
            req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
            try:
                resp = urllib.request.urlopen(req).read().decode()
            except urllib.error.HTTPError as e:
                resp = e.read().decode()
            out += [f"$ {cmd}", resp.strip(), ""]
    finally:
        srv.terminate()
    return "\n".join(out)


def main():
    os.makedirs("terminal_logs", exist_ok=True)
    os.makedirs("screenshots/terminal", exist_ok=True)
    os.makedirs("screenshots/code", exist_ok=True)
    for name, cmd in TERMINALS:
        out = run(cmd)
        with open(f"terminal_logs/{name}.txt", "w", encoding="utf-8") as f:
            f.write(f"$ {cmd}\n{out}")
        terminal_png(cmd, out, f"screenshots/terminal/{name}.png")
        print("saved terminal", name)
    text = api_demo()
    open("terminal_logs/06_api_requests.txt", "w", encoding="utf-8").write(text)
    terminal_png(None, text, "screenshots/terminal/06_api_requests.png")
    print("saved terminal 06_api_requests")
    for path, names, name in CODE:
        try:
            code_png(path, names, f"screenshots/code/{name}.png")
            print("saved code", name)
        except Exception as e:                                    # pygments missing etc.
            print("skipped code", name, "->", e)


if __name__ == "__main__":
    main()
