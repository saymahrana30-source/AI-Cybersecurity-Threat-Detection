import dataclasses
from src.api import create_app
from src.config import Config
from src.pipeline import train_and_evaluate
from src.simulator import simulate_dataset

_state = {}


def _client():
    if not _state:
        cfg = dataclasses.replace(Config(), rf_trees=40, iso_trees=50)
        det, _, _ = train_and_evaluate(simulate_dataset(6000, seed=5), cfg, cv=False, log=lambda *_: None)
        _state["c"] = create_app(detector=det).test_client()
    return _state["c"]


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_predict_flags_a_port_scan():
    r = _client().post("/predict", json={"unique_dst_ports": 200, "conn_per_min": 60, "syn_ratio": 0.5,
                                          "conn_state": "REJ", "src_bytes": 70, "dst_bytes": 0, "duration": 0.05})
    body = r.get_json()
    assert r.status_code == 200 and body["threat_detected"] is True and 0 <= body["confidence"] <= 1


def test_predict_accepts_partial_payload_as_normal():
    r = _client().post("/predict", json={"service": "https", "conn_per_min": 3})
    assert r.status_code == 200 and r.get_json()["threat_detected"] is False


def test_bad_input_is_rejected_with_400():
    c = _client()
    assert c.post("/predict", json={"conn_per_min": "lots"}).status_code == 400
    assert c.post("/predict", json={"bogus_field": 1}).status_code == 400
    assert c.post("/predict", data="not json", content_type="text/plain").status_code == 400
    assert c.post("/predict_batch", json={"a": 1}).status_code == 400


def test_batch_endpoint():
    r = _client().post("/predict_batch", json=[{"conn_per_min": 3}, {"unique_dst_ports": 300, "conn_per_min": 80}])
    assert r.status_code == 200 and len(r.get_json()) == 2


def test_home_page_is_served_instead_of_404():
    r = _client().get("/")
    assert r.status_code == 200 and b"Threat Detection API" in r.data


def test_home_page_javascript_keeps_escaped_newlines():
    """Regression: a real line break inside a JS string literal silently disables every button."""
    from src.api import INDEX_HTML
    script = INDEX_HTML.split("<script>")[1]
    assert "'\\n\\n'" in script
