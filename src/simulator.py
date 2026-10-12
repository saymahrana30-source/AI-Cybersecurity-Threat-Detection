"""Virtual cyber range: generates realistic network-flow records for normal users and attackers.

Each row is ONE network flow (like a NetFlow / CICFlowMeter record). Attack distributions deliberately
overlap with normal behaviour (backups, mistyped passwords, bursty traffic, low-and-slow attacks) so
the models are not trivially perfect - like real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import KNOWN_ATTACKS, NORMAL, NOVEL_ATTACK

COLUMNS = ["timestamp", "src_ip", "protocol", "service", "conn_state", "duration", "src_bytes", "dst_bytes",
           "packet_count", "syn_ratio", "unique_dst_ports", "conn_per_min", "login_attempts",
           "failed_logins", "label"]


def _pick(rng, options, p, n):
    return rng.choice(options, n, p=p)


def _frame(proto, service, state, duration, src, dst, pk, syn, ports, cpm, attempts, failed, label):
    return pd.DataFrame({
        "protocol": proto, "service": service, "conn_state": state,
        "duration": np.round(np.clip(duration, 0.001, 3600), 3),
        "src_bytes": np.round(np.clip(src, 20, None)).astype(int),
        "dst_bytes": np.round(np.clip(dst, 0, None)).astype(int),
        "packet_count": np.maximum(1, np.round(pk)).astype(int),
        "syn_ratio": np.round(np.clip(syn, 0, 1), 3),
        "unique_dst_ports": np.maximum(1, np.round(ports)).astype(int),
        "conn_per_min": np.maximum(1, np.round(cpm)).astype(int),
        "login_attempts": np.maximum(0, np.round(attempts)).astype(int),
        "failed_logins": np.maximum(0, np.round(failed)).astype(int),
        "label": label})


def gen_normal(rng, n):
    proto = _pick(rng, ["tcp", "udp", "icmp"], [.76, .20, .04], n)
    service = np.empty(n, dtype=object)
    tcp, udp = proto == "tcp", proto == "udp"
    service[tcp] = _pick(rng, ["http", "https", "ssh", "ftp", "smtp", "other"], [.30, .42, .05, .03, .06, .14], tcp.sum())
    service[udp] = _pick(rng, ["dns", "other"], [.85, .15], udp.sum())
    service[proto == "icmp"] = "other"
    duration = rng.lognormal(0.2, 1.0, n)
    src, dst = rng.lognormal(6.3, 1.1, n), rng.lognormal(7.8, 1.7, n)
    big = rng.random(n) < 0.02                                   # backups / large uploads
    src[big] = rng.lognormal(12, 0.8, big.sum()); duration[big] *= 8
    pk = 1 + rng.poisson(3 + (src + dst) / 1200)
    syn = rng.beta(1, 25, n)
    ports = 1 + rng.poisson(0.5, n)
    disc = rng.random(n) < 0.02                                  # service discovery / browsing
    ports[disc] = rng.integers(5, 14, disc.sum())
    cpm = 1 + rng.poisson(5, n)
    burst = rng.random(n) < 0.04                                 # bursty but legitimate (CDN, sync)
    cpm[burst] = 10 + rng.poisson(45, burst.sum())
    auth = np.isin(service, ["ssh", "ftp"])
    attempts = np.where(auth, 1 + rng.poisson(0.3, n), 0)
    failed = rng.binomial(attempts, 0.08)
    forgot = auth & (rng.random(n) < 0.015)                      # forgotten password
    attempts[forgot] = rng.integers(3, 7, forgot.sum()); failed[forgot] = rng.integers(2, 5, forgot.sum())
    state = _pick(rng, ["SF", "REJ", "RSTO", "S0"], [.93, .03, .02, .02], n)
    return _frame(proto, service, state, duration, src, dst, pk, syn, ports, cpm, attempts, failed, NORMAL)


def _gen_attack_raw(rng, kind, n):
    """Attack generators. A share of each attack is 'stealthy' (low-rate, blends into normal traffic)."""
    z = np.zeros(n)
    stealth = rng.random(n) < 0.40
    if kind == "DoS":              # SYN / HTTP flood: huge connection rate, tiny payloads
        syn_flood = rng.random(n) < 0.7
        cpm = np.where(stealth, rng.lognormal(np.log(35), 0.5, n), rng.lognormal(np.log(200), 0.7, n))
        syn = np.where(stealth, rng.beta(1.5, 14, n), np.where(syn_flood, rng.beta(8, 2, n), rng.beta(1, 20, n)))
        state = np.where(syn_flood & ~stealth, _pick(rng, ["S0", "SF", "REJ"], [.7, .2, .1], n), "SF")
        src = np.where(stealth, rng.lognormal(6.0, 0.8, n), rng.lognormal(4.8, 0.6, n))
        return _frame(np.full(n, "tcp"), _pick(rng, ["http", "https", "other"], [.5, .3, .2], n), state,
                      rng.lognormal(-0.3, 0.9, n), src, rng.lognormal(3.5, 1.4, n),
                      rng.poisson(30, n) + 5, syn, 1 + rng.poisson(0.3, n), cpm, z, z, kind)
    if kind == "PortScan":         # many distinct ports, tiny probes, mostly refused
        ports = np.where(stealth, rng.integers(3, 10, n), rng.lognormal(np.log(80), 0.9, n))
        cpm = np.where(stealth, rng.lognormal(np.log(10), 0.6, n), rng.lognormal(np.log(50), 0.6, n))
        return _frame(np.full(n, "tcp"), _pick(rng, ["other", "ssh", "http", "ftp"], [.7, .1, .1, .1], n),
                      np.where(stealth, _pick(rng, ["SF", "REJ", "S0"], [.5, .3, .2], n), _pick(rng, ["REJ", "S0", "RSTO", "SF"], [.5, .3, .1, .1], n)),
                      rng.lognormal(-1.5, 1.1, n), rng.lognormal(4.8, 0.6, n), rng.lognormal(2.5, 1.6, n),
                      rng.poisson(2, n) + 1, np.where(stealth, rng.beta(2, 8, n), rng.beta(4, 4, n)), ports, cpm, z, z, kind)
    if kind == "BruteForce":       # many login attempts, most of them failing
        slow = rng.random(n) < 0.40                               # low-and-slow variant
        attempts = np.where(slow, rng.integers(2, 6, n), rng.lognormal(np.log(25), 0.7, n))
        failed = attempts * np.where(slow, rng.beta(4, 4, n), rng.beta(8, 2, n))
        return _frame(np.full(n, "tcp"), _pick(rng, ["ssh", "ftp", "http"], [.55, .2, .25], n),
                      _pick(rng, ["SF", "RSTO", "REJ"], [.7, .2, .1], n),
                      rng.lognormal(1.0, 0.8, n), rng.lognormal(6.0, 0.7, n), rng.lognormal(5.8, 0.8, n),
                      rng.poisson(20, n) + 4, rng.beta(1, 15, n), 1 + rng.poisson(0.2, n),
                      np.where(slow, rng.lognormal(np.log(8), 0.6, n), rng.lognormal(np.log(25), 0.6, n)),
                      attempts, failed, kind)
    if kind == "DataExfiltration":  # long, upload-heavy flows to unusual services / DNS tunnelling
        dns = rng.random(n) < 0.25
        src = np.where(stealth, rng.lognormal(11.5, 0.9, n), rng.lognormal(13.5, 1.0, n))
        dur = np.where(stealth, rng.lognormal(3.2, 0.8, n), rng.lognormal(4.0, 0.9, n))
        return _frame(np.where(dns, "udp", "tcp"), np.where(dns, "dns", _pick(rng, ["https", "other", "ftp"], [.5, .35, .15], n)),
                      "SF", dur, src, rng.lognormal(5.5, 1.2, n), rng.poisson(900, n) + 50,
                      rng.beta(1, 25, n), 1 + rng.poisson(0.4, n), 1 + rng.poisson(2, n), z, z, kind)
    if kind == NOVEL_ATTACK:       # long-lived, low-volume covert C2 channel (never trained on)
        return _frame(np.full(n, "tcp"), _pick(rng, ["https", "other"], [.6, .4], n), "SF",
                      rng.lognormal(np.log(600), 0.4, n), rng.lognormal(6.0, 0.4, n), rng.lognormal(5.5, 0.4, n),
                      rng.poisson(30, n) + 10, rng.beta(1, 40, n), np.full(n, 1), 1 + rng.poisson(1, n), z, z, kind)
    raise ValueError(kind)


def gen_attack(rng, kind, n):
    """Attack flows; ~15% are 'camouflaged' (look like normal traffic - e.g. an attacker's harmless handshake)."""
    df = _gen_attack_raw(rng, kind, n)
    cam = rng.random(n) < 0.15
    if cam.any():
        norm = gen_normal(rng, int(cam.sum()))
        norm["label"] = kind
        df.loc[cam, :] = norm.values
        df = df.astype(dict(zip(norm.columns, norm.dtypes)))
    return df


def simulate_dataset(n=30000, seed=42, include_novel=False):
    """Static labelled dataset (random timestamps / IPs) for training and evaluation."""
    rng = np.random.default_rng(seed)
    mix = {"DoS": .04, "PortScan": .03, "BruteForce": .03, "DataExfiltration": .02}
    parts = [gen_normal(rng, n - sum(int(n * f) for f in mix.values()))]
    parts += [gen_attack(rng, k, int(n * f)) for k, f in mix.items()]
    if include_novel:
        parts.append(gen_attack(rng, NOVEL_ATTACK, int(n * .03)))
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    start = pd.Timestamp("2026-01-15 00:00:00")
    df.insert(0, "timestamp", (start + pd.to_timedelta(rng.uniform(0, 86400, len(df)), unit="s")).round("ms"))
    df.insert(1, "src_ip", [f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(len(df))])
    return df[COLUMNS].sort_values("timestamp").reset_index(drop=True)


# (src_ip, attack, start_min, end_min, flows_per_minute)
CAMPAIGNS = [
    ("203.0.113.66", NOVEL_ATTACK, 4, 56, 4),        # stealthy C2 beaconing - the "zero-day"
    ("45.133.1.7", "DoS", 10, 14, 60),
    ("185.220.4.21", "PortScan", 18, 22, 25),
    ("91.240.118.9", "BruteForce", 27, 35, 25),
    ("192.168.1.23", "DataExfiltration", 40, 50, 3),  # insider, low and slow
    ("198.51.100.4", "BruteForce", 50, 58, 3),        # low-and-slow brute force
]


def simulate_stream(seed=2024, minutes=60):
    """One hour of live traffic: 40 normal hosts + scripted attack campaigns (for the SOC replay)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-03-02 09:00:00")
    n_norm = int(minutes * 80)
    normal = gen_normal(rng, n_norm)
    normal["timestamp"] = (start + pd.to_timedelta(np.sort(rng.uniform(0, minutes * 60, n_norm)), unit="s")).round("ms")
    normal["src_ip"] = [f"192.168.1.{h}" for h in rng.integers(10, 50, n_norm)]
    parts, truth = [normal], []
    for ip, kind, m0, m1, rate in CAMPAIGNS:
        n = int((m1 - m0) * rate)
        a = gen_attack(rng, kind, n)
        a["timestamp"] = (start + pd.to_timedelta(np.sort(rng.uniform(m0 * 60, m1 * 60, n)), unit="s")).round("ms")
        a["src_ip"] = ip
        parts.append(a)
        truth.append({"src_ip": ip, "attack": kind, "start": start + pd.Timedelta(minutes=m0),
                      "end": start + pd.Timedelta(minutes=m1), "flows": n})
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return df[COLUMNS], truth
