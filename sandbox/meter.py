#!/usr/bin/env python3
"""The credit meter: a proxy between the agents and an Anthropic-compatible endpoint that records the
tokens and the cost of every request, stops at a budget, and keeps the endpoint's key on the host.

  meter.py serve --upstream URL --key-file FILE --price IN,CACHED,OUT --ledger FILE [--budget USD]
                 [--port N] [--bind ADDR]
  meter.py report LEDGER [--since ISO]

`serve` listens on ADDR:N (default 0.0.0.0:18800). The agents use `http://<host>:N/<label>` as
ANTHROPIC_BASE_URL (the label, such as `prover` or `reviewer`, tags their requests in the ledger) and
the meter's own client key as ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY: a random key the meter writes
to LEDGER.client-key (mode 600) at its first start and keeps across restarts. A request without it, or
from an address outside the private ranges, is refused. The meter forwards each request to URL with the
endpoint's key from FILE, which never leaves the host.

The cost of a request is IN, CACHED and OUT dollars per million uncached input, cached input and output
tokens, read from the response's `usage` (in a stream, from its last `message_delta`, the one with the
final counts). Each request is a line of LEDGER (JSON): time, label, model, status, tokens, cost and
duration; the ledger is appended to, never rewritten, and a restarted meter counts what it holds. Once
the total reaches the budget, the meter answers every request with HTTP 402 (`billing_error`), so the
agents stop.

The endpoint caches prompts per replica, so the meter routes each label's requests to one replica: it adds
`x-session-affinity: <label>-<id>` (the id is fixed per ledger) unless the request carries one.

`report` sums a ledger by label and by model.

Standard library only.
"""
import argparse
import datetime
import hashlib
import http.server
import ipaddress
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request

PRIVATE = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "authorization", "x-api-key",
       "accept-encoding", "proxy-authorization", "te", "upgrade"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_ledger(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def usage_of(raw, stream):
    """The usage a response reports: (uncached input, cached input, output) tokens, and the model."""
    u, model = {}, None
    if stream:
        for line in raw.decode("utf-8", "replace").splitlines():
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            if ev.get("type") == "message_start":
                model = (ev.get("message") or {}).get("model", model)
                for k, v in ((ev.get("message") or {}).get("usage") or {}).items():
                    u.setdefault(k, v)
            elif ev.get("type") == "message_delta" and ev.get("usage"):
                u.update(ev["usage"])
    else:
        try:
            body = json.loads(raw)
        except ValueError:
            body = {}
        u, model = body.get("usage") or {}, body.get("model")
    n = lambda k: int(u.get(k) or 0)
    return n("input_tokens"), n("cache_read_input_tokens"), n("output_tokens"), model


class Meter:
    def __init__(self, a):
        self.upstream = a.upstream.rstrip("/")
        self.key = open(a.key_file).read().strip()
        self.price = [float(x) / 1e6 for x in a.price.split(",")]
        self.budget = a.budget
        self.ledger = a.ledger
        kp = a.ledger + ".client-key"
        if not os.path.exists(kp):
            fd = os.open(kp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, ("meter-" + secrets.token_urlsafe(24)).encode())
            os.close(fd)
        self.client_key = open(kp).read().strip()
        # the endpoint caches per replica: one routing key per ledger keeps an agent's prompts on one replica
        self.affinity = hashlib.sha256(os.path.abspath(a.ledger).encode()).hexdigest()[:16]
        self.lock = threading.Lock()
        self.spent = sum(r.get("cost", 0) for r in read_ledger(a.ledger))

    def record(self, row):
        with self.lock:
            self.spent += row["cost"]
            row["total"] = round(self.spent, 6)
            with open(self.ledger, "a") as f:
                f.write(json.dumps(row) + "\n")


def handler(M):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, typ, msg):
            data = json.dumps({"type": "error", "error": {"type": typ, "message": msg}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.forward(None)

        def do_POST(self):
            self.forward(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

        def forward(self, body):
            if not any(ipaddress.ip_address(self.client_address[0]) in n for n in PRIVATE):
                return self.reply(403, "permission_error", "meter: not from a private address")
            given = self.headers.get("x-api-key") or (self.headers.get("authorization") or "").removeprefix("Bearer ").strip()
            if not secrets.compare_digest(given, M.client_key):
                return self.reply(401, "authentication_error", "meter: unknown client key")
            parts = self.path.lstrip("/").split("/", 1)
            label, path = (parts[0], "/" + (parts[1] if len(parts) > 1 else "")) if parts[0] != "v1" else ("-", self.path)
            if M.spent >= M.budget:
                return self.reply(402, "billing_error", f"meter: the budget of ${M.budget:.2f} is spent")
            stream = False
            if body:
                try:
                    stream = bool(json.loads(body).get("stream"))
                except ValueError:
                    pass
            req = urllib.request.Request(M.upstream + path, data=body, method=self.command)
            for k, v in self.headers.items():
                if k.lower() not in HOP:
                    req.add_header(k, v)
            req.add_header("Authorization", f"Bearer {M.key}")
            if not self.headers.get("x-session-affinity"):
                req.add_header("x-session-affinity", f"{label}-{M.affinity}")
            req.add_header("Accept-Encoding", "identity")
            t0, raw, status = time.time(), b"", 0
            try:
                resp = urllib.request.urlopen(req, timeout=900)
            except urllib.error.HTTPError as e:
                resp = e
            except (urllib.error.URLError, OSError) as e:
                M.record({"t": now(), "label": label, "path": path, "status": 0, "error": str(e)[:200],
                          "in": 0, "cached": 0, "out": 0, "cost": 0.0, "ms": int(1000 * (time.time() - t0))})
                return self.reply(502, "api_error", f"meter: upstream unreachable ({str(e)[:100]})")
            status = resp.status if hasattr(resp, "status") else resp.code
            self.send_response(status)
            for k, v in resp.headers.items():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            ok_stream = stream and status == 200
            if ok_stream:
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for line in resp:
                        raw += line
                        self.wfile.write(line)
                        self.wfile.flush()
                except OSError:
                    pass
            else:
                raw = resp.read()
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except OSError:
                    pass
            i, c, o, model = usage_of(raw, ok_stream) if status == 200 else (0, 0, 0, None)
            row = {"t": now(), "label": label, "path": path, "model": model, "status": status, "stream": ok_stream,
                   "in": i, "cached": c, "out": o, "cost": round(i * M.price[0] + c * M.price[1] + o * M.price[2], 8),
                   "ms": int(1000 * (time.time() - t0))}
            if status != 200:
                row["error"] = raw[:200].decode("utf-8", "replace")
            M.record(row)
    return H


def report(path, since=None):
    rows = [r for r in read_ledger(path) if not since or r.get("t", "") >= since]
    if not rows:
        print("no requests")
        return
    def table(key):
        agg = {}
        for r in rows:
            a = agg.setdefault(r.get(key) or "-", [0, 0, 0, 0, 0.0, 0])
            a[0] += 1
            a[1] += r.get("in", 0)
            a[2] += r.get("cached", 0)
            a[3] += r.get("out", 0)
            a[4] += r.get("cost", 0)
            a[5] += r.get("status") != 200
        print(f"{'by ' + key:<52} {'requests':>8} {'errors':>6} {'input':>12} {'cached':>13} {'output':>11} {'cost $':>9}")
        for k, a in sorted(agg.items()):
            print(f"{k[-52:]:<52} {a[0]:>8} {a[5]:>6} {a[1]:>12,} {a[2]:>13,} {a[3]:>11,} {a[4]:>9.4f}")
    table("label")
    table("model")
    print(f"{len(rows)} requests from {rows[0]['t']} to {rows[-1]['t']}: ${sum(r.get('cost', 0) for r in rows):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--upstream", required=True)
    s.add_argument("--key-file", required=True)
    s.add_argument("--price", required=True, help="dollars per million uncached input, cached input, output tokens")
    s.add_argument("--ledger", required=True)
    s.add_argument("--budget", type=float, default=10.0)
    s.add_argument("--port", type=int, default=18800)
    s.add_argument("--bind", default="0.0.0.0")
    r = sub.add_parser("report")
    r.add_argument("ledger")
    r.add_argument("--since")
    a = ap.parse_args()
    if a.cmd == "report":
        return report(a.ledger, a.since)
    M = Meter(a)
    srv = http.server.ThreadingHTTPServer((a.bind, a.port), handler(M))
    print(f"meter on {a.bind}:{a.port} -> {M.upstream}; spent ${M.spent:.4f} of ${M.budget:.2f}; "
          f"client key in {a.ledger}.client-key", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
