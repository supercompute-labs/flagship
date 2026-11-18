#!/usr/bin/env python3
"""Basalt Qwen tuner: restart per config, quick suite, parse engine log for true tg/pp rates."""
import json, os, re, subprocess, sys, time, uuid, urllib.request
B = os.environ.get("BASALT_DIR", os.path.expanduser("~/basalt-0.4.1")); REL = f"{B}/basalt-v0.4.1-linux-x64"
KEY = os.environ["BASALT_API_KEY"]; MODEL = "qwen3.8-flash-next-v31-basalt"
LOG = os.environ.get("BASALT_LOG", "./basalt-qwen.log"); OUT = "./results/tune-ledger.jsonl"
BASE = ["--max-context", "131072", "--kv", "int8", "--kv-resident", "32768", "--vision", "dynamic"]
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

def port_pid():
    o = subprocess.run("ss -ltnp | grep ':8080 '", shell=True, capture_output=True, text=True).stdout
    m = re.search(r"pid=(\d+)", o); return m and m.group(1)

def stop():
    p = port_pid()
    if p: subprocess.run(["kill", p])
    for _ in range(30):
        time.sleep(2)
        if not port_pid() and not subprocess.run("pgrep -f 'bin/basalt --serve'", shell=True, capture_output=True).stdout:
            return
    subprocess.run("pkill -9 -f 'bin/basalt --serve'", shell=True); time.sleep(3)

def start(extra):
    cmd = [f"{REL}/bin/basalt", "serve", f"{B}/Qwen3.8-FN-UD-IQ4_XS.basalt", "--python", f"{B}/.venv/bin/python",
           "--host", "0.0.0.0", "--port", "8080", "--api-key", KEY, "--alias", MODEL] + BASE + extra
    t0 = time.time()
    subprocess.Popen(cmd, cwd=REL, stdout=open(LOG, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    while time.time() - t0 < 600:
        time.sleep(4)
        try:
            h = json.load(urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5))
            if h.get("loaded"): return time.time() - t0
        except Exception: pass
    return None

REC = "alpha beta gamma delta epsilon zeta eta theta."
def prompt(kind):
    n = uuid.uuid4().hex
    if kind == "prose": return f"[{n}] Write a detailed, vivid 600-word short story about a lighthouse keeper who finds a message in a bottle."
    if kind == "code": return f"[{n}] Write a complete Python module implementing an LRU cache class with get/put, type hints, docstrings and unit tests."
    k = int(kind); recs = "\n".join(f"record{i:06d}: {REC}" for i in range(1, int(k / 19.3) + 1))
    return f"[{n}]\n{recs}\nSummarize the structure of these records and explain how to look up a record by its ID."

def req(kind, out):
    body = {"model": MODEL, "max_tokens": out, "messages": [{"role": "user", "content": prompt(kind)}],
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                  json.dumps(body).encode(), HDR), timeout=900))
    return time.time() - t0, d["usage"]

def done_lines():
    return [l for l in open(LOG, errors="ignore") if " done " in l]

def parse(l):
    g = lambda p: (re.search(p, l) or [None, None])[1]
    return {"pp": int(g(r"pp=(\d+)") or 0), "pp_tps": float(g(r"pp=\d+/\d+c [\d.]+s (\d+)t/s") or 0) or None,
            "tg": int(g(r"tg=(\d+)") or 0), "tg_tps": float(g(r"tg=\d+ ([\d.]+)t/s") or 0), "acc": g(r"acc=(\d+)%"),
            "ttft": float(g(r"ttft=([\d.]+)s") or 0), "hit": g(r"hit=([\d.]+)%"), "ssd": g(r"ssd=([\d.]+)%")}

SUITE = [("warm", "prose", 64), ("prose", "prose", 512), ("code", "code", 512), ("prose2", "prose", 512),
         ("p16k", "16384", 256), ("p64k", "65536", 256)]

def bench(name, extra):
    stop(); load = start(extra)
    rec = {"name": name, "extra": extra, "load_s": load and round(load, 1), "t": time.strftime("%H:%M:%S")}
    if load is None:
        rec["error"] = "failed to load: " + "".join(open(LOG, errors="ignore").readlines()[-5:])[-600:]
    else:
        rec["startup"] = [l.strip()[:200] for l in open(LOG, errors="ignore") if re.search(r"warning|ram |auto |kv ", l)][:8]
        res = {}
        for cid, kind, out in SUITE:
            n0 = len(done_lines())
            try:
                wall, u = req(kind, out); time.sleep(0.5)
                ln = done_lines()[n0:]
                r = parse(ln[-1]) if ln else {}
                r.update(wall=round(wall, 2), prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"))
                res[cid] = r
            except Exception as e:
                res[cid] = {"error": repr(e)[:300]}
        rec["res"] = res
        ok = [res[k]["tg_tps"] for k in ("prose", "code", "prose2") if res.get(k, {}).get("tg_tps")]
        rec["decode_short"] = round(sorted(ok)[len(ok) // 2], 1) if ok else None
        rec["ttft16k"] = res.get("p16k", {}).get("ttft"); rec["ttft64k"] = res.get("p64k", {}).get("ttft")
        rec["dec64k"] = res.get("p64k", {}).get("tg_tps")
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("name", "load_s", "decode_short", "dec64k", "ttft16k", "ttft64k", "error")}), flush=True)

if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    for name, extra in plan: bench(name, extra)
    print("PLAN DONE", flush=True)
