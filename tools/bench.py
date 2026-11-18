#!/usr/bin/env python3
"""NA cross-engine bench v2. One streaming request per measurement; unique nonce at the
start of every prompt so no prefix cache is reused (cold prefill each time).
Client-side, engine-agnostic: TTFT, prefill tok/s = prompt_tokens/TTFT,
decode tok/s = (completion_tokens-1)/(t_last-t_first). Engine timings kept if sent.
Usage: bench.py LABEL OUT.json BASE_URL MODEL API_KEY
"""
import json, sys, time, uuid, urllib.request

LABEL, OUT, BASE, MODEL, KEY = sys.argv[1:6]
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
REC = "alpha beta gamma delta epsilon zeta eta theta."
TOK_PER_REC = 19.3  # measured: 273 records -> 5266 tokens

def records(n):
    return "\n".join(f"record{i:06d}: {REC}" for i in range(1, n + 1))

def make_body(cid, target, out_tokens):
    nonce = uuid.uuid4().hex
    if target == 0:
        p = "Explain how a hash table finds a value, including collisions and one small example."
    else:
        p = records(int(target / TOK_PER_REC)) + \
            "\nSummarize the structure of these records and explain how to look up a record by its ID."
    return {"model": MODEL, "max_tokens": out_tokens, "temperature": 0.7, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[run {nonce}] [case {cid}]\n{p}"}]}

SUITE = [("ctx-0k", 0, 256), ("ctx-4k", 4096, 256), ("ctx-16k", 16384, 256),
         ("ctx-64k", 65536, 256), ("ctx-120k", 120000, 256)]
REPS = 2

def run(name, target, out_tokens, rep):
    body = make_body(name, target, out_tokens)
    t0 = time.monotonic()
    req = urllib.request.Request(f"{BASE}/chat/completions", json.dumps(body).encode(), HDR)
    t_first = t_last = None; chunks = 0; usage = timings = None; finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"): usage = ev["usage"]
            if ev.get("timings"): timings = ev["timings"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.monotonic()
                    if t_first is None: t_first = now
                    t_last = now; chunks += 1
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    wall = time.monotonic() - t0
    pt = (usage or {}).get("prompt_tokens") or (timings or {}).get("prompt_n")
    ct = (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n") or chunks
    ttft = (t_first - t0) if t_first else None
    dec_s = (t_last - t_first) if t_first and t_last and t_last > t_first else None
    return {"case": name, "rep": rep, "wall_s": round(wall, 3),
            "ttft_s": round(ttft, 3) if ttft else None,
            "prompt_tokens": pt, "completion_tokens": ct, "chunks": chunks,
            "prefill_tps": round(pt / ttft, 1) if pt and ttft else None,
            "decode_tps": round((ct - 1) / dec_s, 1) if dec_s and ct and ct > 1 else None,
            "engine_prompt_tps": (timings or {}).get("prompt_per_second"),
            "engine_decode_tps": (timings or {}).get("predicted_per_second"),
            "engine_cached": (timings or {}).get("cache_n"), "finish": finish}

results = {"label": LABEL, "model": MODEL, "base": BASE, "suite": "na-v2",
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "requests": []}
run("warmup", 0, 32, -1)
for rep in range(REPS):
    for name, target, out in SUITE:
        print(f"  rep{rep} {name} ...", flush=True)
        try:
            results["requests"].append(run(name, target, out, rep))
        except Exception as e:
            results["requests"].append({"case": name, "rep": rep, "error": repr(e)})
        print("    ", results["requests"][-1], flush=True)
        with open(OUT, "w") as f: json.dump(results, f, indent=1)
results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open(OUT, "w") as f: json.dump(results, f, indent=1)
print("saved", OUT)
