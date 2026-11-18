# Methodology: how we found the fast settings

This is the process behind recipe 001. It took Qwen3.8-Flash-Next on one RTX 5080 from **~47–57 tok/s to ~149 tok/s** decode at 64k context, with the quality gate passed. The process is engine-agnostic. Only the knobs change from engine to engine.

---

## 0. Know the machine before touching flags

Before tuning, answer two questions: *where do the weights live*, and *what limits each phase?*

| Item | Recipe 001 |
|---|---|
| Model | 91 GB MoE (mixture of experts). Only a few experts are active per token. |
| GPU | 16 GB VRAM: the dense layers, the KV cache, and a cache of the hottest experts |
| RAM | 60 GB: most of the remaining experts |
| SSD | about 7.8 GB of experts that fit nowhere else |

We confirmed the following from engine logs during real requests:
- **SSD reads were ~1% of expert fetches.** The SSD was *not* the bottleneck, despite the model being bigger than VRAM + RAM.
- **The GPU expert cache hit ~30–40%.** Most expert work runs from system RAM on the CPU side.
- **Decode is therefore bound by RAM and CPU,** and prefill by GPU compute plus transfer.

This one diagnosis predicted the whole outcome. Flags that touch SSD or caching barely move decode. Anything that keeps more of the work on the GPU moves it a lot.

> **Lesson:** measure where the time goes before changing anything. Most "tuning" fails because it optimizes a phase that isn't the bottleneck.

---

## 1. Build a benchmark you can trust

Our first harness reported prefill numbers that were too good, because the engine was reusing a cached prompt prefix. We threw those results away and rebuilt the harness (`tools/bench.py`) with these rules:

1. **Cold prompts.** Every request starts with a unique nonce (`[run <uuid>]`), so no prefix cache can be reused. Every number is a true cold prefill.
2. **Server-side token counts.** Tokens come from the server's `usage` field, never from counting stream chunks, since one chunk can carry several tokens.
3. **Separate the phases.**
   - Time to first token (TTFT) measures prefill.
   - Decode rate = `(completion_tokens − 1) / (t_last − t_first)`.
   - Engine-reported rates are parsed from the server log when available (`tools/tune.py`).
4. **A depth sweep, not one number.** Context depths are 0k, 4k, 16k, 64k and 120k. Engines that win at short context often lose at long context.
5. **Same suite for every engine.** Strata and Basalt ran the identical script, with identical prompts and output lengths.

---

## 2. Establish the baseline and the noise floor

Run the stock config several times *before* tuning. On this machine, identical runs varied **±5–10%**. 16k TTFT bounced between ~5 and ~8 s with no config change at all.

> **Rule:** a change smaller than the noise floor is not a result. Without this step, half of what looks like tuning is measuring randomness.

Baselines:

| Engine | Decode, short (tok/s) | Decode @ 64k (tok/s) | TTFT @ 64k |
|---|---|---|---|
| Strata (previously tuned) | ~55 | ~47 | 33.6 s |
| Basalt 0.4.1 defaults | ~61 | ~57 | 19.5 s |

---

## 3. One change at a time, with interleaved controls

`tools/tune.py` takes a *plan*: an ordered list of configs. For every entry it:

1. **fully stops and restarts** the server with that config, so no state carries over;
2. waits for `/health` to report the model is loaded, and records the load time;
3. runs a fixed quick suite (short prose, short code, 16k and 64k cold prefill), each with fresh nonces;
4. parses the engine's own per-request log lines for true prefill and decode rates;
5. appends everything to `results/tune-ledger.jsonl`, failures included.

**Controls:** the baseline config reappears every 2–3 entries in every plan, e.g. `EG, A, B, EG, C, D, EG`. That catches drift from thermals, background load or memory fragmentation, which would otherwise be credited to whichever config happened to run at the time.

Plans are in [`recipes/qwen38-flashnext-rtx5080-basalt/plans/`](../recipes/qwen38-flashnext-rtx5080-basalt/plans/).

---

## 4. Climb in rounds: lossless first, lossy last

### Round 1: lossless knobs, broad sweep (11 configs)

We tested each knob alone against the base:

| Knob | Result | Verdict |
|---|---|---|
| `--kv-resident 0`: whole KV cache on GPU (int8 makes it ~0.8 GiB) | decode +4%, 64k TTFT −19% | ✅ keep |
| `--prefill-cpu-share 0.4`: less CPU work during prefill | 64k decode up, TTFT −17% | ✅ keep |
| `--spec 6` / `--spec 10` (draft depth) | within noise | ❌ default 8 is fine |
| `--ram-reserve-mib 1024` | within noise | ❌ SSD wasn't the bottleneck |
| larger prefill chunk | slightly worse | ❌ |

### Round 2: combine winners, confirm with repeats (11 configs)

Combining the two winners gave **EG** = `--kv-resident 0 --prefill-cpu-share 0.4`. We ran it **4 times** against base controls:
- EG: decode 63.1–64.5 tok/s, 64k TTFT 16.9–18.7 s.
- Base: ~62 tok/s, 16.1–18.4 s.

Neighbors were tested to check EG is a local optimum: cpu-share 0.2 and 0.3, spec 10, pool workers 6 and 8. None beat EG.

**Result: about +3% decode and about −7% 64k TTFT.** Real, but small. The lossless knobs were exhausted, which matches the step 0 diagnosis: none of them change how much expert work stays on the CPU.

### Round 3: output-changing knobs (10 configs, all on top of EG)

The diagnosis pointed at *routing*. The model picks experts per token, and most picks land on experts in system RAM. Basalt exposes three output-changing levers:

- `--route-prior`: biases expert selection toward experts already on the GPU.
- `--route-tail-skip`: skips the weakest-weighted experts when they aren't on the GPU.
- `--relaxed-drafting`: accepts speculative draft tokens slightly more loosely. Tool-call and JSON spans stay exact.

| Config | Short decode (tok/s) | 64k decode (tok/s) |
|---|---|---|
| EG (×4 controls) | 61–65 | 55–63 |
| relaxed 0.5 | 65.4 | 63.5 |
| relaxed 0.3 | 67.6 | 66.8 |
| route-prior 0.5 | 77.5 | 69.9 |
| relaxed 0.5 + prior 0.5 | 78.6 | 71.3 |
| **prior 1.5 + tail-skip 7** | **127.0** | **122.9** |
| **relaxed 0.3 + prior 1.5 + tail-skip 7** | **136.8** | **149.3** |

Routing roughly **doubled** decode, exactly as the diagnosis predicted. Prefill was unchanged, since routing only affects decode.

---

## 5. Gate before you ship

A lossy config is not "better" until it passes inspection. `tools/gate.py` runs the same checks on the lossless baseline and each candidate:

1. **20 exact-answer questions** (arithmetic, word problems, facts) at temperature 0 with thinking off.
2. **Needle retrieval:** a random 6-digit code hidden at different depths in **32k, 64k, 110k and 110k** token documents.
3. **Prose sample:** a 600-word story, checked for repetition loops and coherence.

| Config | Q&A | Needles | Prose |
|---|---|---|---|
| EG (exact) | 18/20* | 4/4 | clean |
| prior 1.5 + skip 7 | 17/20, **with 1 real error** (1000 − 387 = 513) | 4/4 | clean |
| **relaxed 0.3 + prior 1.5 + skip 7** | 18/20* | 4/4 | clean |

\*The remaining misses were grader artifacts (`$CO_2$` formatting, and one answer truncated at the token limit), so both are effectively 20/20.

**Decision:**
- Ship **`fast`** = EG + relaxed 0.3 + prior 1.5 + tail-skip 7.
- Keep **`exact`** = EG as the output-preserving mode.
- Don't ship the routing-only config. It was the only one that made a real error. (Our hypothesis: the relaxed-drafting pass adds verification that catches some routing drift. This is unproven.)

---

## 6. Deploy with an escape hatch

`launch.sh fast|exact` switches modes with one command. Both modes use the same port, key, model alias and vision settings, so clients never notice. If anything looks off in real use, switch to `exact`.

---

## What we'd do better next time (and will, in Flagship v0.2)

- **More samples:** 6+ repeats per finalist, reporting medians and spread instead of 1–3 samples.
- **A bigger gate:** a 200-question GSM8K subset, tool-call/JSON exactness, and multi-turn checks.
- **A ceiling estimate up front:** compute the theoretical decode limit from memory bandwidth and the bytes read per token, so you know when to stop.
- **Concurrency and long-session stability:** test multiple users at once, and a sustained full-context run.
- **Automation:** let the tuner propose the next config from the ledger instead of a hand-written plan.

## Summary

1. **Diagnose** where the time goes. Here it was RAM and CPU experts, not SSD.
2. **Measure honestly:** cold prompts, server token counts, a depth sweep.
3. **Know your noise floor:** ±5–10%.
4. **Change one thing at a time,** with interleaved controls.
5. **Exhaust lossless knobs first.** They gave +3% here.
6. **Pull the lever the diagnosis points at.** Routing gave ~2.5×.
7. **Gate lossy wins on accuracy, long context and prose.**
8. **Ship two modes,** fast and exact, with a one-command switch.
