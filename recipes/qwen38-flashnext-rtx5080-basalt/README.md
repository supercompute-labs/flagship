# Recipe 001: Qwen3.8-Flash-Next on one RTX 5080, 47 → 149 tok/s

**Flagship · Recipe 001** · full process: [docs/METHODOLOGY.md](../../docs/METHODOLOGY.md)

This sheet covers Qwen3.8-Flash-Next (Unsloth UD-IQ4_XS) on a single 16 GB consumer GPU, served by [Basalt v0.4.1](https://github.com/jesdga95/basalt/releases/tag/v0.4.1) and tuned with a measured A/B loop. The model does not fit in VRAM + RAM, and the tuned config still decodes at **~137 tok/s on short contexts and ~149 tok/s at 64k**.

## Hardware

| Part | Spec |
|---|---|
| GPU | RTX 5080, 16 GB |
| System RAM | 60 GB |
| Storage | NVMe SSD; the experts that don't fit in RAM stream from it |
| Software | CUDA 13.3, Linux, Basalt v0.4.1 (`d3b7574`) |
| Model | Qwen3.8-Flash-Next, Unsloth UD-IQ4_XS, official `.basalt` pack (90.9 GiB) |
| Server context | 131,072 tokens |

## Results

Single user, cold prompts. Every request starts with a unique nonce, so nothing is served from the prompt cache.

| Engine / config | Decode, short (tok/s) | Decode at 64k (tok/s) | TTFT at 64k |
|---|---|---|---|
| Strata (tuned, previous setup) | ~55 | ~47 | ~34 s |
| Basalt defaults | ~61 | ~57 | ~19.5 s |
| Basalt `exact`: lossless tuning | ~63 | ~59 | ~17 s |
| **Basalt `fast`: lossy routing + relaxed drafting** | **~137** | **~149** | **~16 s** |

TTFT by context depth, Strata → Basalt defaults:

| Context | 0k | 4k | 16k | 64k | 120k |
|---|---|---|---|---|---|
| TTFT (s) | 6.4 → 0.6 | 7.9 → 3.9 | 12.8 → 8.3 | 33.6 → 19.8 | 62.9 → 32.2 |
| Prefill (tok/s) | — | 550 → 1,069 | 1,320 → 1,953 | 1,941 → 3,262 | 1,888 → 3,674 |

## The recipe

```bash
export BASALT_API_KEY=...            # your key
export BASALT_DIR=~/basalt-0.4.1     # release + .venv + .basalt pack
./launch.sh fast    # ~137–149 tok/s, slightly lossy
./launch.sh exact   # output-preserving, ~63 tok/s
```

The flags, on top of `--max-context 131072 --kv int8 --vision dynamic`:

| Flag | Mode | Effect measured here |
|---|---|---|
| `--kv-resident 0` | both | Keeps the whole KV cache on the GPU, which is only ~0.8 GiB with int8 on this hybrid model. 64k TTFT −7 to −19%. |
| `--prefill-cpu-share 0.4` | both | Less CPU work during prefill. 64k TTFT improved slightly. |
| `--route-prior 1.5 --route-tail-skip 7` | fast | **About 2× decode.** Biases routing toward experts already on the GPU and skips weak experts that aren't. ⚠️ Lossy. |
| `--relaxed-drafting 0.3` | fast | A further +8–20% on top of routing. Lossy, but tool-call and JSON spans stay exact. |

### Why routing helps so much on this machine

About 7.8 GB of experts don't fit in RAM, and only **~30–40% of expert lookups hit the GPU cache**. Decode is limited by the CPU and RAM side, not by SSD reads, which are ~1%. Steering routing toward experts already on the GPU removes most of that work. On a machine where the whole model fits in VRAM, the gain would be much smaller.

### What didn't help (within ±3% noise)
- `--spec 6` and `--spec 10`. The default draft depth of 8 is fine.
- `--ram-reserve-mib 1024` and `--prompt-cache 2`: SSD reads were already ~1%.
- `--prefill auto:32768`: worse.
- `--pool-workers 6` and `8`. The default of 7 is fine.
- `--prefill-cpu-share 0.2` and `0.3`.
- `--route-prior 0.5` alone and with relaxed drafting 0.5: only +25%.

## Quality gate

Compared at temperature 0, thinking off:

| Config | 20 Q&A (math + facts) | Needle retrieval at 32k / 64k / 110k / 110k | Prose sample |
|---|---|---|---|
| exact | 18/20* | 4/4 | clean |
| route-prior 1.5 + tail-skip 7 | 17/20 (**1 real arithmetic error**) | 4/4 | clean |
| **fast** (+ relaxed 0.3) | 18/20* | 4/4 | clean |

\*The misses were all grading artifacts: a `$CO_2$` formatting mismatch, and an answer truncated by the 600-token limit on prime counting. Corrected, `exact` and `fast` both score 20/20.

**Caveats.** The quality gate is small (20 questions and 4 needles). `fast` is lossy by design. The routing-only config made one real arithmetic error (1000 − 387 = 513). For work that can't tolerate any change in output, use `exact`.

## Method

1. **Cold prompts only.** A unique nonce in every prompt prevents prompt-cache reuse. An earlier harness without nonces measured cached prefill, and its results were discarded.
2. **One change at a time.** Each candidate config gets a full server restart and runs with unchanged baseline controls interleaved every 2–3 configs.
3. **True decode numbers.** Decode is parsed from the engine's per-request log line, and token counts come from server `usage`, not from stream chunks.
4. **Every run is logged.** Failures are included, in [`results/tune-ledger.jsonl`](results/tune-ledger.jsonl).
5. **Quality gate before deploy.** Lossy configs had to pass Q&A, needle retrieval and a prose check (`tools/gate.py`).

The approach follows the methodology described in [sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill): ceilings first, interleaved controls and a long-context gate. That skill itself was not used for these runs.

### Limitations
- Samples are mostly 1–3 per config. Run-to-run noise is about ±5%, and 16k TTFT varies between ~5 and ~8 s regardless of config.
- No concurrency test and no theoretical-ceiling analysis yet.
- The results depend heavily on how much of the model sits outside VRAM.

## Files

```
launch.sh                fast|exact launcher
../../tools/bench.py       cold-prefill/decode sweep, 0k→120k
../../tools/tune.py            restart-per-config A/B tuner
../../tools/gate.py           Q&A + needle + prose quality gate
plans/*.json             the exact config plans that were run
results/                 raw ledgers and JSON
```

## License

MIT. Model weights and Basalt are under their own licenses.
