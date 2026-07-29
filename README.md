# Kairos

**Workload-aware and reconfigurable LLM serving, built on vLLM.**


Kairos is a serving system that adapts itself to the workload it observes. While
serving requests with a high-quality base model, it samples the live request
stream and evaluates cheaper candidates — quantized variants and smaller
independent models — on those sampled requests in the background. When a
candidate satisfies the configured accuracy and latency objectives (SLOs), Kairos
switches the active model at runtime and reorganizes where model weights reside
across GPU memory, CPU memory, and disk. Disruptive actions are deferred to
forecasted idle windows in the request stream, so adaptation interferes as little
as possible with live serving.

The result: the system discovers on its own, from live traffic, that a cheaper
model is good enough — and moves to it without stopping.

📄 The full design and evaluation are documented in the accompanying Master's
thesis: [**Workload-aware and Reconfigurable LLM Serving** (PDF)](docs/thesis.pdf).

---

## Why

Static LLM deployments pick one model and one placement before the first request
arrives. But the best configuration depends on the workload: a quantized or smaller
model can cut p95 latency substantially while still matching the base model's
answers — whether it does depends on the task, the quantization scheme, and the
hardware, and can only be measured at runtime. Engines like vLLM make a *given*
configuration fast; Kairos decides *when to change* the configuration.

## Design principles

| | Principle | Realized by |
|---|---|---|
| P1 | **Continuous monitoring** — measure candidate models on sampled live requests, with the base model's outputs as accuracy reference (no labels needed) | Monitoring component |
| P2 | **Runtime reconfigurability** — switch the active model and move weights across GPU / CPU / disk while serving | Reconfiguration + vLLM extensions |
| P3 | **Interference-aware scheduling** — delay disruptive commands until a forecasted idle window | Scheduler (Holt-Winters et al.) |

## Architecture

![Kairos architecture](⚠️ VERIFY: export thesis Fig. 3.2 to docs/architecture.png)

Two processes connected by a ZeroMQ IPC layer:

- **KairosAPI** — HTTP entry point (`/infer`); validates requests, records arrival timestamps.
- **KairosCore** — orchestration:
  - **Controller** — admits requests, dispatches them *concurrently* to the active
    vLLM server (one async task per request, exposing overlap for continuous
    batching), executes control commands, runs candidate evaluations.
  - **Monitoring** (P1) — samples requests into a rolling window, freezes evaluation
    snapshots so all models are compared on identical inputs, computes per-model
    SLIs (accuracy vs. base, p95 latency). Discard / recycle / monotonicity
    policies prune the candidate set.
  - **Model catalog** — shared registry: metadata, statistics, current placement.
  - **Reconfiguration** (P2) — scores models (feasible models ranked by weighted
    SLO margin, infeasible by violation), computes target placement via 0/1
    knapsack per memory tier, plans the command sequence with BFS over valid
    system states, schedules evaluations.
  - **Scheduler** (P3) — forecasts request volume over a window (moving average,
    EWMA, Holt, Holt-Winters); releases interfering commands only when the
    predicted volume is below an idle threshold.
- **vLLM layer** — one server per registered model, run in Docker.

## vLLM extensions

Kairos runs on a modified vLLM (⚠️ VERIFY: state how it ships — fork link,
submodule, or Docker image). vLLM's existing sleep/wake mechanisms (L1/L2 sleep,
wake-up from CPU) are extended with:

- **`WAKE_UP_FROM_DISK`** — reallocate GPU memory, reload weights via RPC, reset prefix cache.
- **`PREFETCH` / `WAKE_UP_FROM_PREFETCH`** — stage weights from disk into pinned CPU
  memory *ahead of time*, shrinking the request-critical wake-up to a CPU→GPU copy
  (8B model: 3.5 s disk wake-up → 1.45 s critical path).
- **`WAKE_UP_PERSISTENT`** — keep the CPU copy after wake-up, making the next L1
  sleep a near-free GPU release (8B model: 1.36 s → 0.28 s).
- **`EVICT`** — drop CPU-resident weights directly (~1 ms), avoiding the naive
  wake-then-L2-sleep workaround that transiently needs the model's full GPU
  footprint (21+ GiB for an 8B model).

Transitions are conditioned (e.g. wake-up-from-prefetch requires a prior prefetch);
the reconfiguration planner searches only legal sequences. Quantized models cannot
be restored from disk (runtime weight layout ≠ checkpoint) and move only between
GPU and CPU.

## Results (summary)

Single RTX A5000 (24 GB), four QA datasets (BoolQ, LogiQA, MMLU, OpenBookQA),
synthetic workload client (Poisson, ON/OFF, periodic, bursty, ramp, step):

- **INT8 (W8A8) and INT4 (W4A16) variants and smaller dense models** (Qwen3-4B)
  preserve accuracy near the base model while cutting p95 latency; NF4 degrades on
  reasoning tasks; accuracy loss is task- and scheme-dependent — the case for
  runtime measurement.
- **Holt-Winters** achieves the best idle-window prediction (F1 0.81, FPR 0.11) on
  ON/OFF traffic.
- **End-to-end**, Kairos switches the active model to an INT8 variant at runtime,
  reducing request latency while both SLOs hold; remaining latency spikes trace
  directly to false-positive idle predictions — on a single GPU, reconfiguration
  and inference share the accelerator.

Full evaluation in the thesis (Ch. 4), incl. sleep/wake/prefetch/evict
microbenchmarks across model sizes and families.

## Quickstart

⚠️ VERIFY this entire section against the real repo — commands below are a template.

```bash
git clone <repo-url> && cd kairos
pip install -e .                       # ⚠️ or requirements.txt
# prerequisites: NVIDIA GPU + driver, Docker + NVIDIA Container Toolkit,
# HF_TOKEN for gated models

# 1. configure models + SLOs
$EDITOR configs/config.yaml            # ⚠️ real path

# 2. start KairosCore and KairosAPI
<real command>                         # ⚠️
<real command>                         # ⚠️

# 3. send traffic (single request or workload client)
curl -X POST localhost:8000/infer -H 'Content-Type: application/json' -d '{
  "instruction": "Answer the question using the given context.",
  "prompt": "Context: ... Question: ...",
  "answer": "yes"
}'
<real client command, e.g. python -m client --config client.yaml>   # ⚠️
```

### Configuration

```yaml
models:
  - model_id: meta-llama/Llama-3.1-8B-Instruct
    relation: base          # accuracy ground truth
    port: 8001
    gpu_factor: 1.3         # weights × factor ≈ footprint incl. KV cache
  - model_id: RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8
    relation: quantized
    port: 8002
    gpu_factor: 1.3
  - model_id: Qwen/Qwen3-1.7B
    relation: independent
    port: 8003
    gpu_factor: 1.3
slo:
  accuracy: 0.95            # ≥ 95% agreement with base model
  latency: 200              # p95 ≤ 200 ms
  weight_acc: 0.7
  weight_lat: 0.3
monitoring:
  sample_rate: 10           # sample every 10th request
  sample_size: 50           # rolling window / snapshot size
  monotonicity: true        # INT8 fails ⇒ skip INT4
  discard: 3                # drop candidate after 3 failed rounds
  recycle: 3                # readmit after 3 rounds
scheduler:
  method: holt              # moving_avg | ewma | holt | holt_winters
  window: 10                # forecast horizon (s)
```

## Repository layout

⚠️ VERIFY — replace with the actual tree.

## Status

Research prototype from a Master's thesis; built for controlled single-node
experiments, not production serving.

## Citation

```bibtex
@mastersthesis{enjilian2026kairos,
  title  = {Workload-aware and Reconfigurable {LLM} Serving},
  author = {Enjilian, Ren{\'e} Richard},
  school = {Technische Universit{\"a}t Berlin},
  year   = {2026},
  month  = jun
}
```