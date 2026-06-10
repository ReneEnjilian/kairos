# Kairos

<p align="center">
  <b>Workload-aware reconfigurable LLM serving</b>
</p>

<p align="center">
  <i>Continuous monitoring, adaptive model selection, and memory-aware runtime reconfiguration for LLM inference.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20prototype-blue" />
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/backend-vLLM-green" />
  <img src="https://img.shields.io/badge/platform-NVIDIA%20GPU-76B900" />
</p>

---

## Overview

Kairos is a research system for adaptive large language model serving.

Instead of serving all requests with a fixed model configuration, Kairos continuously observes the incoming workload, evaluates alternative models on sampled requests, and reconfigures the serving setup at runtime.

The system is designed around one central question:

> Which model should serve the current workload, and where should all other models be placed to enable fast adaptation?

Kairos combines online monitoring, background evaluation, model ranking, memory-aware placement, and runtime control over vLLM model servers. It can switch the active model, keep alternative models warm or prefetched, and move models between GPU, CPU, and disk depending on current workload behavior and service objectives.

---

## Motivation

LLM serving workloads are not static.

A high-quality base model may be the best choice when latency pressure is low.  
A quantized model may satisfy the same accuracy objective while reducing latency.  
A smaller independent model may be sufficient for easier workloads.  
Under changing request rates and workload distributions, the best serving configuration can change over time.

Kairos treats this as a systems problem.

Rather than assuming a single optimal model, Kairos adapts the serving configuration during runtime. It monitors real requests, evaluates candidate models on representative samples, and selects the model that best satisfies the configured accuracy and latency objectives.

---

## Key Features

### Workload-Aware Monitoring

Kairos samples live inference requests and stores representative evaluation snapshots. Candidate models are evaluated on requests that actually occur during serving, rather than only on static offline benchmarks.

### Adaptive Model Selection

Kairos compares base, quantized, and independent models using configurable service objectives. Models are ranked according to latency, relative accuracy, and SLO satisfaction.

### Runtime Reconfiguration

Kairos can change the active serving model during runtime. It can also adjust the placement of non-active models to prepare for future workload changes.

### Memory-Aware Placement

Kairos explicitly models GPU, CPU, and disk placement. It accounts for the full GPU memory required by an awake model, the reduced memory footprint of sleeping models, and the additional memory required to wake a model.

### Interference-Aware Evaluation

Background evaluation is not free. Kairos controls when and how candidate models are evaluated to reduce interference with active serving.

### vLLM-Based Execution

Kairos uses vLLM as the inference backend and builds a separate control plane around multiple model servers.

---

## System Components

Kairos consists of several cooperating components.

### API Layer

Receives inference requests from clients and forwards them to the Kairos core.

### Dispatcher

Routes incoming requests to the currently active model server.

### Monitor

Samples requests and records model outputs, latency measurements, labels, and workload metadata.

### Scheduler

Decides when background evaluations should be executed based on workload behavior and predicted idle windows.

### Controller

Executes runtime actions such as evaluating models, waking models, sleeping models, and changing the active serving model.

### Reconfiguration Engine

Ranks candidate models, computes target placements, and generates the command sequence required to transition from the current state to the target state.

### Memory Manager

Tracks available GPU, CPU, and disk capacity and validates whether model movement operations are feasible.

---

## Model States

Kairos manages models across multiple runtime states.

| State | Description |
|---|---|
| Active on GPU | The model currently serving incoming requests |
| Awake on GPU | The model is fully loaded and ready for inference |
| Sleeping on GPU | The model keeps a reduced GPU memory footprint |
| Prefetched in CPU memory | The model weights are available in RAM |
| Evicted to disk | The model is stored on disk and must be loaded before use |

This allows Kairos to trade off memory usage, reconfiguration latency, and serving readiness.

---

## Model Types

Kairos supports different model relations.

| Type | Description |
|---|---|
| Base model | High-quality reference model used as the main quality baseline |
| Quantized model | Variant of the base model with reduced precision and lower serving cost |
| Independent model | Separate model from another family or size class |

This enables Kairos to compare quality-performance trade-offs between multiple serving alternatives.

---

## Example Configuration

Kairos is configured through YAML files.

```yaml
models:
  - model_id: llama-8b-bf16
    name: meta-llama/Llama-3.1-8B-Instruct
    relation: base
    port: 8100
    gpu_factor: 1.30

  - model_id: llama-8b-fp8
    name: meta-llama/Llama-3.1-8B-Instruct-FP8
    relation: quantized
    base_model: llama-8b-bf16
    port: 8101
    gpu_factor: 0.80

  - model_id: qwen-4b
    name: Qwen/Qwen3-4B-Instruct
    relation: independent
    port: 8102
    gpu_factor: 1.30

objectives:
  accuracy_slo: 0.95
  latency_slo_ms: 200

  weights:
    accuracy: 0.7
    latency: 0.3

monitoring:
  sample_rate: 10
  sample_size: 100

scheduling:
  forecast_horizon_seconds: 20
  max_in_flight_evaluation_requests: 4
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/<your-org>/kairos.git
cd kairos
```

Create a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Kairos.

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Requirements

Kairos is designed for Linux-based GPU systems.

Recommended environment:

- Python 3.10+
- Docker
- NVIDIA GPU
- NVIDIA Container Toolkit
- CUDA-compatible NVIDIA driver
- vLLM-compatible model checkpoints
- Hugging Face access token for gated models

---

## Running Kairos

Start the Kairos core.

```bash
python -m kairos.core --config configs/config.yaml
```

Start the API server.

```bash
uvicorn kairos.api.main:app --host 0.0.0.0 --port 8000
```

Send an inference request.

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Answer the following question.",
    "prompt": "Is the sky blue during a clear day?",
    "answer": "yes",
    "dataset": "boolq",
    "arrival_timestamp": 1710000000.0
  }'
```

Example response:

```json
{
  "kairos": {
    "active_model": "llama-8b-bf16",
    "correct": true,
    "infer_latency_ms": 112.4
  },
  "result": "yes"
}
```

---

## Experiments

Kairos includes experiments for studying adaptive LLM serving behavior.

### Model Movement

Measures the cost of moving models between runtime states.

Typical measurements include:

- Sleep latency
- Wake-up latency
- Prefetch latency
- Eviction latency
- Awake GPU memory
- Sleeping GPU memory
- Transient memory requirements

### Model Selection Trade-offs

Compares base, quantized, and independent models under different workloads.

Typical metrics include:

- Relative accuracy compared to the base model
- Median latency
- p95 latency
- SLO satisfaction
- Throughput under different request rates

### End-to-End Reconfiguration

Evaluates whether Kairos can adapt the serving setup when workload conditions change.

Typical questions include:

- Does Kairos select a better model when the workload changes?
- How long does reconfiguration take?
- Which commands dominate reconfiguration latency?
- How much does reconfiguration interfere with active serving?
- Does the system continue to satisfy the configured objectives?

---

## Repository Structure

```text
kairos/
├── api/                     # API server and request handling
├── core/
│   ├── catalog/             # Model metadata and model registry
│   ├── control/             # Runtime controller and control commands
│   ├── memory/              # GPU, CPU, and disk memory management
│   ├── monitoring/          # Online sampling and evaluation snapshots
│   ├── reconfiguration/     # Ranking, placement, and transition planning
│   └── scheduling/          # Background evaluation scheduling
├── experiments/             # Experiment scripts and plotting utilities
├── configs/                 # Example configuration files
├── scripts/                 # Helper scripts
└── README.md
```

---

## Design Principles

Kairos follows three design principles.

### Continuous Monitoring

Serving decisions should be based on the workload currently observed by the system. Kairos therefore samples live requests and evaluates models on representative runtime data.

### Runtime Reconfigurability

The serving configuration should not be fixed after deployment. Kairos can switch the active model and adjust model placement while the system is running.

### Interference-Aware Scheduling

Model evaluation and model movement can affect active serving. Kairos therefore treats background work as a controlled systems operation rather than a free background task.

---

## Research Scope

Kairos focuses on single-node adaptive LLM serving with multiple vLLM model servers and a separate control plane.

The current system investigates:

- Runtime model selection
- Quantized and independent model alternatives
- Online evaluation on sampled requests
- Memory-aware model placement
- vLLM sleep, wake-up, prefetch, and eviction mechanisms
- Reconfiguration under latency and accuracy objectives

---

## Status

Kairos is research software developed as part of a master’s thesis on adaptive LLM serving systems.

The implementation is intended for controlled experiments and systems research. It is not yet a production serving framework.

---

## Citation

```bibtex
@mastersthesis{enjilian2026kairos,
  title  = {Kairos: Workload-Aware and Reconfigurable LLM Serving},
  author = {Rene Enjilian},
  school = {Technical University Berlin},
  year   = {2026}
}
```

---

## Acknowledgements

Kairos builds on vLLM and the broader open-source ecosystem for efficient large language model inference.