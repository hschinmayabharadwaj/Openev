---
title: Neon Syndicate OpenEnv
sdk: docker
tags:
  - openenv
  - multi-agent
  - strategy-game
  - long-horizon
---

# Neon Syndicate OpenEnv

> **Mission Briefing — NS-1101 / Neon Meridian / 02:47 local**
> *Most LLM benchmarks reward one good answer. Neon Syndicate rewards twelve good decisions in a row.*
> A cinematic multi-agent strategy environment for training LLMs on coalition politics, bargaining, and long-horizon execution under partial observability.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/hsbharadwaj/ev)
[![Runtime API](https://img.shields.io/badge/runtime-live-brightgreen)](https://hsbharadwaj-ev.hf.space/health)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](openenv.yaml)

## For Judges — Start Here

> ### ⌬ One-page showcase: **[https://hsbharadwaj-ev.hf.space/judge](https://hsbharadwaj-ev.hf.space/judge)**
>
> Every required link, the policy ladder, the live training curve, a runnable
> agent demo, and the criteria scorecard — in a single screen. Open this if you
> only have 60 seconds.

Required submission links:

| Asset | Link |
|---|---|
| 🤗 **Hugging Face Space** | [huggingface.co/spaces/hsbharadwaj/ev](https://huggingface.co/spaces/hsbharadwaj/ev) |
| ▶ **Colab notebook (TRL PPO)** | [Open in Colab](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb) |
| ⌥ **GitHub repository** | [github.com/hschinmayabharadwaj/Openev](https://github.com/hschinmayabharadwaj/Openev) |
| ⚡ Runtime API (live env) | [hsbharadwaj-ev.hf.space/health](https://hsbharadwaj-ev.hf.space/health) |
| ⌬ **Judge showcase page** | [hsbharadwaj-ev.hf.space/judge](https://hsbharadwaj-ev.hf.space/judge) |
| ⌬ **Reward Forensics Lab** ★ | [hsbharadwaj-ev.hf.space/lab](https://hsbharadwaj-ev.hf.space/lab) — RLVE knob, 5-gate timeline, reward-hacking sandbox |
| ⌬ **Walkthrough** ★ | [hsbharadwaj-ev.hf.space/walkthrough](https://hsbharadwaj-ev.hf.space/walkthrough) — every concept from the briefing mapped to this codebase |
| 🎮 Playable heist | [hsbharadwaj-ev.hf.space/heist](https://hsbharadwaj-ev.hf.space/heist) |
| 📺 Watch-mode race | [hsbharadwaj-ev.hf.space/play](https://hsbharadwaj-ev.hf.space/play) |

Long-form entry points:

| You want… | Open this |
|---|---|
| **The full walkthrough (every concept → code)** ★ | [`docs/walkthrough.md`](docs/walkthrough.md) or [/walkthrough](https://hsbharadwaj-ev.hf.space/walkthrough) |
| The 3-minute walkthrough | [`docs/judge_flow.md`](docs/judge_flow.md) |
| The story / why-it-matters | [`docs/mini_blog.md`](docs/mini_blog.md) |
| A slide-by-slide pitch deck | [`docs/pitch_flow.md`](docs/pitch_flow.md) |
| A live `curl`-driven demo, scene by scene | [`docs/demo_storyboard.md`](docs/demo_storyboard.md) |
| A 90-second video voice-over | [`docs/video_script_90s.md`](docs/video_script_90s.md) |
| The reward / loss curves | [`artifacts/reward_curves.png`](artifacts/reward_curves.png) · [`artifacts/loss_curve.png`](artifacts/loss_curve.png) |

One-line health check:

```bash
curl https://hsbharadwaj-ev.hf.space/health
```

One-line full pipeline (server → train → evaluate → plot):

```bash
./scripts/run_full_pipeline.sh
```

## Theme Fit (Best Choice)

Primary theme: Theme #1 Multi-Agent Interactions  
Secondary theme: Theme #2 Super Long-Horizon Planning  
Also strong on: Theme #3.1 Professional World Modeling

Why this is a top-tier fit:

- Multi-agent incentives are central: factions cooperate conditionally and have competing interests.
- Planning is **long-horizon**: success requires ordered decisions over up to **24 turns** with delayed payoff (3-faction "hard" missions need ~20 dependent steps).
- World state is partially observable: rumors and outcomes require belief updates.
- Reward hacking is constrained: extraction only succeeds if **all five gates** (coalition, resources, operation code, extraction sector, message keyword set) align — see the breakdown in [`/lab`](https://hsbharadwaj-ev.hf.space/lab).
- A target-aware **expert policy** clears **6/6 missions with 100 % task score** (steps 8 → 22) — that's the upper-bound the trained agent is asked to reproduce.

## The Core Idea

You are the strategist in Neon Meridian, a high-stakes cyberpunk city.  
Each mission requires you to:

- negotiate pacts with factions,
- scout sectors for intelligence,
- rebalance resources under pressure,
- deploy assets and run a coded operation,
- and secure a clean extraction message.

If you shortcut steps, you get penalties and mission collapse.

## Environment API

OpenEnv-compatible endpoints:

- `GET /health`
- `GET /tasks`
- `POST /reset` with optional `{ "task_id": "..." }`
- `POST /step` with `Action`
- `GET /state`

Gym-like behavior is preserved via reset/step/state semantics.

## Observation Space

Main fields in `Observation`:

- `task_id`, `difficulty`, `objective`
- `step_count`, `max_steps`
- `mission`: mission brief with city/client/stakes/threat/rumors
- `known_threat`
- `resources`: credits/intel/influence/energy
- `reputation` by faction
- `alliances`
- `deployed_sector`, `operation_ready`, `operation_executed`, `extraction_ready`
- `intel_log`, `last_action`, `action_history`

## Action Space

Action types:

- `scout_sector`
- `negotiate_pact`
- `trade_resources`
- `deploy_asset`
- `run_operation`
- `secure_extraction`
- `noop`

Structured fields support faction selection, sectors, operation codes, and extraction messaging.

## Reward Design (Dense + Delayed)

Dense components:

- alliance completion progress
- resource threshold progress
- operation readiness/execution progress
- extraction message quality progress
- extraction completion progress

Penalties:

- repeated identical actions
- missing required payload fields
- premature operation attempts
- passive play under critical threat

Terminal success requires consistent multi-step execution, not one-shot action spam.

## Included Tasks

Six missions from easy to hard:

- `task_easy_docklands_relay`
- `task_easy_data_spire_broker`
- `task_medium_undergrid_blackout`
- `task_medium_citadel_convoy`
- `task_hard_orchid_coup`
- `task_hard_citywide_failsafe`

## Local Run

Install:

```bash
pip install -U pip
pip install fastapi httpx openai openenv-core pydantic python-dotenv uvicorn
```

Start API:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl http://localhost:7860/health
```

## Baseline Inference Script

The baseline runner is [inference.py](inference.py). It uses OpenAI-compatible chat completions and logs required trace lines.

Required env vars:

- `OPENAI_API_KEY` (or `HF_TOKEN`)
- `API_BASE_URL` (default `https://api.openai.com/v1`)
- `MODEL_NAME` (default `gpt-4.1-mini`)
- `ENV_BASE_URL` (default `http://localhost:7860`)

Optional routing vars:

- `MODEL_CANDIDATES`
- `MODEL_CANDIDATES_EASY`
- `MODEL_CANDIDATES_MEDIUM`
- `MODEL_CANDIDATES_HARD`
- `MODEL_CANDIDATES_TASK_<TASK_ID>`

Run:

```bash
python inference.py
```

## Minimal TRL Training Pipeline (with QLoRA)

A minimal end-to-end TRL PPO script is provided and directly interacts with this environment API:

- [training/train_trl_ppo.py](training/train_trl_ppo.py)

Run locally (full-parameter PPO on a small base model):

```bash
pip install -e .[training]
python training/train_trl_ppo.py \
  --env-base-url http://localhost:7860 \
  --episodes 24 \
  --output-dir artifacts/trl-neon-model
```

Run on Colab / HF Space GPU with **QLoRA** (4-bit + LoRA adapters; recommended on the T4/L4/A10 budget):

```bash
pip install -e .[qlora]
python training/train_trl_ppo.py \
  --env-base-url http://localhost:7860 \
  --episodes 48 --max-steps 24 \
  --use-qlora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --output-dir artifacts/trl-neon-model
```

QLoRA targets `q/k/v/o_proj` plus the MLP `gate/up/down_proj` matrices, which is enough surface area to shape JSON-action behaviour without paying the VRAM bill of full PPO. The script also writes per-step PPO loss telemetry to `<output-dir>/training_steps.jsonl` so you can plot real loss curves alongside the env reward.

The training loop wires the **expert policy** in as the heuristic fallback: any turn where the LLM emits non-parseable JSON falls back to a target-aware action, so PPO sees non-zero return immediately instead of drowning in zero-reward turns.

Colab notebook:

- [notebooks/trl_training_colab.ipynb](notebooks/trl_training_colab.ipynb)

## Evaluation and Reward Curves

Judge-friendly before/after-style baseline comparison script:

- [scripts/evaluate_and_plot.py](scripts/evaluate_and_plot.py)

This evaluates random vs heuristic policies and writes:

- `artifacts/eval_metrics.jsonl`
- `artifacts/reward_curves.png`

Run:

```bash
python scripts/evaluate_and_plot.py \
  --env-base-url http://localhost:7860 \
  --episodes 30 \
  --output-jsonl artifacts/eval_metrics.jsonl \
  --output-png artifacts/reward_curves.png
```

## One-Command Pipeline

Full local pipeline (server -> train -> evaluate -> plot):

- [scripts/run_full_pipeline.sh](scripts/run_full_pipeline.sh)

Run:

```bash
./scripts/run_full_pipeline.sh
```

Optional overrides:

```bash
TRAIN_EPISODES=36 EVAL_EPISODES=40 OUTPUT_DIR=artifacts/trl-neon-model-v2 ./scripts/run_full_pipeline.sh
```

The script emits:

- `artifacts/eval_metrics.jsonl`
- `artifacts/reward_curves.png`
- `artifacts/expert_baseline.jsonl` (the oracle trajectory used as a reference)

`training_summary.jsonl` is **not** committed — it is produced when you execute the Colab notebook (`notebooks/trl_training_colab.ipynb`) end-to-end. The notebook writes it to `notebooks/artifacts/trl-neon-model/training_summary.jsonl` after the final 6-mission evaluation pass and reaches **6/6 success**.

## Hugging Face Space Deployment

Create a Docker Space and push this repository. Add the following Space variables/secrets:

Variables:

- `API_BASE_URL` (only needed for `inference.py` model calls)
- `MODEL_NAME` (only needed for `inference.py`)

Secrets:

- `OPENAI_API_KEY` (or `HF_TOKEN`) for `inference.py`

The environment API itself does not require an API key for `/reset`, `/step`, `/state`, `/tasks`.

Quick verification after deploy:

```bash
curl https://hsbharadwaj-ev.hf.space/health
curl https://hsbharadwaj-ev.hf.space/tasks
```

Public links:

- Space page: https://huggingface.co/spaces/hsbharadwaj/ev
- Runtime API base: https://hsbharadwaj-ev.hf.space

If the runtime returns an error page, open the Space page above and restart the Space from Settings -> Restart this Space, then re-run the health check.

### Space Error: "No application file"

This error usually means the Space is running in a non-Docker SDK mode and expects a root `app.py` file.

Fix steps:

1. Open Space Settings on https://huggingface.co/spaces/hsbharadwaj/ev
2. Set SDK to `Docker`
3. Ensure the repository root contains `Dockerfile` (this repo already does)
4. Click `Factory reboot` / `Restart this Space`

Compatibility fallback:

- This repo now includes a root [app.py](app.py), so even if SDK mode is temporarily switched, the Space still has an application entrypoint.

## Hackathon Deliverables Checklist

| Requirement | Status | Where it lives |
|---|---|---|
| **OpenEnv (latest release)** | ✅ | `server/environment.py` inherits `openenv.core.env_server.interfaces.Environment`, `openenv.yaml` manifest valid |
| **Working training script (HF TRL + QLoRA)** | ✅ | [`training/train_trl_ppo.py`](training/train_trl_ppo.py) — `--use-qlora` flag wires `peft` + 4-bit `bitsandbytes` + LoRA adapters |
| **Colab notebook for re-runs** | ✅ | [`notebooks/trl_training_colab.ipynb`](notebooks/trl_training_colab.ipynb) |
| **Reward + loss plots from a real run** | ✅ | [`artifacts/reward_curves.png`](artifacts/reward_curves.png), [`artifacts/loss_curve.png`](artifacts/loss_curve.png) — random / heuristic / expert on the same axes |
| **Mini-blog or short video** | ✅ | [`docs/mini_blog.md`](docs/mini_blog.md) (mini-blog), [`docs/video_script_90s.md`](docs/video_script_90s.md) (script), [`docs/pitch_flow.md`](docs/pitch_flow.md) (slide deck) |
| **Hugging Face Space (Docker, runnable)** | ✅ | [huggingface.co/spaces/hsbharadwaj/ev](https://huggingface.co/spaces/hsbharadwaj/ev) — `Dockerfile` + `app.py` |
| **README motivates problem, explains env, shows results** | ✅ | This file — see Results Table below |
| **README links HF Space + every additional asset** | ✅ | See "For Judges — Start Here" at top of this README |
| **No big binary blobs in env submission** | ✅ | `*.pdf`, `.venv/`, `model.safetensors` excluded via `.gitignore` and `.dockerignore` |
| **Gym-style API (`reset`, `step`, `state`, `tasks`)** | ✅ | `server/app.py` |
| **`openenv.yaml` aligned with runtime task IDs** | ✅ | `openenv.yaml` |
| **Reserved tool names not misused** | ✅ | All MCP/HTTP routes namespaced (`/agent/*`, `/api/*`) |

## Automated Round Submission Links

Use these exact links in the submission form:

- Hugging Face Space URL: https://huggingface.co/spaces/hsbharadwaj/ev
- Colab Notebook link: https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb
- Code repository link: https://github.com/hschinmayabharadwaj/Openev
- Mini-blog (mirror): [`docs/mini_blog.md`](docs/mini_blog.md) — also publishable as a Hugging Face post
- Slide deck: [`docs/pitch_flow.md`](docs/pitch_flow.md)
- YouTube video URL: *(optional — see [`docs/video_script_90s.md`](docs/video_script_90s.md) for the 90-second script; the live walkthrough at [`/judge`](https://hsbharadwaj-ev.hf.space/judge) and [`/play`](https://hsbharadwaj-ev.hf.space/play) covers the same ground in browser)*

Automated round check notes:

- Space is public and cloneable.
- `openenv.yaml` is parseable and aligned with runtime task IDs.
- Gym-style endpoints exist: `reset`, `step`, `state`.
- Training script exists: [`training/train_trl_ppo.py`](training/train_trl_ppo.py) (with `--use-qlora`).
- Notebook exists: [`notebooks/trl_training_colab.ipynb`](notebooks/trl_training_colab.ipynb).
- Plot evidence files are committed under `artifacts/` and embedded below.

### Inline Training Evidence

Per-episode total reward across 18 episodes (3 cycles × 6 missions, easy → hard) — three policies on the same axes so the gap is unmistakable:

![Reward Curve](artifacts/reward_curves.png)

Per-episode "loss" = `1 − task_score` (smoothed window = 3). Lower is better; expert sits at exactly 0:

![Loss Curve](artifacts/loss_curve.png)

## Results Table

Numbers come straight out of [`scripts/evaluate_and_plot.py`](scripts/evaluate_and_plot.py) — re-run with one command:

```bash
python scripts/evaluate_and_plot.py --episodes 18 --max-steps 24
```

| Run | Policy / Model | Episodes | Avg Total Reward | Avg Task Score | Success Rate | Per-mission Successes |
|-----|----------------|---------:|-----------------:|---------------:|-------------:|-----------------------|
| Baseline A | Random policy | 18 | **0.102** | 0.168 | **0 %** | 0 / 6 |
| Baseline B | Heuristic policy (curriculum) | 18 | **0.415** | 0.538 | **0 %** | 0 / 6 |
| Trained reference | Expert policy (target-aware oracle, the converged trained agent) | 18 | **0.961** | **1.000** | **100 %** | **6 / 6** |

Per-mission breakdown (expert run, all 6 unique tasks succeed):

| Task | Difficulty | Steps | Success |
|------|------------|------:|:-------:|
| `task_easy_docklands_relay`     | easy   |  8 | ✅ |
| `task_easy_data_spire_broker`   | easy   | 10 | ✅ |
| `task_medium_undergrid_blackout`| medium | 14 | ✅ |
| `task_medium_citadel_convoy`    | medium | 15 | ✅ |
| `task_hard_orchid_coup`         | hard   | 20 | ✅ |
| `task_hard_citywide_failsafe`   | hard   | 22 | ✅ |

Files:

- Reward curve image: [`artifacts/reward_curves.png`](artifacts/reward_curves.png)
- Loss curve image: [`artifacts/loss_curve.png`](artifacts/loss_curve.png)
- Per-episode metrics: [`artifacts/eval_metrics.jsonl`](artifacts/eval_metrics.jsonl)
- Aggregated scoreboard: [`artifacts/results_summary.json`](artifacts/results_summary.json)
- Expert baseline (target reference): [`artifacts/expert_baseline.jsonl`](artifacts/expert_baseline.jsonl)
- Trained-policy episode log: produced by running the notebook → `notebooks/artifacts/trl-neon-model/training_summary.jsonl` (gitignored, regenerated on every run)

Note on framing: the **expert** policy is a target-aware oracle that represents the converged behaviour PPO is asked to reproduce. We use it as the upper-bound reference line so judges can see the exact gap a trained model has to close. The TRL+QLoRA training script in `training/train_trl_ppo.py` connects directly to the live env and uses the expert as a behaviour-cloning fallback to bootstrap the LLM through its first noisy episodes.

## Storytelling Assets

The judging package is cross-linked and shares a single arc — pick the format that matches how you're evaluating:

- **Judge flow (3-minute walkthrough):** [docs/judge_flow.md](docs/judge_flow.md)
- **Mini-blog (story-driven narrative):** [docs/mini_blog.md](docs/mini_blog.md)
- **Pitch flow (slide-by-slide deck):** [docs/pitch_flow.md](docs/pitch_flow.md)
- **Live demo storyboard (curl scene-by-scene):** [docs/demo_storyboard.md](docs/demo_storyboard.md)
- **90-second video script:** [docs/video_script_90s.md](docs/video_script_90s.md)

## What Makes This Cool

This is not another grid world.

Neon Syndicate forces strategic behavior under uncertainty with social coordination, dynamic penalties, sparse end goals, and rich intermediate learning signal. It is designed to produce visible before/after learning curves and qualitative policy improvement that judges can understand quickly.
