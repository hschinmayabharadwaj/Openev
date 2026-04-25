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

A cinematic OG-style strategy game environment for training LLM agents on coalition politics, bargaining, and long-horizon execution.

## Theme Fit (Best Choice)

Primary theme: Theme #1 Multi-Agent Interactions  
Secondary theme: Theme #2 Super Long-Horizon Planning  
Also strong on: Theme #3.1 Professional World Modeling

Why this is a top-tier fit:

- Multi-agent incentives are central: factions cooperate conditionally and have competing interests.
- Planning is long-horizon: success requires ordered decisions over up to 12 turns with delayed payoff.
- World state is partially observable: rumors and outcomes require belief updates.
- Reward hacking is constrained: extraction only succeeds if coalition, resources, operation code, and message quality all align.

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

## Minimal TRL Training Pipeline

A minimal end-to-end TRL PPO script is provided and directly interacts with this environment API:

- [training/train_trl_ppo.py](training/train_trl_ppo.py)

Run locally:

```bash
pip install -e .[training]
python training/train_trl_ppo.py \
  --env-base-url http://localhost:7860 \
  --episodes 24 \
  --output-dir artifacts/trl-neon-model
```

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
- `artifacts/trl-neon-model/training_summary.jsonl`

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

Minimum requirements mapping:

- OpenEnv latest release: yes (manifest + API env)
- Minimal training script (Unsloth or HF TRL): add a Colab notebook linked below
- Reward/loss evidence: add reward curve PNGs in repo
- Mini-blog or <2 min video: link in this README
- Hugging Face Space deployment: link in this README

Fill these placeholders before submission:

- HF Space URL: https://huggingface.co/spaces/hsbharadwaj/ev
- Colab training notebook: [notebooks/trl_training_colab.ipynb](notebooks/trl_training_colab.ipynb)
- Reward curve image: `artifacts/reward_curves.png`
- Mini-blog draft: [docs/mini_blog.md](docs/mini_blog.md)
- Short video script: [docs/video_script_90s.md](docs/video_script_90s.md)

## Results Table Template

Use this table in your final README before submission:

| Run | Policy/Model | Episodes | Avg Total Reward | Avg Task Score | Success Rate |
|-----|--------------|----------|------------------|----------------|--------------|
| Baseline A | Random policy | 30 | TODO | TODO | TODO |
| Baseline B | Heuristic policy | 30 | TODO | TODO | TODO |
| Trained | TRL PPO (`Qwen2.5-0.5B-Instruct`) | 30 | TODO | TODO | TODO |

Attach below the table:

- Reward curve image: `artifacts/reward_curves.png`
- Metrics file: `artifacts/eval_metrics.jsonl`

## Storytelling Assets

- Mini-blog draft: [docs/mini_blog.md](docs/mini_blog.md)
- 90-second demo script: [docs/video_script_90s.md](docs/video_script_90s.md)

## What Makes This Cool

This is not another grid world.

Neon Syndicate forces strategic behavior under uncertainty with social coordination, dynamic penalties, sparse end goals, and rich intermediate learning signal. It is designed to produce visible before/after learning curves and qualitative policy improvement that judges can understand quickly.
