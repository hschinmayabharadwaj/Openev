# Neon Syndicate OpenEnv: Training LLMs for Multi-Agent Strategy Under Uncertainty

## Problem
LLMs still struggle with strategic behavior that requires theory-of-mind style reasoning over multiple actors, delayed rewards, and partial observability.

Neon Syndicate OpenEnv targets this gap with a cyberpunk mission world where an agent must coordinate factions, balance resources, and execute long-horizon plans.

## Environment
In each mission, the agent operates in Neon Meridian and must:

- negotiate faction pacts,
- scout sectors,
- trade constrained resources,
- deploy assets,
- execute a specific operation code,
- and secure extraction with a quality message.

The world is partially observable via mission rumors, threat levels, and evolving reputation signals.

## Why It Matters
This benchmark captures realistic properties needed for next-gen agent training:

- multi-agent bargaining,
- sequential dependency across turns,
- sparse terminal success with dense shaping,
- and anti-shortcut reward structure.

## Reward Design
Reward components include:

- alliance completion progress,
- resource threshold progress,
- operation execution progress,
- extraction message quality,
- terminal extraction success.

Penalties include repeated actions, invalid payloads, and passive play at critical threat.

## Training Setup
We provide a minimal TRL PPO loop that interacts directly with the environment API:

- script: [training/train_trl_ppo.py](../training/train_trl_ppo.py)
- notebook: [notebooks/trl_training_colab.ipynb](../notebooks/trl_training_colab.ipynb)

We also provide a reproducible baseline comparison and reward-curve plotting script:

- script: [scripts/evaluate_and_plot.py](../scripts/evaluate_and_plot.py)

## Results Snapshot
Add your run outputs here:

- reward curve image: `artifacts/reward_curves.png`
- metrics dump: `artifacts/eval_metrics.jsonl`

Key metrics to report:

- average total reward,
- average task score,
- success rate,
- before vs after behavior summary.

## Links
- Hugging Face Space: https://huggingface.co/spaces/hsbharadwaj/ev
- Runtime API: https://hsbharadwaj-ev.hf.space
- Colab training run: `TODO`
- WandB run (optional): `TODO`
- Demo video (<2 min): `TODO`
