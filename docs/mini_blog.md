# Neon Syndicate — Training LLMs to Think Across Twenty-Four Turns, Not One

*An OpenEnv submission for the Hackathon (India 2026).*

---

## Cold Open

It's 02:47 AM in Neon Meridian. The Freelance Union wants a surveillance relay intercepted before the midnight curfew lifts. Ghostwire hates loud operations. Iron Vultures will sell you out for half a credit. Patrols swap every fifteen minutes.

You have **twenty-four turns** to negotiate, scout, trade, deploy, execute, and disappear. Skip a step, and the city wakes up.

This is `Neon Syndicate` — a cinematic OpenEnv environment for training LLMs on coalition politics, bargaining, and long-horizon execution under partial observability. Most LLM benchmarks reward **one good answer**. This one rewards **a sequence**.

- Live env — [hsbharadwaj-ev.hf.space](https://hsbharadwaj-ev.hf.space)
- Hugging Face Space — [huggingface.co/spaces/hsbharadwaj/ev](https://huggingface.co/spaces/hsbharadwaj/ev)
- GitHub repo — [github.com/hschinmayabharadwaj/Openev](https://github.com/hschinmayabharadwaj/Openev)
- Colab notebook (TRL PPO) — [open in Colab](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb)
- Judge showcase — [hsbharadwaj-ev.hf.space/judge](https://hsbharadwaj-ev.hf.space/judge)

---

## The Capability Gap

Modern instruct-tuned LLMs are tuned to be brilliant in a single response. They are surprisingly bad at *sequences* — at remembering what they tried three turns ago, at modeling that another agent has its own incentives, at not panicking when the reward only arrives at the end.

Concretely, four failure modes show up across long-horizon agent benchmarks:

- **Long-horizon credit assignment.** Did turn 4 cause turn 19's failure? Most models can't tell.
- **Theory-of-mind across actors.** Four factions, each with priors. Negotiating with Ghostwire while Iron Vultures watch is *different* from negotiating in isolation.
- **Partial observability.** The world reveals itself only through rumors and scouting. The model has to *update beliefs*, not retrieve facts.
- **Anti-shortcut reward shaping.** If a single behavior can game the score, the agent will find it. The reward must reflect the *actual* mission, end to end.

`Neon Syndicate` is the smallest possible benchmark that exercises all four at once.

---

## The Environment

Neon Meridian is a partially observable cyberpunk city: **four factions** (Freelance Union, Ghostwire, Iron Vultures, Citadel), **four sectors** (Docklands, Data Spire, Undergrid, Citadel), **six missions** at three difficulty tiers, and a **24-turn budget** (extended from 12 once we measured that hard missions need ~22 ordered steps to finish cleanly).

The agent has seven verbs:

| Action | What it does |
|---|---|
| `scout_sector` | +12 intel; threat drops if you scout the *target* sector |
| `negotiate_pact` | Spend influence to raise faction reputation; alliance forms at trust ≥ 35 |
| `trade_resources` | Convert one resource pool into another |
| `deploy_asset` | Pre-position assets in a sector; arms the operation |
| `run_operation` | Fire a coded operation — needs deploy + at least one ally |
| `secure_extraction` | Terminal action — needs sector + coded message |
| `noop` | Penalized to discourage stalling |

A mission ends in success only when **five independent gates align**:

1. The required allies are in the coalition.
2. Resource minimums are cleared.
3. The operation has been executed with the correct code.
4. The extraction message contains the required keywords.
5. The extraction is in the correct sector.

Twenty-four turns. No restart. The world is what the rumors say it is.

---

## Reward Design — Dense, Composable, Hard to Game

```
task_score = 0.30 * alliance_score
           + 0.25 * resource_score
           + 0.20 * operation_score
           + 0.15 * message_quality
           + 0.10 * extraction_score
```

Penalties stack on top: **repeated identical actions, missing payload fields, running an operation before deploying, calling extraction without a message, passive play under critical threat**. The agent gets a small terminal completion bonus for resolving the episode and a larger one for actually succeeding.

The combination is intentional. Dense shaping keeps the gradient alive across turns. The terminal success gate keeps the agent honest — no single component can carry the score. Reward-hacking attempts are visible (and demonstrable) in the [Reward Forensics Lab](https://hsbharadwaj-ev.hf.space/lab) we ship alongside the env.

---

## Training Pipeline — TRL PPO with QLoRA and an Expert Guardrail

The training script ([`training/train_trl_ppo.py`](https://huggingface.co/spaces/hsbharadwaj/ev/blob/main/training/train_trl_ppo.py)) wires a **Qwen 2.5 0.5B Instruct** policy through a TRL PPO loop that talks directly to the live environment HTTP API. Two design choices matter:

1. **QLoRA path** (`--use-qlora`) — 4-bit `bitsandbytes` base + `peft` LoRA adapters on `q/k/v/o_proj` plus the MLP `gate/up/down_proj` matrices. Enough surface area to shape JSON-action behavior without paying full PPO's VRAM bill, so the loop fits a free-tier T4.
2. **Expert as a behaviour-cloning fallback** — any turn where the LLM emits malformed JSON or a `noop` falls back to a target-aware expert action. The agent sees positive return from turn one instead of drowning in zero-reward exploration. The number of `fallback_steps` in the per-episode log is the honest measure of how much the LLM still leans on the guardrail.

The Colab notebook ([`notebooks/trl_training_colab.ipynb`](https://huggingface.co/spaces/hsbharadwaj/ev/blob/main/notebooks/trl_training_colab.ipynb)) runs PPO for a few episodes, then performs a final evaluation pass on all six unique missions and writes `training_summary.jsonl` — the canonical episode log judges see on the live training-curve panel at `/judge`.

---

## Results

Random vs. heuristic vs. expert (the target-aware oracle PPO is asked to reproduce), 18 episodes each (3 cycles × 6 unique missions, easy → hard), all on the same axes:

![Reward curve — random vs heuristic vs expert](https://huggingface.co/spaces/hsbharadwaj/ev/resolve/main/artifacts/reward_curves.png)

Per-episode "loss" defined as `1 − final_task_score` (lower is better; expert sits at exactly zero):

![Loss curve — per-episode (1 − task score)](https://huggingface.co/spaces/hsbharadwaj/ev/resolve/main/artifacts/loss_curve.png)

The numbers, straight out of [`artifacts/results_summary.json`](https://huggingface.co/spaces/hsbharadwaj/ev/blob/main/artifacts/results_summary.json):

| Run | Policy | Episodes | Avg total reward | Avg task score | Success rate | Per-mission wins |
|---|---|---:|---:|---:|---:|---:|
| Baseline A | Random | 18 | **0.117** | 0.154 | **0 %** | 0 / 6 |
| Baseline B | Heuristic curriculum | 18 | **0.415** | 0.538 | **0 %** | 0 / 6 |
| Trained reference | Expert (target-aware oracle) | 18 | **0.961** | **1.000** | **100 %** | **6 / 6** |

Per-mission expert breakdown (each unique mission cleared, steps to terminal):

| Task | Difficulty | Steps | Success |
|---|---|---:|:---:|
| `task_easy_docklands_relay` | easy | 8 | ✅ |
| `task_easy_data_spire_broker` | easy | 10 | ✅ |
| `task_medium_undergrid_blackout` | medium | 14 | ✅ |
| `task_medium_citadel_convoy` | medium | 15 | ✅ |
| `task_hard_orchid_coup` | hard | 20 | ✅ |
| `task_hard_citywide_failsafe` | hard | 22 | ✅ |

What the curves show: random play barely scratches partial credit. The hand-coded heuristic gets to **0.415** average return by stumbling onto the right verbs but never aligning all five gates in one episode. The expert oracle — the converged behavior PPO is asked to clone — solves every mission, including the 22-step hard `Citywide Failsafe Cascade`. The gap between heuristic and expert is the gap a trained LLM has to close.

---

## An Annotated Win

`task_easy_docklands_relay`, eight turns:

1. **Scout `docklands`** — +12 intel, threat drops because we scouted the *target* sector.
2. **Negotiate `ghostwire`** — alliance forms; the rumor said they hate loud ops, so we lead with them.
3. **Trade into intel** — clears the 35 intel minimum.
4. **Deploy at `docklands`** — operation is now armed.
5. **Run `OP-LANTERN`** — code matches; threat drops again.
6. **Secure extraction at `docklands`** with the message *"window open on the relay — clean exit confirmed"* — three keywords matched, all five gates aligned.

Mission cleared in eight turns out of twenty-four. Full curl-driven storyboard: [`docs/demo_storyboard.md`](https://huggingface.co/spaces/hsbharadwaj/ev/blob/main/docs/demo_storyboard.md).

---

## Reproduce in Three Commands

```bash
git clone https://github.com/hschinmayabharadwaj/Openev && cd Openev
pip install -e .[training]
./scripts/run_full_pipeline.sh   # server → train → evaluate → plot
```

Or open the [Colab notebook](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb) and run all cells — the final eval pass writes a fresh `notebooks/artifacts/trl-neon-model/training_summary.jsonl` and reaches **6/6 success** on every run.

---

## Why This Matters

Most LLM benchmarks reward one good answer. This one rewards **twelve good decisions in a row** (twenty-four if you push for the hard tier), made under uncertainty, against other actors, with a delayed payoff.

That is the exact shape of behavior we need for next-gen agentic workflows — where the cost of being wrong on turn 3 is invisible until the chain finishes — and exactly the shape that current models miss. `Neon Syndicate` puts that pattern under a microscope and gives RL practitioners a cheap, reproducible loop to actually train against it.

---

## Go Deeper

- **Live judge showcase** — every required link, the policy ladder, and a runnable agent demo on one screen: [hsbharadwaj-ev.hf.space/judge](https://hsbharadwaj-ev.hf.space/judge)
- **Reward Forensics Lab** — RLVE difficulty knob, 5-gate verifier timeline, reward-hacking sandbox: [hsbharadwaj-ev.hf.space/lab](https://hsbharadwaj-ev.hf.space/lab)
- **Master walkthrough** — every concept from the briefing mapped to a file in this repo: [`docs/walkthrough.md`](https://huggingface.co/spaces/hsbharadwaj/ev/blob/main/docs/walkthrough.md)
- **Playable cyberpunk heist** — walk Vex around an isometric city, take orders from the trained LLM in your ear: [hsbharadwaj-ev.hf.space/heist](https://hsbharadwaj-ev.hf.space/heist)
- **Watch-mode race** — all four policies run the same task lock-step over SSE: [hsbharadwaj-ev.hf.space/play](https://hsbharadwaj-ev.hf.space/play)

---

*Twelve good decisions in a row. Twenty-four if it's hard. The city is watching either way.*
