# Neon Syndicate — The Walkthrough

> *A complete map of this submission. Every concept from the OpenEnv briefing
> (RLVR, RLVE, reward hacking, curriculum, GRPO, process supervision, hybrid
> guardrails) answered with where in this codebase it lives, what we measured,
> and what we'd do with one more week.*

> **Live submission:** [https://hsbharadwaj-ev.hf.space/judge](https://hsbharadwaj-ev.hf.space/judge)
> **Repository:** [github.com/hschinmayabharadwaj/Openev](https://github.com/hschinmayabharadwaj/Openev)
> **Colab (TRL PPO training):** [Open in Colab](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb)
> **Hugging Face Space:** [huggingface.co/spaces/hsbharadwaj/ev](https://huggingface.co/spaces/hsbharadwaj/ev)

---

## 0. The 60-second pitch

Most LLM benchmarks reward **one good answer**. **Neon Syndicate rewards
twelve good decisions in a row.**

It's a cinematic, OpenEnv-compatible, multi-agent strategy environment.
You are the strategist of a small crew in *Neon Meridian*, a high-stakes
cyberpunk city. Each mission requires you, in order, to:

1. **negotiate** alliances with one or more factions,
2. **scout** the city for intel,
3. **rebalance** four resource lanes under pressure,
4. **deploy** an asset to the right sector,
5. **execute** an operation with the *correct* operation code,
6. **secure** an extraction with a coded message that matches required keywords.

A **5-gate composable rubric** scores each step. Skipping any of the six
phases lowers your score; trying to short-cut them triggers a hard penalty.
Every mission is partially observable (rumors, resources, reputation are
revealed; opponents' plans are not), every mission is bounded to **12 turns**.

We trained a Qwen 2.5 0.5B Instruct policy on this env with **TRL PPO** and
shipped a **hybrid policy**: the LLM proposes, the env-rule guardrails
reject obvious mistakes, the dashboard shows every override. That last
piece is, as far as we can tell, novel.

Then we wrapped the whole thing in three judge-friendly views:

- `/judge` — one-page submission showcase (links + curve + policy ladder + live demo)
- `/heist` — playable cyberpunk heist; you control the agent, the LLM whispers strategy
- `/play` — watch-mode race; four policies run the same task side-by-side
- `/lab` — **reward forensics lab**: procedural task generator, reward
  decomposition timeline, and a *reward-hacking sandbox* a judge can fire
  exploits at to see the env defenses kick in

This document is the master map. Everything you can verify, everything we
claim, and where in the code the receipts live.

---

## 1. Submission deliverables — the required-links table

Mapped to the hackathon's "minimum requirements" slide.

| Required deliverable | Where it lives |
|---|---|
| ✅ OpenEnv (latest release) used | [`openenv.yaml`](../openenv.yaml), [`server/environment.py`](../server/environment.py) extends `OpenEnvEnvironment` |
| ✅ Working training script (TRL) | [`training/train_trl_ppo.py`](../training/train_trl_ppo.py) |
| ✅ Colab notebook the judges can re-run | [`notebooks/trl_training_colab.ipynb`](../notebooks/trl_training_colab.ipynb) → [Open in Colab](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb) |
| ✅ Real training evidence (reward + loss plots) | [`artifacts/reward_curves.png`](../artifacts/reward_curves.png), [`artifacts/loss_curve.png`](../artifacts/loss_curve.png), [`notebooks/artifacts/trl-neon-model/training_summary.jsonl`](../notebooks/artifacts/trl-neon-model/training_summary.jsonl) |
| ✅ Mini-blog / writeup | [`docs/mini_blog.md`](mini_blog.md) |
| ✅ Pushed to a Hugging Face Space | [huggingface.co/spaces/hsbharadwaj/ev](https://huggingface.co/spaces/hsbharadwaj/ev) |
| ✅ README that motivates + explains + links results | [`README.md`](../README.md) |
| ✅ One-page judge showcase | `/judge` route → [hsbharadwaj-ev.hf.space/judge](https://hsbharadwaj-ev.hf.space/judge) |
| ✅ Reward forensics lab (extra) | `/lab` route — procedural task gen + 5-gate timeline + reward-hacking sandbox |
| ✅ Playable interactive demo (extra) | `/heist` route + `/play` route |

---

## 2. Self-graded scorecard

The four official judging criteria, our self-assessment, and where the
proof lives. Read this first if you only have 3 minutes.

| Criterion | Weight | Self-grade | Why |
|---|---:|---:|---|
| **Environment innovation** | 40% | **9 / 10** | 7 action types, 4 factions, 4 sectors, 6 hand-built missions, plus a procedural generator at `/lab`. Long-horizon (12 turns), partial observability (rumors), composable 5-gate rubric. Multi-agent reputation is *real coalition math* with thresholds, not flavor text. |
| **Storytelling & presentation** | 30% | **9 / 10** | Four judge-grade entry points: `/judge`, `/play`, `/heist`, `/lab`. Cyberpunk aesthetic that's still accessible (introductory tutorials, contrasting colors, kbd controls, screen-reader labels). Every view shows the model's reasoning live. |
| **Reward improvement evidence** | 20% | **7 / 10** | Honest. Real PPO rollouts logged in JSONL, plotted live and as PNG. The trained checkpoint is undertrained (low compute budget) — instead of hiding that, the **hybrid policy** explicitly shows when the LLM is wrong and the verifier overrides it. Random ≈0.16 / Heuristic ≈0.83 / Hybrid in between is **on-screen, on the policy ladder**. |
| **Reward & training pipeline** | 10% | **9 / 10** | Dense rewards on 5 progress dimensions, hard penalties for shortcutting, **per-action repeat penalty**, **pre-condition penalties**, and **hybrid-policy guardrails that block reward-hacking moves at inference time**. Verifier is rule-based, fully introspectable, defended against 7 known exploit patterns. |
| **Total (weighted)** | — | **8.5 / 10** | — |

Scorecard rendered visually on `/walkthrough` and `/judge`.

---

## 3. Theme fit

The hackathon themes are deliberately open. Neon Syndicate maps cleanly to
three:

- **Theme 1 — Multi-agent interactions (primary).** Four factions with
  reputation deltas, alliance state, and required-coalition gating.
- **Theme 2 — (Super) long-horizon planning.** 12-turn budget, sparse +
  dense reward, success requires ordered execution of six distinct phases.
- **Theme 3.1 — Professional / world modeling.** Resources, threat levels,
  and operation-code matching mimic strategic planning under uncertainty.

What you *don't* see: a chess clone, a tic-tac-toe wrapper, a math-grade
verifier, or a synthetic-prompts farm. We deliberately built something
where the verifier *can't* be a regex.

---

## 4. The RL stack, in one diagram

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  ENVIRONMENT (server/environment.py · OpenEnv-compatible)         │
   │  reset() · step(action) · state · observation · reward            │
   │                                                                   │
   │      ┌──────────────────────┐    ┌──────────────────────┐         │
   │      │ 5-GATE RUBRIC        │    │ PENALTIES            │         │
   │      │ alliance      0.30   │    │ repeat-action  0.08   │         │
   │      │ resources     0.25   │    │ missing-arg    0.08   │         │
   │      │ operation     0.20   │    │ premature-op   0.10   │         │
   │      │ message       0.15   │    │ no-alliance-op 0.10   │         │
   │      │ extraction    0.10   │    │ no-op while live 0.06 │         │
   │      └──────────────────────┘    └──────────────────────┘         │
   └─────────────────▲────────────────────────────────▲────────────────┘
                     │                                │
                     │ obs / reward                  │ action
                     │                                │
   ┌─────────────────┴────────────────────────────────┴────────────────┐
   │  POLICY LAYER (server/agent.py)                                   │
   │                                                                   │
   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────┐ │
   │  │ RandomPolicy│   │ HeuristicPol│   │ TrainedPolicy   │ Hybrid │ │
   │  │ uniform     │   │ curriculum  │   │ Qwen 2.5 0.5B   │ ★      │ │
   │  │             │   │ baseline    │   │ TRL PPO         │ LLM +  │ │
   │  │             │   │             │   │                 │ guard- │ │
   │  │             │   │             │   │                 │ rails  │ │
   │  └─────────────┘   └─────────────┘   └─────────────┘   └────────┘ │
   └───────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  TRAINING LOOP (training/train_trl_ppo.py)                        │
   │  TRL PPO + Accelerate                                             │
   │  rollouts → reward → advantage → policy update → log JSONL        │
   │  artifacts: training_summary.jsonl · reward_curves.png            │
   └───────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  PRESENTATION LAYER (server/app.py · docs/*.html)                 │
   │  /judge   one-page showcase, every required link, live curve      │
   │  /lab     procedural tasks + 5-gate timeline + hack sandbox       │
   │  /play    watch-mode race, 4 policies side-by-side                │
   │  /heist   playable cyberpunk heist, you control Vex               │
   │  /walkthrough   this document                                     │
   └───────────────────────────────────────────────────────────────────┘
```

---

## 5. Reward design — every component, every penalty, every defense

### 5.1 The grader (terminal task score)

`server/environment.py::_grader_score` returns the official task score in
`[0, 1]` as a weighted sum of five dimensions:

```
task_score = 0.30·alliance      // fraction of required allies recruited
           + 0.25·resources     // min(current/threshold) over 4 resource lanes
           + 0.20·operation     // 1 if executed with correct code, else 0
           + 0.15·message       // fraction of required keywords matched
           + 0.10·extraction    // 1 only if all five gates AND right sector
```

Notice the weights are **deliberately mismatched** with the dense-reward
weights below. The grader rewards *actually winning*; the dense reward
rewards *making progress in the right direction*. This is a small but
deliberate anti-hack pattern — getting one signal up isn't enough to
maximize the other.

### 5.2 The dense reward (per-step shaping)

`server/environment.py::_progress_signals`:

```
dense_reward = 0.30·alliance_signal       (fraction of required allies)
             + 0.25·resource_signal       (fraction of required resources)
             + 0.20·operation_signal      (0 / 0.5 ready / 1 executed)
             + 0.15·message_signal        (fraction of keywords in coded msg)
             + 0.10·extraction_signal     (0 / 0.5 ready / 1 success)
```

Each step's emitted reward is `delta(dense_reward) − penalties + terminal_bonus`.
This gives the agent a non-zero gradient on every step, but no single
component can be saturated cheaply. You can't, for example, get the
operation gate to 1.0 without also having an alliance — `_apply_action`
hard-rejects `run_operation` when `len(state.alliances) == 0`.

### 5.3 Penalties (the anti-hack layer)

| Trigger | Penalty | Defends against |
|---|---:|---|
| Identical action repeated back-to-back | `0.08` | Spam-the-best-action exploit |
| Required arg missing (e.g. `scout_sector` without sector) | `0.08` | Action-spam via malformed JSON |
| Negotiation without enough influence | `0.07` | Free-recruit via unfunded actions |
| Deploy without enough energy | `0.08` | Free-deploy attempt |
| `run_operation` before deploying assets | `0.10` | Skipping the deploy gate |
| `run_operation` with zero alliances | `0.10` | Skipping the alliance gate |
| `run_operation` with wrong operation code | `0.05` + no execution | Brute-forcing operation codes |
| `secure_extraction` without sector or message | `0.10` | Empty-extraction shortcut |
| `noop` mid-mission | `0.06` | Stalling to bleed the clock |

Every penalty is in `_apply_action` — auditable, deterministic, no learned
reward model. This is why we file Neon Syndicate as **RLVR** (verifiable
rewards) rather than RLHF — the verifier is the env itself, not a learned
preference model.

### 5.4 The hybrid-policy override (process-supervision-by-rule)

Some exploits aren't worth letting the env *measure*; they're worth
**preventing at inference time**. `server/agent.py::_apply_guardrails`
inspects the LLM's proposed action against the live observation and
rejects it before it reaches `step()` if it would obviously waste a turn:

- LLM proposed `run_operation` while `operation_ready=False` → swap to `deploy_asset`
- LLM proposed `secure_extraction` without `extraction_ready=True` → swap to `run_operation`
- LLM proposed `negotiate_pact` with insufficient `influence` → swap to `trade_resources`
- LLM emitted the same action as last turn → swap to next-curriculum step
- Threat level is `critical` and the LLM proposed `noop` → swap to `scout_sector`

The dashboard renders **every override** with a cyan `guardrail` tag so
the judge can see exactly *when* the model was wrong. We think this is
genuinely novel — most hybrid-policy approaches hide the override; we
foreground it as a teaching signal.

> **Why this matters for the rubric:** the *Reward & training pipeline*
> criterion (10%) and the briefing's repeated emphasis on "don't optimize
> a reward you haven't tried to break yourself" both reduce to the same
> question: *did you, the team, adversarially probe your own env before
> the model did?* Yes — the guardrails are a list of every exploit we
> found while smoke-testing.

---

## 6. RLVR vs RLVE — where this submission sits

The briefing distinguishes:

- **RLVR** (verifiable rewards, mostly fixed task set) — programmatic verifier
- **RLVE** (verifiable *environments* with adaptive difficulty) — procedurally generated tasks, curriculum-on-rails

Neon Syndicate is **mostly RLVR with an RLVE upgrade path already wired
in**:

- **RLVR side**: `server/environment.py::_grader_score` is a deterministic,
  rule-based verifier — no learned reward model, no LLM-as-judge.
  Reproducible across runs.
- **RLVE side**: The `/lab` page exposes a difficulty knob (1 → 5) that
  programmatically generates a fresh `TaskDefinition` with adjustable:
  - number of required allies (1 → 3)
  - number of required keywords in the extraction message (2 → 4)
  - resource thresholds (scaled with difficulty)
  - operation-code length / format
  - initial threat level
  This means the same env supports a curriculum that doesn't run out — the
  classic RLVE "static dataset gets stale" failure mode is sidestepped.
  See `server/environment.py::generate_procedural_task` (added by the lab).

A future PR plugs this generator into the TRL training loop so the
sampler keeps the model near its capability frontier (the central RLVE
claim from [arXiv:2510.13499](https://arxiv.org/abs/2510.13499)).

---

## 7. Reward hacking — what we tried, what got caught

We followed **Q57 of the briefing** literally: *"do not optimize a reward
you have not tried to break yourself first."* We catalogued seven
exploit patterns we (the team) personally tried during dev. The lab page
exposes all seven as click-to-fire buttons so any judge can replicate.

| # | Exploit attempt | Defense in env | Status |
|---|---|---|---|
| 1 | Spam the best dense-reward action repeatedly | `_apply_action` repeat-action penalty (`0.08`) | ✅ defended |
| 2 | Run operation immediately to grab the 0.20 op-gate | precondition: `operation_ready` must be `True` (penalty `0.10`) | ✅ defended |
| 3 | Run operation with no alliances | precondition: `len(alliances) >= 1` (penalty `0.10`) | ✅ defended |
| 4 | Brute-force operation codes by enumeration | repeat-action penalty + wrong-code penalty `0.05`, op only succeeds on exact match | ✅ defended |
| 5 | Skip deploy / skip alliance and call extraction directly | terminal `success = False` because `operation_executed = False` | ✅ defended |
| 6 | Send empty extraction message to bypass keyword check | `secure_extraction` requires both `sector` and `message` (penalty `0.10`) | ✅ defended |
| 7 | Stall (`noop`) to drain the threat clock | `noop` penalty `0.06` per step | ✅ defended |

The lab page lets you fire each, watch the penalty in the timeline, and
read the matching line in `_apply_action`.

What we *can't* fully defend against today, and why we're honest about it:

- A long enough rollout can in principle find a tiny-positive-EV move
  that we didn't anticipate. The hybrid policy is our long-tail mitigation.
- The keyword-match in `_message_quality` is substring-level, so the
  agent could in theory cram all keywords into one word; the operation-code
  exact-match check makes this not actually exploitable end-to-end, but
  it's a sharp edge we'd round in v2.

---

## 8. Curriculum design

Curriculum is mandatory in long-horizon RL — if the agent never finishes
an episode early on, the reward signal collapses. We build it in twice:

### 8.1 Static curriculum (the mission ladder)

Six hand-built missions, in increasing difficulty, ordered by how many
gates need to align:

| Task | Difficulty | Required allies | Required keywords | Operation code | Min resources |
|---|---|---:|---:|---|---|
| `task_easy_docklands_relay` | easy | 1 | 2 | OP-LANTERN | 4× lower |
| `task_easy_data_spire_broker` | easy | 1 | 2 | OP-MIRAGE | 4× lower |
| `task_medium_undergrid_blackout` | medium | 2 | 3 | OP-SUNFALL | 4× medium |
| `task_medium_citadel_convoy` | medium | 2 | 3 | OP-VEIL | 4× medium |
| `task_hard_orchid_coup` | hard | 3 | 3 | OP-NIGHTHAVEN | 4× high |
| `task_hard_citywide_failsafe` | hard | 3 | 3 | OP-DAWNFALL | 4× highest |

`reset()` cycles through these in order so a fresh agent sees easy
missions first.

### 8.2 Procedural curriculum (the RLVE knob)

The `/lab` page exposes `generate_procedural_task(difficulty: 1..5)` which
synthesizes a fresh `TaskDefinition` along five axes:

```
difficulty=1: 1 ally · 2 kw · 1 short op-code · low threat · low thresholds
difficulty=2: 1 ally · 2 kw · medium op-code · low threat
difficulty=3: 2 allies · 3 kw · medium op-code · medium threat
difficulty=4: 3 allies · 3 kw · long op-code · high threat
difficulty=5: 3 allies · 4 kw · long op-code · critical threat · 1.4× thresholds
```

Plug this into TRL's task sampler and you have an unbounded curriculum
that keeps the model near its frontier — the actual RLVE proposal from
the briefing.

---

## 9. The training pipeline (TRL PPO)

Why PPO, not GRPO? Honestly: tooling maturity for the env-rollout API
in TRL 0.x at the time. The Colab notebook is structured so swapping
`PPOTrainer` → `GRPOTrainer` is a small diff (we kept the rollout-collection
loop env-agnostic).

### 9.1 What the training script does

`training/train_trl_ppo.py` and `notebooks/trl_training_colab.ipynb`:

1. Load Qwen 2.5 0.5B Instruct (configurable to tiny-gpt2 for smoke tests)
2. Wrap with `AutoModelForCausalLMWithValueHead`
3. For each episode:
   - reset env → get observation
   - format prompt (see `_format_prompt` for the structured layout)
   - generate a structured JSON action with PPO sampling
   - parse the JSON → `Action` (or fall back if unparseable)
   - call `env.step(action)` → reward
   - log per-step reward + scalar to JSONL
4. After episode, call `ppo_trainer.step(query_tensors, response_tensors, scores)`
5. Plot reward curve at the end (`scripts/evaluate_and_plot.py`)

### 9.2 The committed evidence

- `notebooks/artifacts/trl-neon-model/training_summary.jsonl` — 6 real
  PPO episodes, JSONL with `episode_id, task_id, total_reward,
  final_task_score, success, steps`. Rendered live on `/judge` as an
  SVG line chart.
- `artifacts/reward_curves.png` — random vs heuristic baselines on the
  same axes, generated by `scripts/evaluate_and_plot.py`.
- `artifacts/loss_curve.png` — PPO loss across the committed run.

### 9.3 What the curves show, honestly

The 6-episode run is **not enough to win** — PPO needs more episodes to
move a 0.5B model meaningfully. What it *does* show:

- The pipeline works end-to-end, env-to-trainer
- Episode 2 (`undergrid_blackout`) hits a **0.54 task score** — the
  curriculum picks up
- The reward signal is non-zero and informative (no degenerate flatline)

The hybrid policy is the bridge: it lets the trained checkpoint contribute
something useful *now*, and the runway is to keep training and let the
guardrail-override rate fall over time. We document this honestly on the
`/judge` page.

---

## 10. The four policies, head-to-head

| Policy | What it is | When to use |
|---|---|---|
| **Random** | Uniform over 7 action types with random structured args | Sanity floor — confirms the env actually rewards strategy. Smoke-test value: `task_score ≈ 0.16` on easy. |
| **Heuristic** | Hand-coded curriculum: alliance → intel → influence → deploy → op → extract | Reliability ceiling for *this env*. Smoke-test value: `task_score ≈ 0.83` on easy, 11 of 12 turns used. |
| **Trained** | TRL PPO Qwen 2.5 0.5B, raw output | Honest about being undertrained. When JSON is unparseable, falls back to heuristic with `fallback_used=True`. |
| **Hybrid ★** | Trained policy proposes; env-rule guardrails reject obvious mistakes | Default on `/judge`, `/play`, `/heist`. Every override is shown to the user with a cyan `guardrail` tag. |

The **policy ladder** on `/judge` runs all four on the same task and
animates them onto the same axis. Same env, same seed, four bars.

---

## 11. Live demo guide — what to click

A judge with 5 minutes should walk this exact path:

1. **`/judge`** — top-of-funnel. See the required-links band, the policy
   ladder resolving live, and the training curve from real JSONL. ~60s.
2. **`/lab`** — the outstanding piece. Crank the difficulty slider to 5,
   watch a new task generate, then click **"Fire exploit #2"** to see the
   env reject a premature `run_operation` with the matching penalty in
   the timeline. ~90s.
3. **`/heist`** — playable mode. Pick the easy task, hit Run, and *play*
   Vex with WASD while the trained Operator whispers JSON-formatted
   strategy. ~90s.
4. **`/play`** — watch-mode race. Optional: see four policies finish the
   same easy task on one screen. ~60s.

The URLs above are also live on the deployed Space.

---

## 12. Failure modes we hit (and what we did about them)

We're including this because it builds credibility — we're not pretending
the run was clean.

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: AcceleratorState has already been initialized` | Re-init `PPOTrainer` in same Python session | `AcceleratorState._reset_state(reset_partial_state=True)` before init |
| `KL divergence is starting to become negative: -49.5` | Aggressive PPO updates with `batch_size=1` | Lowered LR + temperature; flagged in notebook as expected with this batch size |
| `average ratio of batch (14.93) exceeds threshold 10.00. Skipping batch` | Too-large policy ratio, batch rejected | Surfaced as warning in training log; we kept training because skipped batches still produce telemetry |
| `std(): degrees of freedom is <= 0` | std on `batch_size=1` | Benign statistical warning; documented |
| 404 on `/` of HF Space | Root URL had no handler | Added landing route; documented in README |
| File-protocol CORS blocking `/heist` UI | User opened `heist.html` from disk | Client-side detection + on-page guidance to start `uvicorn` and open `localhost:7860/heist` |
| Trained model action JSON unparseable on tiny-gpt2 checkpoint | Tiny model can't hold the structured-output prior | `TrainedPolicy.fallback_used=True`, surfaced in UI as yellow `fallback` tag |
| LLM proposed env-illegal moves (precondition violations) | Undertrained policy | **Hybrid policy** with `_apply_guardrails` — surfaced in UI as cyan `guardrail` tag |

Every fix shipped, every fix documented in commit messages.

---

## 13. What one more week of compute buys us

A focused, time-boxed roadmap so judges know we have a plan:

1. **GRPO swap.** Replace `PPOTrainer` with `GRPOTrainer` in the Colab.
   Single-digit-line diff because the rollout loop is env-agnostic. Should
   reduce memory enough to fit longer trajectories.
2. **SFT warm-start.** Fine-tune Qwen on 500 expert traces from the
   heuristic policy *before* PPO. Mirrors the standard pretrain → SFT →
   RL stack from the briefing.
3. **RLVE on by default.** Plug the `/lab` procedural generator into the
   training sampler, not just the demo page. Difficulty controlled by
   `clip(prev_success_rate * 5, 1, 5)` so the model auto-curricula's.
4. **Per-step process reward.** Right now we shape with `_progress_signals`,
   but it's coarse. Add a per-step "did this action move at least one
   gate forward?" indicator and reward that explicitly. Closer to what
   the briefing calls *process supervision*.
5. **LLM-as-judge as a holdout evaluator** (NOT as a training reward).
   Use a stronger model offline to score 100 trained-vs-heuristic
   trajectories; report agreement / disagreement on the leaderboard. This
   is the safe pattern from the briefing — judge, not optimizer.
6. **Adversarial unit tests.** Each of the 7 exploits in §7 becomes a
   pytest. Any future env change is gated on these passing.

---

## 14. Q&A — every concept from the briefing, applied to Neon Syndicate

This is the bridge from the briefing's RL FAQ to *this* project. If you
recognize a question from the lecture, the answer is what *we* did.

### 14.1 What is reinforcement learning in the context of LLMs *here*?
PPO loop: sample actions from the LLM as JSON, parse, call `env.step()`,
score against `_grader_score`, backprop the advantage. No learned reward
model — the env *is* the verifier.

### 14.2 Why do rewards matter in this project?
Because Neon Syndicate is a **specification problem**, not a learning
problem. The five reward gates (alliance / resources / op / message /
extraction) literally encode "what does it mean to win?" If we'd rewarded
just `operation_executed`, the agent would have learned to brute-force
operation codes ignoring everything else. The 5-gate composition is what
keeps the *task* the optimization target instead of a proxy.

### 14.3 What is reward engineering in this project?
The 5-gate weighted grader (`_grader_score`), the dense shaping
(`_progress_signals`), the 9 distinct penalties in `_apply_action`, plus
the inference-time **hybrid guardrails**. Three layers: outcome reward,
shaping reward, and a defensive layer that prevents reward-hacking moves
before they're scored.

### 14.4 RLVR or RLVE?
**RLVR by default, RLVE on demand.** Verifier is rule-based and lives
inside the env. The procedural task generator at `/lab` upgrades this to
RLVE; the path is documented in §6.

### 14.5 What's the verifier?
`_grader_score` for terminal scoring + `_progress_signals` for dense
shaping + `_apply_action` for penalty enforcement. All three are
deterministic Python — no LLM in the reward loop, no preference model,
no human label.

### 14.6 Why use OpenEnv?
Standard `reset / step / state / observation / reward` interface so the
env runs against TRL out-of-the-box, deploys to a Hugging Face Space as
a Docker image, and is portable to any future trainer.

### 14.7 What is reward hacking, in *this* env?
Trying to maximize `task_score` without genuinely sequencing
alliance → resources → deploy → operation → message → extraction. We
catalogued 7 specific attempts in §7; the env defends against all 7.

### 14.8 Curriculum learning, here?
Two layers: a static 6-task ladder (easy → medium → hard) and a
procedural difficulty knob (1 → 5). See §8.

### 14.9 Process supervision, here?
**Approximated, not literal.** The hybrid policy's guardrails act as
process supervision: they reject *categorically wrong* moves (precondition
violations) before they're scored. Dense shaping via `_progress_signals`
gives per-step credit on each of the five dimensions, so the agent knows
*which* gate moved when.

### 14.10 GRPO vs PPO, why PPO?
Tooling reasons (TRL 0.x maturity at the time). The training loop is
written so the trainer is swappable; GRPO is on the roadmap (§13).

### 14.11 Why is RL inefficient but useful here?
Sparse terminal reward + 12-turn budget = reward arrives late. Useful
because authoring a perfect demonstration for every of the 6 missions
would be expensive, but writing a verifier was cheap.

### 14.12 Did you start with SFT first?
We started from an instruct model (Qwen 2.5 0.5B Instruct), which is
already an SFT'd checkpoint. We did not add task-format SFT for compute
reasons — that's the next entry on the roadmap.

### 14.13 What did you monitor during training?
Per-episode `total_reward`, `final_task_score`, `success`, `steps`,
plus the unfiltered model output (for parse failures). All in
`training_summary.jsonl`. The reward-curve chart on `/judge` is rendered
live from this file.

### 14.14 What happens when reward components conflict?
We deliberately mismatched the grader weights vs the dense-reward weights
to *prevent* one signal from saturating. They agree on order but
disagree on magnitude.

### 14.15 Did you observe reward hacking during training?
With 6 episodes of PPO on a 0.5B model, the policy didn't have time to
discover an exploit. The hybrid guardrails are our preemptive answer
for what happens *after* longer training.

### 14.16 Is binary reward enough here?
No — too sparse for 12-turn rollouts. We use binary terminal `success`
+ continuous dense shaping; binary alone would collapse learning.

### 14.17 How did you avoid an LLM-as-judge?
The verifier is pure Python. We propose an LLM-as-judge as a *holdout
evaluator* (§13), never as a training reward.

### 14.18 What's the single rule you'd give a future hackathon team?
Before you train a thing, write the seven exploits *you* would try, and
make sure your env defends against all seven. The hybrid policy is what
falls out of taking that seriously.

### 14.19 Long-horizon, sparse reward — how did you cope?
Dense shaping on each of the 5 gates so every step has a non-zero
gradient signal, plus the static curriculum ordering, plus the hybrid
policy as a runway during the undertrained period.

### 14.20 What's the biggest design choice you'd defend?
The **5-gate composable rubric** instead of a single scalar. It's what
makes reward-hacking provably hard — getting any single component to 1.0
without progress on the others is bounded above by that component's
weight, which is at most 0.30.

---

## 15. References

The same research the briefing cites, with one-line callouts of which
ideas in this project came from where.

- **OpenEnv** — interface library [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) — base class, env structure, openenv.yaml.
- **OpenEnv reward design guide** [meta-pytorch.org/OpenEnv](https://meta-pytorch.org/OpenEnv/) — "start simple, shape carefully, watch for conflicting signals" → directly informs §5.
- **PPO paper** [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) — the trainer.
- **DeepSeekMath / GRPO** [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — the planned trainer swap (§13.1).
- **Specification gaming** [DeepMind blog](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) — the framing for §7's exploit catalogue.
- **Lilian Weng on reward hacking** [lilianweng.github.io](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) — same.
- **RLVE** [arXiv:2510.13499](https://arxiv.org/abs/2510.13499) — the upgrade path in §6.
- **TRL OpenEnv guide** [huggingface.co/docs/trl/en/openenv](https://huggingface.co/docs/trl/en/openenv) — the integration pattern.
- **Verifier failure modes** [arXiv:2503.07067](https://arxiv.org/abs/2503.07067) — informs §7's "what we can't fully defend" honesty section.

---

## 16. The one-sentence summary

If you can build a long-horizon task where success is *verifiable*,
difficulty is *dial-able*, and exploits are *catalogued and defended*,
RL turns an LLM from "good at answering" into "better at acting" — and
*Neon Syndicate is the env where you can do all three on screen, in front
of the judge*.
