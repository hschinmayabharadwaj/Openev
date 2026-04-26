# Neon Syndicate OpenEnv

### *Training LLMs to think across twelve turns, not one.*

> **Cold open.** It's 2:47 AM in Neon Meridian. The Freelance Union wants a surveillance relay intercepted before midnight curfew. Ghostwire hates loud operations. Patrols swap every fifteen minutes. The agent has twelve turns to negotiate, scout, trade, deploy, execute, and disappear.
> If it skips a step, the city wakes up.

---

## The Problem

Modern LLMs are tuned to be brilliant in a single response. They are surprisingly bad at *sequences* — at remembering what they tried three turns ago, at modeling that another agent has its own incentives, at not panicking when the reward only arrives at the end.

The capability gap, in plain terms:

- **Long-horizon credit assignment.** Did turn 4 cause turn 11's failure? Most models can't tell.
- **Theory-of-mind across actors.** Four factions, each with priors. Negotiating with Ghostwire while Iron Vultures watch is *different* from negotiating in isolation.
- **Partial observability.** The world reveals itself only through rumors and scouting. The model has to *update beliefs*, not just retrieve facts.
- **Anti-shortcut reward shaping.** If a single behavior can game the score, the agent will find it. The reward must reflect the *actual* mission, end to end.

We built Neon Syndicate to target that exact shape of failure.

---

## The Environment

Neon Meridian is a partially observable cyberpunk city with four factions, four sectors, and six missions at three difficulty tiers.

The agent has seven verbs:

| Action | What it does |
|---|---|
| `scout_sector` | +12 intel; threat drops if you scout the *target* sector |
| `negotiate_pact` | Spend influence to raise faction reputation; alliance forms at trust ≥ 35 |
| `trade_resources` | Convert one resource pool into another |
| `deploy_asset` | Pre-position assets in a sector; arms the operation |
| `run_operation` | Fire a coded operation — needs deploy + at least one ally |
| `secure_extraction` | Terminal action — needs sector + message |
| `noop` | Penalized to discourage stalling |

A mission ends in success only when **five independent conditions align**:

1. The required allies are in the coalition.
2. Resource minimums are cleared.
3. The operation has been executed with the correct code.
4. The extraction message contains the required keywords.
5. The extraction is in the correct sector.

Twelve-turn budget. No restart. The world is what the rumors say it is.

---

## Why This Matters

Most LLM benchmarks reward one good answer. This one rewards twelve good decisions in a row, made under uncertainty, against other actors, with a delayed payoff.

That is the exact shape of behavior we need for next-gen agentic workflows — where the cost of being wrong on turn 3 is invisible until the chain finishes — and exactly the shape that current models miss.

Neon Syndicate captures four properties simultaneously:

- multi-agent bargaining,
- sequential dependency across turns,
- sparse terminal success with dense intermediate shaping,
- and an anti-shortcut reward structure that resists single-action exploitation.

---

## Reward Design

Dense per-turn shaping, weighted toward the capabilities we care about:

```
task_score = 0.30 * alliance_score
           + 0.25 * resource_score
           + 0.20 * operation_score
           + 0.15 * message_quality
           + 0.10 * extraction_score
```

Penalties stack: repeated identical actions, missing payload fields, running an operation before deploying, calling extraction without a message, and passive play under critical threat. The agent gets a small terminal completion bonus for resolving the episode and a larger one for actually succeeding.

The combination is intentional. Dense shaping keeps the gradient alive across turns. The terminal success gate keeps the agent honest — no single component can carry the score.

---

## Training Setup

We provide a minimal TRL PPO loop that talks directly to the live environment API:

- Script: [`training/train_trl_ppo.py`](../training/train_trl_ppo.py)
- Colab notebook: [`notebooks/trl_training_colab.ipynb`](../notebooks/trl_training_colab.ipynb)

We also ship a baseline comparison and reward-curve plotter:

- Script: [`scripts/evaluate_and_plot.py`](../scripts/evaluate_and_plot.py)

Reproduce everything in one command:

```bash
./scripts/run_full_pipeline.sh
```

That spins up the server, trains, evaluates random + heuristic + trained policies, and writes:

- `artifacts/eval_metrics.jsonl`
- `artifacts/reward_curves.png`
- `artifacts/expert_baseline.jsonl`

`training_summary.jsonl` (the trained-policy episode log) is **not** committed to the repo. Run the Colab notebook to produce it — the notebook writes a fresh `notebooks/artifacts/trl-neon-model/training_summary.jsonl` after the final 6-mission evaluation pass and reaches **6/6 success**.

---

## Results Snapshot

Reward curve and loss curve:

![Reward Curve](../artifacts/reward_curves.png)
![Loss Curve](../artifacts/loss_curve.png)

Numbers (filled from `artifacts/eval_metrics.jsonl`):

| Run | Policy / Model | Episodes | Avg Total Reward | Avg Task Score | Success Rate |
|---|---|---|---|---|---|
| Baseline A | Random policy | 30 | _see metrics file_ | _see metrics file_ | _see metrics file_ |
| Baseline B | Heuristic policy | 30 | _see metrics file_ | _see metrics file_ | _see metrics file_ |
| Trained | TRL PPO (`Qwen2.5-0.5B-Instruct`) | 30 | _see metrics file_ | _see metrics file_ | _see metrics file_ |

What we want a reader to see in the curves: alliances forming earlier, fewer premature `run_operation` calls, and tighter extraction messages as training progresses.

---

## A Run, Annotated

A clean pass on `task_easy_docklands_relay`:

1. **Scout `docklands`** — +12 intel, threat drops because we scouted the target sector.
2. **Negotiate `ghostwire`** — alliance forms; the rumor said they hate loud ops, so we lead with them.
3. **Trade into intel** — clears the 35 intel minimum.
4. **Deploy at `docklands`** — operation is now armed.
5. **Run `OP-LANTERN`** — code matches; threat drops again.
6. **Secure extraction at `docklands`** with the message *"window open on the relay — clean exit confirmed"* — three keywords matched, all five gates aligned.

Mission cleared in seven turns out of twelve. Full storyboard with curl payloads: [`demo_storyboard.md`](demo_storyboard.md).

---

## Where to Go Next

- **Judge flow (3-minute walkthrough):** [`judge_flow.md`](judge_flow.md)
- **Pitch deck flow:** [`pitch_flow.md`](pitch_flow.md)
- **Live demo storyboard:** [`demo_storyboard.md`](demo_storyboard.md)
- **90-second video script:** [`video_script_90s.md`](video_script_90s.md)

## Links

- Hugging Face Space: <https://huggingface.co/spaces/hsbharadwaj/ev>
- Runtime API: <https://hsbharadwaj-ev.hf.space>
- Colab training run: [one-click open](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb)
- Code repository: <https://github.com/hschinmayabharadwaj/Openev>
- Demo video (<2 min): `TODO_ADD_YOUTUBE_LINK`

---

*Neon Syndicate is the smallest possible benchmark that asks an LLM to plan, bargain, and finish — in that order. If the model can't do all three, the city notices.*
