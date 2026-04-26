# Neon Syndicate — Judge Flow (3 Minutes to "I Get It")

> One page. One arc. Zero hunting.
> Judges should be able to grade us without leaving this document.

This is the canonical walkthrough. It mirrors [`mini_blog.md`](mini_blog.md) but is structured for *evaluation*, not reading. Every section says **what to look at** and **what to verify**.

---

## 00:00 — The Hook (10 seconds)

Most LLM benchmarks reward one good answer.
**Neon Syndicate rewards twelve good decisions in a row.**

The agent runs missions in Neon Meridian — a partially observable cyberpunk city — where it must negotiate with factions, manage four resource pools, execute a coded operation, and secure a clean extraction. Skip a step, the mission collapses.

Why it matters: this is the exact failure mode (long-horizon planning + coalition reasoning + delayed reward) that current models flunk.

---

## 00:10 — The Problem We're Solving (20 seconds)

| Capability gap in current LLMs | How Neon Syndicate forces it |
|---|---|
| Long-horizon credit assignment | Up to 12 turns, terminal success only |
| Theory-of-mind over multiple actors | 4 factions with conflicting incentives |
| Partial observability | World state revealed only via `scout_sector` and rumors |
| Reward hacking resistance | Extraction needs alliances **AND** resources **AND** op code **AND** message keywords **AND** correct sector |
| Anti-shortcut shaping | Repeated actions, premature ops, and passive play under critical threat all penalized |

See: [`server/environment.py`](../server/environment.py) — the `_apply_action` and `_grader_score` methods are the source of truth.

---

## 00:30 — The Environment (30 seconds)

**Theme fit:** Theme #1 Multi-Agent Interactions (primary) + Theme #2 Super Long-Horizon Planning (secondary).

**Action space (7 verbs):**
`scout_sector` · `negotiate_pact` · `trade_resources` · `deploy_asset` · `run_operation` · `secure_extraction` · `noop`

**World:**
- 4 factions — `ghostwire`, `iron_vultures`, `civic_shield`, `black_orchid`
- 4 sectors — `docklands`, `data_spire`, `undergrid`, `citadel_gate`
- 6 missions — easy → hard, each with its own coalition + op code + keyword set
- 12-turn episode budget

**API (OpenEnv-compatible):**
`GET /health` · `GET /tasks` · `POST /reset` · `POST /step` · `GET /state`

Live runtime: <https://hsbharadwaj-ev.hf.space>

---

## 01:00 — Reward Design (20 seconds)

Dense per-turn shaping + delayed terminal completion. Weights (from [`environment.py`](../server/environment.py) `_progress_signals`):

```
task_score = 0.30 * alliance_score
           + 0.25 * resource_score
           + 0.20 * operation_score
           + 0.15 * message_quality
           + 0.10 * extraction_score
```

Penalties stack on: repeated identical actions, missing payload fields, premature operation attempts, passive play at critical threat.

**Why this is hard to game:** every weight is bounded in `[0,1]` and the extraction success flag is gated on *all five* conditions aligning. The agent cannot spam one action type to climb the curve.

---

## 01:20 — Live Demo (40 seconds)

Full storyboard with real `curl` payloads: [`docs/demo_storyboard.md`](demo_storyboard.md).

**TL;DR scene list:**

1. **Brief.** `POST /reset {"task_id":"task_easy_docklands_relay"}` → mission brief reveals stakes + rumors.
2. **Scout.** `POST /step {"action_type":"scout_sector","sector":"docklands"}` → intel jumps, threat drops.
3. **Negotiate.** `POST /step {"action_type":"negotiate_pact","faction":"ghostwire"}` → alliance forms (the rumor said Ghostwire hates loud ops; we listened).
4. **Resource.** `POST /step {"action_type":"trade_resources","resource":"intel","amount":20}`.
5. **Deploy.** `POST /step {"action_type":"deploy_asset","sector":"docklands"}` → `operation_ready: true`.
6. **Run op.** `POST /step {"action_type":"run_operation","operation_code":"OP-LANTERN"}` → `operation_executed: true`.
7. **Extract.** `POST /step {"action_type":"secure_extraction","sector":"docklands","message":"window open, relay clean exit"}` → `success: true`.

If a judge wants chaos: re-run with `task_hard_orchid_coup` (3-faction coalition, critical threat, no margin for error).

---

## 02:00 — Training Evidence (30 seconds)

**What we trained:** `Qwen2.5-0.5B-Instruct` with TRL PPO directly against the live environment API.

**Where to look:**
- Script: [`training/train_trl_ppo.py`](../training/train_trl_ppo.py)
- Colab (one-click reproduce): [`notebooks/trl_training_colab.ipynb`](../notebooks/trl_training_colab.ipynb)
- Reward curve: ![reward](../artifacts/reward_curves.png)
- Loss curve: ![loss](../artifacts/loss_curve.png)

**Reproducibility command (one line):**

```bash
./scripts/run_full_pipeline.sh
```

That spins up the server, trains, evaluates, and writes:

- `artifacts/eval_metrics.jsonl`
- `artifacts/reward_curves.png`
- `artifacts/expert_baseline.jsonl`

`training_summary.jsonl` (the trained-policy episode log) is produced when you execute the Colab notebook — it is gitignored and regenerated on every run.

---

## 02:30 — Results (20 seconds)

Baseline comparison (random vs heuristic vs trained) lives in [`scripts/evaluate_and_plot.py`](../scripts/evaluate_and_plot.py).

| Run | Policy | Episodes | Avg Reward | Task Score | Success |
|---|---|---|---|---|---|
| A | Random | 30 | _see metrics file_ | _see metrics file_ | _see metrics file_ |
| B | Heuristic | 30 | _see metrics file_ | _see metrics file_ | _see metrics file_ |
| C | TRL PPO (Qwen2.5-0.5B) | 30 | _see metrics file_ | _see metrics file_ | _see metrics file_ |

Numbers are written to `artifacts/eval_metrics.jsonl` by the eval script — judges can re-run and verify without trusting our README.

**Qualitative trend (what we want judges to see in the curves):**

- Earlier alliance formation → alliance component saturates faster.
- Fewer premature `run_operation` calls → operation penalty drops.
- Tighter extraction messages (keyword coverage ↑) → message-quality component climbs.

---

## 02:50 — Hackathon Deliverables Checklist (10 seconds)

| Requirement | Status | Pointer |
|---|---|---|
| OpenEnv latest release | ✅ | [`openenv.yaml`](../openenv.yaml) + Gym-style `/reset`,`/step`,`/state` |
| Minimal training script (TRL or Unsloth) | ✅ | [`training/train_trl_ppo.py`](../training/train_trl_ppo.py) |
| Reward / loss evidence | ✅ | [`artifacts/reward_curves.png`](../artifacts/reward_curves.png), [`artifacts/loss_curve.png`](../artifacts/loss_curve.png) |
| Mini-blog or <2 min video | ✅ | [`docs/mini_blog.md`](mini_blog.md), [`docs/video_script_90s.md`](video_script_90s.md) |
| Hugging Face Space deployment | ✅ | <https://huggingface.co/spaces/hsbharadwaj/ev> |
| Public code repo | ✅ | <https://github.com/hschinmayabharadwaj/Openev> |
| Colab notebook | ✅ | [Open in Colab](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb) |

---

## 03:00 — One-Click Verification

If you only run one command:

```bash
curl https://hsbharadwaj-ev.hf.space/health
curl https://hsbharadwaj-ev.hf.space/tasks
```

If you have 60 seconds:

```bash
./scripts/run_full_pipeline.sh
```

If you have a minute and want the cinematic version: [`docs/demo_storyboard.md`](demo_storyboard.md).

---

## Companion Documents

- **Story-driven mini-blog:** [`docs/mini_blog.md`](mini_blog.md) — the human-readable narrative.
- **Pitch / presentation flow:** [`docs/pitch_flow.md`](pitch_flow.md) — slide-by-slide deck outline with speaker notes.
- **Live demo storyboard:** [`docs/demo_storyboard.md`](demo_storyboard.md) — exact API calls, scene by scene.
- **90-second video script:** [`docs/video_script_90s.md`](video_script_90s.md) — recorded demo voice-over.

All four documents share the same arc. Pick the format that matches how you're evaluating.
