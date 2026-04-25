# Neon Syndicate — Pitch Flow

> 10 slides. ~3 minutes. Built for hackathon judging.
> Mirrors [`mini_blog.md`](mini_blog.md) but compressed into a deck.

Each slide has: **visual**, **on-slide text**, **speaker note**, and **timing**. Drop straight into Slides / Keynote / Pitch.

---

## Slide 1 — Title (0:00 → 0:08)

**Visual:** Neon Meridian skyline silhouette + project name in a cyberpunk sans.
**On-slide:**

> # Neon Syndicate OpenEnv
> **Training LLMs to think across 12 turns, not 1.**
> *Multi-agent strategy under uncertainty.*

**Speaker note (8s):**
"Hi — Neon Syndicate. We trained an LLM to play a long-horizon strategy game where every shortcut gets punished. Three minutes, then numbers."

---

## Slide 2 — The Problem (0:08 → 0:30)

**Visual:** Split panel — left: "What current LLMs do well" (single-turn QA, code completion). Right: "Where they fail" (multi-turn coalitions, delayed rewards, partial state).

**On-slide:**

> LLMs are good at one good answer.
> They're bad at **twelve good decisions in a row**.
> - Long-horizon credit assignment
> - Theory-of-mind across multiple actors
> - Partial observability
> - Resistance to reward hacking

**Speaker note (22s):**
"Today's models can ace a code interview question and then fall apart the moment a task spans multiple turns with delayed reward and other agents to reason about. We built an environment that targets that exact gap."

---

## Slide 3 — The World (0:30 → 0:55)

**Visual:** Four faction icons + four sector tiles + a 12-turn timeline.

**On-slide:**

> ## Neon Meridian
> 4 factions × 4 sectors × 6 missions × 12 turns
> - Factions: `ghostwire`, `iron_vultures`, `civic_shield`, `black_orchid`
> - Sectors: `docklands`, `data_spire`, `undergrid`, `citadel_gate`
> - 7-action verb space
> - Partially observable (rumors + scout)

**Speaker note (25s):**
"The agent is dropped into a cyberpunk city with a mission brief, a few rumors, and a threat level. To win, it has to negotiate alliances, manage four resources, deploy assets, run a coded operation, and secure extraction with the right message. Twelve turns. No restart."

---

## Slide 4 — Why It's Hard to Game (0:55 → 1:20)

**Visual:** A "five locks, one key" diagram — extraction succeeds only when all five gates align.

**On-slide:**

> ### Extraction succeeds only if **all** of:
> 1. Required allies present
> 2. Resource minimums met
> 3. Operation executed with the **right code**
> 4. Extraction message contains all keywords
> 5. Correct extraction sector

> Plus penalties for: repeated actions, premature ops, passive play under critical threat.

**Speaker note (25s):**
"This is what makes the benchmark anti-shortcut. You can't spam one action to climb the curve, because terminal success is gated on five independent conditions, and the dense shaping signal is bounded so reward hacking flatlines fast."

---

## Slide 5 — Reward Structure (1:20 → 1:40)

**Visual:** Stacked-weights bar chart.

**On-slide:**

```
task_score = 0.30 * alliances
           + 0.25 * resources
           + 0.20 * operation
           + 0.15 * message_quality
           + 0.10 * extraction
```

> Dense shaping every turn. Delayed completion bonus only on success.

**Speaker note (20s):**
"Weights chosen so coalition reasoning gets the highest signal — that's the capability we're targeting. Resource and op weights ensure mid-mission planning still matters. Message quality forces the model to *write*, not just act."

---

## Slide 6 — Live Demo (1:40 → 2:20)

**Visual:** Terminal capture / GIF — 7 API calls in 30 seconds, ending with `success: true`.

**On-slide:**

> ## Demo: Docklands Relay Hijack
> 7 API calls. 30 seconds. One clean extraction.
>
> Full storyboard → [`docs/demo_storyboard.md`](demo_storyboard.md)

**Speaker note (40s):**
"Watch what's on screen — agent scouts the target sector, negotiates with Ghostwire because the rumor warned us they hate loud ops, trades into intel, deploys the asset, runs OP-LANTERN, and extracts with a message that hits all three keywords. Mission cleared. Notice the threat dropped twice — that's the rumor-aware behavior we want."

> *(Cut to terminal recording or run [`scripts/run_full_pipeline.sh`](../scripts/run_full_pipeline.sh) live.)*

---

## Slide 7 — Training Pipeline (2:20 → 2:40)

**Visual:** Boxes — `Env API` → `TRL PPO` → `Reward curve`.

**On-slide:**

> ### Training
> - Model: `Qwen2.5-0.5B-Instruct`
> - Algorithm: TRL PPO
> - Driver: live HTTP calls against the env (no fake transitions)
>
> Reproduce in one line:
> ```bash
> ./scripts/run_full_pipeline.sh
> ```
>
> Colab: [one-click open](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb)

**Speaker note (20s):**
"We hit the live environment API directly — same one a judge can curl right now. One script spins up the server, trains, evaluates against random and heuristic baselines, and writes the curves we're about to show."

---

## Slide 8 — Evidence (2:40 → 3:00)

**Visual:** Two side-by-side plots — `artifacts/reward_curves.png` + `artifacts/loss_curve.png`.

**On-slide:**

> ![reward](../artifacts/reward_curves.png)
> ![loss](../artifacts/loss_curve.png)

> Random vs heuristic vs trained — full metrics in `artifacts/eval_metrics.jsonl`.

**Speaker note (20s):**
"Reward curve climbs as the policy learns to form alliances earlier, stop calling `run_operation` before deploy, and tighten extraction messages. Loss is well-behaved. Numbers replicable from the JSONL we ship."

---

## Slide 9 — What Makes This Different (3:00 → 3:20)

**Visual:** Comparison table.

**On-slide:**

| | Typical LLM env | Neon Syndicate |
|---|---|---|
| Episode length | 1–3 turns | up to 12 turns |
| Reward | terminal only | dense + delayed |
| Other agents | none | 4 factions w/ conflicting interests |
| Observability | full | partial (rumors + scout) |
| Reward hacking | possible | gated by 5-condition AND |

**Speaker note (20s):**
"Most LLM benchmarks test what the model already knows. This one tests whether it can *plan*."

---

## Slide 10 — Links + Ask (3:20 → 3:40)

**Visual:** QR codes for the four canonical links.

**On-slide:**

> ## Try it now
> - Hugging Face Space: <https://huggingface.co/spaces/hsbharadwaj/ev>
> - Runtime API: <https://hsbharadwaj-ev.hf.space>
> - Code: <https://github.com/hschinmayabharadwaj/Openev>
> - Colab: [one-click](https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb)
>
> **Judge flow:** [`docs/judge_flow.md`](judge_flow.md) — everything graded in 3 minutes.

**Speaker note (20s):**
"One-line health check: `curl https://hsbharadwaj-ev.hf.space/health`. Six tasks, all OpenEnv-compatible, all reproducible. Thanks."

---

## Backup Slides (Q&A)

### B1 — Why these 4 factions?

Each faction has a *deliberate* personality: Ghostwire = stealth, Iron Vultures = mercenary, Civic Shield = lawful, Black Orchid = political. The pairwise distrust matrix is what makes the medium and hard tasks hard — `task_hard_citywide_failsafe` requires Ghostwire and Civic Shield in the same coalition even though their default rumor says they don't trust each other.

### B2 — Why 12 turns specifically?

Long enough to require sequencing (scout → negotiate → trade → deploy → run → extract is already 6 turns minimum), short enough that the credit-assignment signal isn't lost in noise. Empirically the easy tasks resolve in 7–9 turns; hard tasks pressure right up to the limit.

### B3 — How do you prevent reward hacking?

Three layers:
1. Per-component progress is bounded `[0,1]` and dense reward is clamped per turn.
2. Penalties on repeated actions, premature ops, passive play.
3. Terminal `success` requires *five* independent gates aligning — alliances AND resources AND op AND message AND sector. Math doesn't let one component carry the score.

### B4 — Why does message quality matter?

It forces the model to *generate* in addition to *act*. The extraction message must contain all task-specific keywords (e.g. `window`, `relay`, `clean exit` for the easy docklands task). This is where pure RL-on-actions fails and an LLM-grounded policy wins.

### B5 — Reproducibility chain

```bash
git clone https://github.com/hschinmayabharadwaj/Openev
cd Openev
pip install -U pip && pip install -e .
uvicorn server.app:app --port 7860 &
python scripts/evaluate_and_plot.py --episodes 30
```

Three commands, end to end. No external accounts required for the env itself.

---

## Companion Documents

- **Mini-blog:** [`mini_blog.md`](mini_blog.md)
- **Judge flow (3-min walkthrough):** [`judge_flow.md`](judge_flow.md)
- **Live demo storyboard:** [`demo_storyboard.md`](demo_storyboard.md)
- **90-second video script:** [`video_script_90s.md`](video_script_90s.md)
