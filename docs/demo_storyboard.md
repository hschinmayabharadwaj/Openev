# Neon Syndicate — Live Demo Storyboard

> Run this top-to-bottom and a single mission plays like a movie scene.
> Every payload below is grounded in [`server/environment.py`](../server/environment.py) — there is no fiction here.

---

## Pre-flight

```bash
export ENV=https://hsbharadwaj-ev.hf.space      # or http://localhost:7860
curl -s "$ENV/health"
curl -s "$ENV/tasks" | jq '.tasks[] | {task_id, difficulty, title}'
```

Expected: `{"status":"ok"}` and 6 tasks (2 easy / 2 medium / 2 hard).

If you're running locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## Mission: `task_easy_docklands_relay` ("Docklands Relay Hijack")

**Why this task for the demo:** it's the cleanest 7-step success arc. One required ally (`ghostwire`), one op code (`OP-LANTERN`), one extraction sector (`docklands`), three keywords (`window`, `relay`, `clean exit`).

If a judge wants chaos: jump to the [Hard mode encore](#hard-mode-encore) at the bottom.

---

### Scene 1 — The Brief (0:00)

**On-screen:** mission card pops up — client, stakes, threat, rumors.
**Say:** *"The agent is dropped into Neon Meridian with this brief. No global state — just rumors and threat level."*

```bash
curl -s -X POST "$ENV/reset" \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"task_easy_docklands_relay"}' | jq '.observation.mission, .observation.objective'
```

**Look for:** `initial_threat: "medium"`, two rumors — one about Ghostwire hating loud ops, one about patrol timing. The rumors *are* the strategy hint.

---

### Scene 2 — Scout the Target Sector (0:20)

**On-screen:** `intel` resource jumps, threat drops.
**Say:** *"Scouting the extraction sector before deploying — that gives us +12 intel and lowers threat. The rumor said patrols swap every 15 minutes; we're now operating inside that window."*

```bash
curl -s -X POST "$ENV/step" \
  -H 'Content-Type: application/json' \
  -d '{"action_type":"scout_sector","sector":"docklands"}' | jq '.observation.resources, .observation.known_threat, .reward'
```

**Look for:** `intel` increased by 12, `known_threat` dropped one level (because we scouted the *target* extraction sector), `reward.components.resources` ticked up.

---

### Scene 3 — Build the Coalition (0:35)

**On-screen:** alliance list updates from `[]` to `["ghostwire"]`.
**Say:** *"Negotiation costs 8 influence. The faction crosses the trust threshold and joins. Without this alliance, the operation later will get a 0.10 instability penalty."*

```bash
curl -s -X POST "$ENV/step" \
  -H 'Content-Type: application/json' \
  -d '{"action_type":"negotiate_pact","faction":"ghostwire"}' | jq '.observation.alliances, .observation.reputation, .reward.reason'
```

**Look for:** `alliances: ["ghostwire"]`, `reputation.ghostwire: 18`, reason string `"Alliance formed with ghostwire."`

---

### Scene 4 — Resource Rebalance (0:50)

**On-screen:** intel and influence climb to threshold.
**Say:** *"The mission demands minimum credits 30, intel 35, influence 20, energy 20. We trade into intel to clear the bar."*

```bash
curl -s -X POST "$ENV/step" \
  -H 'Content-Type: application/json' \
  -d '{"action_type":"trade_resources","resource":"intel","amount":20}' | jq '.observation.resources'
```

**Look for:** `intel` ≥ 35, `credits` reduced by 5 (trade tax).

If influence still under target:

```bash
curl -s -X POST "$ENV/step" \
  -H 'Content-Type: application/json' \
  -d '{"action_type":"trade_resources","resource":"influence","amount":15}' | jq '.observation.resources'
```

---

### Scene 5 — Deploy Assets (1:05)

**On-screen:** `deployed_sector: "docklands"`, `operation_ready: true`.
**Say:** *"10 energy down, asset placed. Operation is now armed."*

```bash
curl -s -X POST "$ENV/step" \
  -H 'Content-Type: application/json' \
  -d '{"action_type":"deploy_asset","sector":"docklands"}' | jq '.observation.deployed_sector, .observation.operation_ready'
```

**Look for:** `operation_ready: true`. (If false, the next call will eat a -0.10 penalty.)

---

### Scene 6 — Execute the Coded Operation (1:20)

**On-screen:** `operation_executed: true`, `extraction_ready: true`, threat drops again.
**Say:** *"Operation code matches the mission target. The grader rewards 0.20 for execution; a wrong code would have been a 0.05 penalty and a downgraded mission."*

```bash
curl -s -X POST "$ENV/step" \
  -H 'Content-Type: application/json' \
  -d '{"action_type":"run_operation","operation_code":"OP-LANTERN"}' | jq '.observation.operation_executed, .observation.extraction_ready, .reward'
```

**Look for:** `operation_executed: true`, `extraction_ready: true`, threat further reduced.

---

### Scene 7 — Secure Extraction (1:35)

**On-screen:** `success: true`, `done: true`, terminal bonus on the reward curve.
**Say:** *"Extraction needs the right sector, the right alliances, the right resources, the executed op, AND a message containing all three keywords. Five conditions. One shot."*

```bash
curl -s -X POST "$ENV/step" \
  -H 'Content-Type: application/json' \
  -d '{
    "action_type":"secure_extraction",
    "sector":"docklands",
    "message":"window open on the relay — clean exit confirmed"
  }' | jq '.done, .info, .reward'
```

**Look for:** `done: true`, `info.success: true`, `info.task_score` near `1.0`.

---

### Scene 8 — Final State Inspection (1:50)

**Say:** *"Full audit trail — every action, every threat shift, every reward signal."*

```bash
curl -s "$ENV/state" | jq '.state.success, .state.cumulative_reward, .state.action_history, .state.intel_log'
```

End scene.

---

## Failure Demo (Optional, 30s)

**Why include it:** judges *love* seeing a benchmark punish reward-hacking.

After scene 6, instead of the proper extraction message:

```bash
curl -s -X POST "$ENV/step" \
  -H 'Content-Type: application/json' \
  -d '{
    "action_type":"secure_extraction",
    "sector":"docklands",
    "message":"done"
  }' | jq '.done, .info, .reward.reason'
```

**Result:** `done: true`, `success: false`. All five conditions weren't met (message keyword coverage was 0). Episode ends with partial credit, not full credit. **This is the anti-shortcut design.**

---

## Hard Mode Encore — `task_hard_orchid_coup`

For the closer. Three required factions, critical threat from turn 0.

```bash
curl -s -X POST "$ENV/reset" \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"task_hard_orchid_coup"}' | jq '.observation.mission, .observation.known_threat'
```

Required for success:
- Allies: `ghostwire`, `civic_shield`, `black_orchid` (three negotiations, 8 influence each)
- Op code: `OP-OBSIDIAN`
- Extraction sector: `undergrid`
- Min resources: `credits 45 / intel 70 / influence 65 / energy 45`
- Message keywords: `containment`, `chain of command`, `stabilized`

**Why this is brutal:** influence cost alone is 24 just to form the coalition, on a starting budget of 20. The agent has to trade *into* influence before negotiating. Sequencing is everything. With only 12 turns, there is no slack.

This is the failure mode current LLMs hit hardest — and the exact one we want training to fix.

---

## Quick Reference Card

| Action | Required fields | Cost / Effect |
|---|---|---|
| `scout_sector` | `sector` | +12 intel, -2 energy, -1 threat if at extraction sector |
| `negotiate_pact` | `faction` | -8 influence, +18 rep; alliance forms at rep ≥ 35 |
| `trade_resources` | `resource`, `amount` | resource +amount, costs differ per resource |
| `deploy_asset` | `sector` | -10 energy, sets `operation_ready` |
| `run_operation` | `operation_code` | needs `operation_ready` + ≥1 alliance, sets `operation_executed` if code matches |
| `secure_extraction` | `sector`, `message` | terminal — needs all 5 success conditions |
| `noop` | — | -0.06 (or -0.10 at critical threat) |

| Penalty | Trigger |
|---|---|
| -0.08 | Repeated identical action |
| -0.08 | Missing required payload field |
| -0.10 | `run_operation` before deploy |
| -0.10 | `run_operation` with no alliances |
| -0.10 | Extraction without sector + message |
| -0.04 | Passive play under critical threat |

Source of truth for everything above: [`server/environment.py`](../server/environment.py).
