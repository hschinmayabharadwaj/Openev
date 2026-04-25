from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from models import Action, ResetRequest, StepResponse
from server.agent import (
	HeuristicPolicy,
	Policy,
	RandomPolicy,
	TrainedPolicy,
	get_policy,
)
from server.environment import NeonSyndicateEnvironment

app = FastAPI(title="Neon Syndicate OpenEnv", version="1.1.0")
env = NeonSyndicateEnvironment()

# Lazy singleton -- the heavy weights only load on the first /agent call.
_TRAINED = TrainedPolicy()

# Resolve bundled HTML pages once at import time.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LANDING_PAGE = _REPO_ROOT / "docs" / "visual_demo.html"
_GAME_PAGE = _REPO_ROOT / "docs" / "game_visualization.html"
_PLAY_PAGE = _REPO_ROOT / "docs" / "play.html"
_HEIST_PAGE = _REPO_ROOT / "docs" / "heist.html"
_JUDGE_PAGE = _REPO_ROOT / "docs" / "judge.html"
_LAB_PAGE = _REPO_ROOT / "docs" / "lab.html"
_WALKTHROUGH_PAGE = _REPO_ROOT / "docs" / "walkthrough.html"
_WALKTHROUGH_MD = _REPO_ROOT / "docs" / "walkthrough.md"
_ARTIFACTS_DIRS = [
	_REPO_ROOT / "notebooks" / "artifacts",
	_REPO_ROOT / "artifacts",
]
_ALLOWED_ARTIFACT_FILES = {
	"reward_curves.png",
	"loss_curve.png",
}

# Hackathon-grade public links. Mirrors README so the judge page is the
# single source of truth.
PUBLIC_LINKS = {
	"hf_space": "https://huggingface.co/spaces/hsbharadwaj/ev",
	"hf_runtime": "https://hsbharadwaj-ev.hf.space",
	"colab": "https://colab.research.google.com/github/hschinmayabharadwaj/Openev/blob/main/notebooks/trl_training_colab.ipynb",
	"github": "https://github.com/hschinmayabharadwaj/Openev",
	"openenv": "https://github.com/meta-pytorch/OpenEnv",
}


# ---------------------------------------------------------------------------
# Existing routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, response_model=None)
def root():
	"""Serve the animated training demo at the Space root URL."""
	if _LANDING_PAGE.exists():
		return FileResponse(_LANDING_PAGE, media_type="text/html")
	return HTMLResponse(
		"<h1>Neon Syndicate OpenEnv</h1>"
		"<p>API is up. Try <code>/play</code>, <code>/health</code>, "
		"<code>/tasks</code>, or <code>/docs</code>.</p>",
		status_code=200,
	)


@app.get("/game", response_class=HTMLResponse, response_model=None)
def game():
	"""Cyberpunk-styled scripted mission visualizer."""
	if _GAME_PAGE.exists():
		return FileResponse(_GAME_PAGE, media_type="text/html")
	return HTMLResponse("<p>Game visualization not found.</p>", status_code=404)


@app.get("/play", response_class=HTMLResponse, response_model=None)
def play():
	"""Live-play UI where the trained model executes missions in real time."""
	if _PLAY_PAGE.exists():
		return FileResponse(_PLAY_PAGE, media_type="text/html")
	return HTMLResponse("<p>Play UI not found.</p>", status_code=404)


@app.get("/heist", response_class=HTMLResponse, response_model=None)
def heist():
	"""Player-controlled heist: walk the city, the trained LLM is your Operator."""
	if _HEIST_PAGE.exists():
		return FileResponse(_HEIST_PAGE, media_type="text/html")
	return HTMLResponse("<p>Heist UI not found.</p>", status_code=404)


@app.get("/judge", response_class=HTMLResponse, response_model=None)
def judge():
	"""Single-page judge showcase. Bundles every required submission link
	(HF Space / Colab / GitHub) with live training evidence and policy
	comparison. Designed to be opened first by reviewers."""
	if _JUDGE_PAGE.exists():
		return FileResponse(_JUDGE_PAGE, media_type="text/html")
	return HTMLResponse("<p>Judge page not found.</p>", status_code=404)


@app.get("/lab", response_class=HTMLResponse, response_model=None)
def lab():
	"""Reward Forensics Lab: procedural task generator (RLVE knob),
	live 5-gate verifier timeline, and a reward-hacking sandbox a judge
	can fire pre-baked exploits at to see the env defenses kick in."""
	if _LAB_PAGE.exists():
		return FileResponse(_LAB_PAGE, media_type="text/html")
	return HTMLResponse("<p>Lab page not found.</p>", status_code=404)


@app.get("/walkthrough", response_class=HTMLResponse, response_model=None)
def walkthrough_html():
	"""Long-form walkthrough rendered with sidebar TOC + search +
	self-grade scorecard. Maps every concept from the OpenEnv briefing
	(RLVR, RLVE, reward hacking, curriculum, GRPO, process supervision)
	to where in this codebase it lives."""
	if _WALKTHROUGH_PAGE.exists():
		return FileResponse(_WALKTHROUGH_PAGE, media_type="text/html")
	return HTMLResponse("<p>Walkthrough page not found.</p>", status_code=404)


@app.get("/walkthrough.md", response_class=HTMLResponse, response_model=None)
def walkthrough_markdown():
	"""Raw markdown source of the walkthrough — same content as
	/walkthrough but readable on GitHub or by any markdown tool."""
	if _WALKTHROUGH_MD.exists():
		return FileResponse(_WALKTHROUGH_MD, media_type="text/markdown")
	raise HTTPException(status_code=404, detail="Walkthrough markdown missing")


@app.get("/api/reward_design")
def api_reward_design() -> dict:
	"""Machine-readable reward design summary. Used by the lab page to
	render the 5-gate rubric and the penalty table from the same source
	of truth as ``server/environment.py::_apply_action``."""
	return env.reward_design()


class GenerateTaskRequest(BaseModel):
	difficulty: int = Field(default=3, ge=1, le=5, description="1 (easy) … 5 (critical)")
	seed: Optional[int] = Field(default=None, description="Optional RNG seed")
	persist: bool = Field(
		default=True,
		alias="register",
		description="Register the generated task with the env so /reset can use it.",
	)

	model_config = {"populate_by_name": True}


@app.post("/api/generate_task")
def api_generate_task(req: GenerateTaskRequest) -> dict:
	"""RLVE knob: generate a brand-new mission at the requested difficulty.

	Returns the full ``TaskDefinition`` so the lab UI can render it,
	and (by default) registers it into the live env so a subsequent
	``POST /reset`` with that ``task_id`` works immediately.
	"""
	task = env.generate_procedural_task(difficulty=req.difficulty, seed=req.seed)
	if req.persist:
		env.register_procedural_task(task)
	return {
		"task": task.model_dump(),
		"registered": req.persist,
		"reset_with": f"POST /reset {{\"task_id\": \"{task.task_id}\"}}",
	}


@app.get("/links")
def links() -> dict:
	"""Stable, machine-readable list of public submission links."""
	return PUBLIC_LINKS


@app.get("/artifacts/training_summary")
def artifacts_training_summary() -> dict:
	"""Parse the most recent ``training_summary.jsonl`` and expose it as
	JSON so the judge page can render the live training curve client-side
	without depending on a static PNG.

	The notebook artifact is preferred (longer Qwen runs); the canonical
	repo artifact is used as fallback. Always returns a list, possibly
	empty, plus a ``source`` field saying where the data came from.
	"""
	candidates: List[Path] = [
		_REPO_ROOT / "notebooks" / "artifacts" / "trl-neon-model" / "training_summary.jsonl",
		_REPO_ROOT / "artifacts" / "trl-neon-model" / "training_summary.jsonl",
	]
	for path in candidates:
		if not path.is_file():
			continue
		episodes: List[Dict[str, Any]] = []
		try:
			with path.open("r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if not line:
						continue
					try:
						episodes.append(json.loads(line))
					except json.JSONDecodeError:
						continue
		except OSError:
			continue
		return {
			"source": str(path.relative_to(_REPO_ROOT)),
			"episode_count": len(episodes),
			"episodes": episodes,
		}
	return {"source": None, "episode_count": 0, "episodes": []}


@app.get("/artifacts/{filename}", response_model=None)
def artifacts_file(filename: str):
	"""Serve a small allowlist of artifacts (reward/loss PNGs) so the
	judge page can embed them without exposing the entire artifact tree.
	"""
	if filename not in _ALLOWED_ARTIFACT_FILES:
		raise HTTPException(status_code=404, detail="Not allowed")
	for base in _ARTIFACTS_DIRS:
		candidate = base / filename
		if candidate.is_file():
			return FileResponse(candidate)
	raise HTTPException(status_code=404, detail="Artifact not found")


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict:
	return {"tasks": [task.model_dump() for task in env.tasks()]}


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict:
	requested_task = request.task_id if request is not None else None
	try:
		observation = env.reset(task_id=requested_task)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	return {"observation": observation.model_dump()}


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
	try:
		return env.step(action)
	except RuntimeError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict:
	try:
		return {"state": env.state.model_dump()}
	except RuntimeError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------


class AgentActRequest(BaseModel):
	policy: str = Field(default="hybrid", description="trained | hybrid | heuristic | random")
	observation: Dict[str, Any]


@app.get("/agent/info")
def agent_info() -> dict:
	"""Tells the UI which policies are usable on this deployment."""
	return {
		"policies": ["hybrid", "trained", "heuristic", "random"],
		"default_policy": "hybrid",
		"trained": _TRAINED.info(),
		"policy_descriptions": {
			"hybrid": "Trained LLM proposes; env-rule guardrails reject obvious mistakes.",
			"trained": "Raw trained LLM. Honest but may fail on the partially-trained checkpoint.",
			"heuristic": "Hand-coded curriculum. Reliably wins easy missions; the safety baseline.",
			"random": "Random valid actions. Sanity floor for the leaderboard.",
		},
	}


@app.post("/agent/act")
def agent_act(request: AgentActRequest) -> dict:
	"""Single-shot inference. Returns the action plus the full reasoning trace."""
	try:
		policy = get_policy(request.policy, trained_singleton=_TRAINED)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	trace = policy.act_with_trace(request.observation)
	return trace.to_dict()


def _sse(event: str, data: Dict[str, Any]) -> str:
	"""Format one SSE message. Note the double newline terminator."""
	return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _run_episode(
	policy: Policy,
	episode_env: NeonSyndicateEnvironment,
	task_id: str,
	max_steps: int,
	lane_id: Optional[str] = None,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
	"""Generate (event_name, payload) tuples for one episode.

	If ``lane_id`` is provided, every payload carries it -- the race endpoint
	multiplexes several lanes onto a single SSE stream and the UI sorts
	by lane.
	"""

	def tag(payload: Dict[str, Any]) -> Dict[str, Any]:
		if lane_id is not None:
			return {"lane": lane_id, **payload}
		return payload

	observation_obj = episode_env.reset(task_id=task_id)
	observation = observation_obj.model_dump()
	yield "reset", tag({"observation": observation})

	cumulative = 0.0
	for turn in range(max_steps):
		trace = policy.act_with_trace(observation)
		yield "think", tag({"turn": turn, "trace": trace.to_dict()})

		try:
			action_model = Action(**trace.action)
		except Exception:
			action_model = Action(action_type="noop")

		step_response = episode_env.step(action_model)
		reward_score = float(step_response.reward.score)
		cumulative += reward_score
		observation = step_response.observation.model_dump()

		yield "step", tag(
			{
				"turn": turn,
				"action": action_model.model_dump(exclude_none=True),
				"observation": observation,
				"reward": step_response.reward.model_dump(),
				"info": step_response.info,
				"done": step_response.done,
				"cumulative_reward": cumulative,
			}
		)

		if step_response.done:
			info = step_response.info or {}
			yield "done", tag(
				{
					"turn": turn,
					"success": bool(info.get("success", False)),
					"task_score": float(info.get("task_score", 0.0)),
					"cumulative_reward": cumulative,
					"steps": turn + 1,
				}
			)
			return

	final_score = 0.0
	try:
		final_score = float(episode_env._grader_score(episode_env.state))  # type: ignore[attr-defined]
	except Exception:
		final_score = 0.0
	yield "done", tag(
		{
			"turn": max_steps - 1,
			"success": False,
			"task_score": final_score,
			"cumulative_reward": cumulative,
			"steps": max_steps,
			"reason": "max_steps",
		}
	)


@app.get("/agent/episode")
def agent_episode(
	task_id: str = Query(...),
	policy: str = Query("hybrid"),
	max_steps: int = Query(12, ge=1, le=24),
) -> StreamingResponse:
	"""Stream a full episode as SSE events. Used by the live play UI."""
	try:
		policy_impl = get_policy(policy, trained_singleton=_TRAINED)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	episode_env = NeonSyndicateEnvironment()

	def gen() -> Iterable[str]:
		yield _sse(
			"meta",
			{
				"task_id": task_id,
				"policy": policy,
				"max_steps": max_steps,
				"trained_available": _TRAINED.available,
				"trained_info": _TRAINED.info(),
			},
		)
		try:
			for event_name, payload in _run_episode(
				policy_impl, episode_env, task_id, max_steps
			):
				yield _sse(event_name, payload)
		except Exception as exc:  # pragma: no cover - defensive
			yield _sse("error", {"message": repr(exc)})

	return StreamingResponse(
		gen(),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
	)


@app.get("/agent/race")
def agent_race(
	task_id: str = Query(...),
	policies: str = Query("trained,heuristic,random"),
	max_steps: int = Query(12, ge=1, le=24),
) -> StreamingResponse:
	"""Run several policies on the same task in lock-step lanes.

	Each lane has its own ``NeonSyndicateEnvironment`` (the env keeps internal
	state). We round-robin one turn per lane so SSE consumers see lanes
	progress in lock-step, perfect for a side-by-side leaderboard.
	"""
	requested = [p.strip().lower() for p in policies.split(",") if p.strip()]
	if not requested:
		raise HTTPException(status_code=400, detail="no policies provided")

	lanes: List[Dict[str, Any]] = []
	for name in requested:
		try:
			lane_policy = get_policy(name, trained_singleton=_TRAINED)
		except ValueError as exc:
			raise HTTPException(status_code=400, detail=str(exc)) from exc
		lane_env = NeonSyndicateEnvironment()
		gen_iter = _run_episode(lane_policy, lane_env, task_id, max_steps, lane_id=name)
		lanes.append({"id": name, "policy": lane_policy, "iter": gen_iter, "done": False})

	def stream() -> Iterable[str]:
		yield _sse(
			"race_meta",
			{
				"task_id": task_id,
				"max_steps": max_steps,
				"lanes": [{"id": lane["id"], "label": lane["id"].title()} for lane in lanes],
				"trained_available": _TRAINED.available,
				"trained_info": _TRAINED.info(),
			},
		)
		# Round-robin draining: yield one event from each lane, looping until all done.
		while not all(lane["done"] for lane in lanes):
			for lane in lanes:
				if lane["done"]:
					continue
				try:
					event_name, payload = next(lane["iter"])
				except StopIteration:
					lane["done"] = True
					continue
				except Exception as exc:  # pragma: no cover - defensive
					yield _sse("error", {"lane": lane["id"], "message": repr(exc)})
					lane["done"] = True
					continue
				yield _sse(event_name, payload)
				if event_name == "done":
					lane["done"] = True
		yield _sse("race_done", {"task_id": task_id})

	return StreamingResponse(
		stream(),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
	)


def main() -> None:
	"""Entry point for the server."""
	uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()
