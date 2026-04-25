from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from models import Action, ResetRequest, StepResponse
from server.environment import NeonSyndicateEnvironment

app = FastAPI(title="Neon Syndicate OpenEnv", version="1.0.0")
env = NeonSyndicateEnvironment()

# Resolve the bundled landing page (animated training demo) once at import time.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LANDING_PAGE = _REPO_ROOT / "docs" / "visual_demo.html"


@app.get("/", response_class=HTMLResponse, response_model=None)
def root():
	"""Serve the animated training demo at the Space root URL.

	Without this route HF probes (and curious visitors) get a bare 404 even
	though the API is healthy. Returning the demo HTML doubles as a judge
	landing page.
	"""
	if _LANDING_PAGE.exists():
		return FileResponse(_LANDING_PAGE, media_type="text/html")
	return HTMLResponse(
		"<h1>Neon Syndicate OpenEnv</h1>"
		"<p>API is up. Try <code>/health</code>, <code>/tasks</code>, or <code>/docs</code>.</p>",
		status_code=200,
	)


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


def main() -> None:
	"""Entry point for the server."""
	uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()
