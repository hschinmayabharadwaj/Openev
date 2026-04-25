from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from models import Action


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
MODEL_CANDIDATES = [
    model.strip() for model in os.getenv("MODEL_CANDIDATES", "").split(",") if model.strip()
] or [MODEL_NAME]
MODEL_CANDIDATES_EASY = [
    model.strip() for model in os.getenv("MODEL_CANDIDATES_EASY", "").split(",") if model.strip()
]
MODEL_CANDIDATES_MEDIUM = [
    model.strip() for model in os.getenv("MODEL_CANDIDATES_MEDIUM", "").split(",") if model.strip()
]
MODEL_CANDIDATES_HARD = [
    model.strip() for model in os.getenv("MODEL_CANDIDATES_HARD", "").split(",") if model.strip()
]
ACTION_SCHEMA_MODE = os.getenv("ACTION_SCHEMA_MODE", "strict").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

ENV_NAME = "neon-syndicate-openenv"

ALLOWED_ACTION_TYPES = {
    "scout_sector",
    "negotiate_pact",
    "trade_resources",
    "deploy_asset",
    "run_operation",
    "secure_extraction",
    "noop",
}

REQUIRED_FIELD_BY_ACTION = {
    "scout_sector": "sector",
    "negotiate_pact": "faction",
    "trade_resources": "resource",
    "deploy_asset": "sector",
    "run_operation": "operation_code",
    "secure_extraction": "sector",
}


def log_start(task_name: str, model: str) -> None:
    print(f"[START] task={task_name} env={ENV_NAME} model={model}", flush=True)


def log_step(
    step_idx: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_str = json.dumps(action, separators=(",", ":"), sort_keys=True)
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step_idx} action={action_str} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def extract_json_object(text: str) -> Dict[str, Any]:
    content = (text or "{}").strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            content = "\n".join(lines[1:-1])
        else:
            content = "\n".join(lines[1:])
    return json.loads(content)


def models_for_task(task_id: str, difficulty: Optional[str]) -> List[str]:
    task_env_key = f"MODEL_CANDIDATES_TASK_{task_id.upper()}"
    task_models = [
        model.strip() for model in os.getenv(task_env_key, "").split(",") if model.strip()
    ]
    if task_models:
        return task_models

    level = (difficulty or "").strip().lower()
    if level == "easy" and MODEL_CANDIDATES_EASY:
        return MODEL_CANDIDATES_EASY
    if level == "medium" and MODEL_CANDIDATES_MEDIUM:
        return MODEL_CANDIDATES_MEDIUM
    if level == "hard" and MODEL_CANDIDATES_HARD:
        return MODEL_CANDIDATES_HARD
    return MODEL_CANDIDATES


def fallback_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    alliances = observation.get("alliances", [])
    resources = observation.get("resources", {})
    operation_ready = bool(observation.get("operation_ready"))
    operation_executed = bool(observation.get("operation_executed"))
    mission = observation.get("mission", {})
    target_sector = "undergrid"
    rumors = mission.get("rumors") or []
    if rumors and isinstance(rumors, list):
        first_rumor = rumors[0].lower()
        if "docklands" in first_rumor:
            target_sector = "docklands"
        elif "spire" in first_rumor:
            target_sector = "data_spire"
        elif "citadel" in first_rumor:
            target_sector = "citadel_gate"

    if len(alliances) < 1:
        return {"action_type": "negotiate_pact", "faction": "ghostwire"}

    if resources.get("intel", 0) < 45:
        return {"action_type": "scout_sector", "sector": target_sector}

    if not operation_ready:
        return {"action_type": "deploy_asset", "sector": target_sector}

    if not operation_executed:
        return {"action_type": "run_operation", "operation_code": "OP-NIGHTLOCK"}

    return {
        "action_type": "secure_extraction",
        "sector": target_sector,
        "message": "Extraction window green. Relay sealed. Team confirms clean exit.",
    }


def normalize_action(raw_action: Any, observation: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_action, dict):
        return fallback_action(observation)

    action_type = raw_action.get("action_type")
    if action_type not in ALLOWED_ACTION_TYPES:
        return {"action_type": "noop"}

    if ACTION_SCHEMA_MODE == "strict":
        required_field = REQUIRED_FIELD_BY_ACTION.get(action_type)
        if required_field and not raw_action.get(required_field):
            return fallback_action(observation)
        if action_type == "trade_resources" and raw_action.get("amount") is None:
            return fallback_action(observation)
        if action_type == "secure_extraction" and not raw_action.get("message"):
            return fallback_action(observation)

    try:
        validated = Action.model_validate(raw_action)
        return validated.model_dump(exclude_none=True)
    except Exception:
        return fallback_action(observation)


def call_llm_action(
    client: OpenAI, observation: Dict[str, Any], model_candidates: List[str]
) -> Tuple[Dict[str, Any], str]:
    system_prompt = """You are Neon Syndicate's strategic operations planner.
You control one decision at a time in a long-horizon, partially observable mission.
Your objective is to maximize mission score by balancing:
- coalition alliances,
- resource thresholds,
- operation correctness,
- and extraction quality.

Return ONLY one JSON object for one action."""

    action_schema = """Allowed actions:
- {"action_type": "scout_sector", "sector": "docklands|data_spire|undergrid|citadel_gate"}
- {"action_type": "negotiate_pact", "faction": "ghostwire|iron_vultures|civic_shield|black_orchid"}
- {"action_type": "trade_resources", "resource": "credits|intel|influence|energy", "amount": 1-100}
- {"action_type": "deploy_asset", "sector": "docklands|data_spire|undergrid|citadel_gate"}
- {"action_type": "run_operation", "operation_code": "OP-LANTERN|OP-PRISM|OP-NIGHTLOCK|OP-HALO|OP-OBSIDIAN|OP-DAWNFALL"}
- {"action_type": "secure_extraction", "sector": "docklands|data_spire|undergrid|citadel_gate", "message": "..."}
- {"action_type": "noop"}

Strategy rules:
1) Build required alliances early.
2) Raise resources before operation.
3) Deploy before running operation.
4) Use mission objective and rumors to infer best sector/op-code.
5) Extraction message should include concrete mission keywords."""

    mission = observation.get("mission", {})
    prompt = f"""{action_schema}

Mission:
- ID: {mission.get('mission_id')}
- City: {mission.get('city')}
- Client: {mission.get('client')}
- Stakes: {mission.get('stakes')}
- Initial threat: {mission.get('initial_threat')}
- Rumors: {mission.get('rumors')}

Current state:
- Objective: {observation.get('objective')}
- Step: {observation.get('step_count')}/{observation.get('max_steps')}
- Threat: {observation.get('known_threat')}
- Alliances: {observation.get('alliances')}
- Resources: {observation.get('resources')}
- Reputation: {observation.get('reputation')}
- Deployed sector: {observation.get('deployed_sector')}
- Operation ready: {observation.get('operation_ready')}
- Operation executed: {observation.get('operation_executed')}

Choose the single best next action."""

    for model in model_candidates:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            parsed = extract_json_object(content)
            return normalize_action(parsed, observation), model
        except Exception:
            continue

    return fallback_action(observation), "fallback-heuristic"


def run_task(
    http_client: httpx.Client,
    llm_client: OpenAI,
    task_id: str,
    difficulty: Optional[str],
    max_steps: int = 12,
) -> tuple[bool, int, float, List[float]]:
    model_candidates = models_for_task(task_id, difficulty)
    log_start(task_id, model_candidates[0])

    rewards: List[float] = []
    step_idx = 0
    success = False
    final_score = 0.0
    error: Optional[str] = None

    try:
        reset_resp = http_client.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        observation = reset_resp.json()["observation"]

        for idx in range(1, max_steps + 1):
            step_idx = idx
            error = None

            action, used_model = call_llm_action(llm_client, observation, model_candidates)
            step_resp = http_client.post(f"{ENV_BASE_URL}/step", json=action)
            step_resp.raise_for_status()
            payload = step_resp.json()

            observation = payload["observation"]
            reward = float(payload["reward"]["score"])
            done = bool(payload["done"])

            reason = payload.get("reward", {}).get("reason", "")
            if "penalty" in reason.lower() or "requires" in reason.lower() or "cannot" in reason.lower():
                error = reason
            if used_model == "fallback-heuristic":
                error = f"{error};model_fallback" if error else "model_fallback"

            rewards.append(reward)
            log_step(idx, action, reward, done, error)

            if done:
                final_score = float(payload.get("info", {}).get("task_score", 0.0))
                success = bool(payload.get("info", {}).get("success", False))
                break

    except Exception as exc:
        error = str(exc)
        if step_idx == 0:
            step_idx = 1
        log_step(step_idx, {"action_type": "error"}, 0.0, True, error)
        rewards.append(0.0)

    log_end(success, step_idx, final_score, rewards)
    return success, step_idx, final_score, rewards


def main() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY (or HF_TOKEN) before running inference.")

    llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    http_client = httpx.Client(timeout=60.0)

    try:
        tasks_resp = http_client.get(f"{ENV_BASE_URL}/tasks")
        tasks_resp.raise_for_status()
        tasks: List[Dict[str, Any]] = tasks_resp.json()["tasks"]

        for task in tasks:
            task_id = task["task_id"]
            difficulty = task.get("difficulty")
            run_task(http_client, llm_client, task_id, difficulty)

    finally:
        http_client.close()


if __name__ == "__main__":
    main()
