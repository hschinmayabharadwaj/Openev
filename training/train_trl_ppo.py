from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


ALLOWED_ACTION_TYPES = {
    "scout_sector",
    "negotiate_pact",
    "trade_resources",
    "deploy_asset",
    "run_operation",
    "secure_extraction",
    "noop",
}


@dataclass
class EpisodeResult:
    episode_id: int
    task_id: str
    total_reward: float
    final_task_score: float
    success: bool
    steps: int


class OpenEnvEpisodeRunner:
    def __init__(self, env_base_url: str, timeout: float = 30.0) -> None:
        self.env_base_url = env_base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def tasks(self) -> List[Dict[str, Any]]:
        resp = self.client.get(f"{self.env_base_url}/tasks")
        resp.raise_for_status()
        return resp.json()["tasks"]

    def reset(self, task_id: str) -> Dict[str, Any]:
        resp = self.client.post(f"{self.env_base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()["observation"]

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.client.post(f"{self.env_base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()


def build_prompt(observation: Dict[str, Any]) -> str:
    mission = observation.get("mission", {})
    return (
        "You are a strategic planner in Neon Syndicate. Return only one JSON action.\n"
        "Goal: maximize mission completion score in this partially observable environment.\n\n"
        f"Task: {observation.get('task_id')} ({observation.get('difficulty')})\n"
        f"Objective: {observation.get('objective')}\n"
        f"Step: {observation.get('step_count')}/{observation.get('max_steps')}\n"
        f"Threat: {observation.get('known_threat')}\n"
        f"Mission ID: {mission.get('mission_id')}\n"
        f"Stakes: {mission.get('stakes')}\n"
        f"Rumors: {mission.get('rumors')}\n"
        f"Resources: {observation.get('resources')}\n"
        f"Reputation: {observation.get('reputation')}\n"
        f"Alliances: {observation.get('alliances')}\n"
        f"Deployed Sector: {observation.get('deployed_sector')}\n"
        f"Operation Ready: {observation.get('operation_ready')}\n"
        f"Operation Executed: {observation.get('operation_executed')}\n"
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    content = (text or "{}").strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            content = "\n".join(lines[1:-1])
        else:
            content = "\n".join(lines[1:])
    return json.loads(content)


def heuristic_fallback(observation: Dict[str, Any]) -> Dict[str, Any]:
    alliances = observation.get("alliances", [])
    resources = observation.get("resources", {})

    if len(alliances) < 1:
        return {"action_type": "negotiate_pact", "faction": "ghostwire"}
    if resources.get("intel", 0) < 45:
        return {"action_type": "scout_sector", "sector": "undergrid"}
    if not observation.get("operation_ready"):
        return {"action_type": "deploy_asset", "sector": "undergrid"}
    if not observation.get("operation_executed"):
        return {"action_type": "run_operation", "operation_code": "OP-NIGHTLOCK"}
    return {
        "action_type": "secure_extraction",
        "sector": "undergrid",
        "message": "Extraction window green. Fallback stable. Team confirms clean exit.",
    }


def normalize_action(action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return heuristic_fallback(observation)
    action_type = action.get("action_type")
    if action_type not in ALLOWED_ACTION_TYPES:
        return {"action_type": "noop"}
    return action


def select_task(task_list: List[Dict[str, Any]], idx: int) -> str:
    if not task_list:
        raise RuntimeError("No tasks returned by environment")
    return task_list[idx % len(task_list)]["task_id"]


def run_training(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    runner = OpenEnvEpisodeRunner(args.env_base_url)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)

    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        log_with=None,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    all_tasks = runner.tasks()
    results: List[EpisodeResult] = []

    for episode in range(args.episodes):
        task_id = select_task(all_tasks, episode)
        observation = runner.reset(task_id)
        total_reward = 0.0
        final_task_score = 0.0
        success = False

        for _ in range(args.max_steps):
            prompt = build_prompt(observation)
            query_tensor = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

            response_tensor = trainer.generate(
                query_tensor,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            ).squeeze(0)

            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            try:
                parsed_action = extract_json_object(response_text)
                action = normalize_action(parsed_action, observation)
            except Exception:
                action = heuristic_fallback(observation)

            step_payload = runner.step(action)
            reward_value = float(step_payload["reward"]["score"])
            done = bool(step_payload["done"])

            trainer.step(
                [query_tensor],
                [response_tensor],
                [torch.tensor(reward_value, dtype=torch.float32)],
            )

            total_reward += reward_value
            observation = step_payload["observation"]

            if done:
                info = step_payload.get("info", {})
                final_task_score = float(info.get("task_score", 0.0))
                success = bool(info.get("success", False))
                break

        results.append(
            EpisodeResult(
                episode_id=episode,
                task_id=task_id,
                total_reward=total_reward,
                final_task_score=final_task_score,
                success=success,
                steps=int(observation.get("step_count", 0)),
            )
        )

        print(
            f"[TRAIN] episode={episode} task={task_id} total_reward={total_reward:.3f} "
            f"task_score={final_task_score:.3f} success={success}",
            flush=True,
        )

    output_dir = args.output_dir.rstrip("/")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary_path = f"{output_dir}/training_summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.__dict__) + "\n")

    avg_reward = sum(r.total_reward for r in results) / max(1, len(results))
    avg_score = sum(r.final_task_score for r in results) / max(1, len(results))
    success_rate = sum(1 for r in results if r.success) / max(1, len(results))

    print(
        f"[SUMMARY] episodes={len(results)} avg_total_reward={avg_reward:.3f} "
        f"avg_task_score={avg_score:.3f} success_rate={success_rate:.3f}",
        flush=True,
    )

    runner.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TRL PPO training loop for Neon Syndicate.")
    parser.add_argument("--env-base-url", type=str, default="http://localhost:7860")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/trl-neon-model")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
