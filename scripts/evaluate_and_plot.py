from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import httpx
import matplotlib.pyplot as plt


FACTIONS = ["ghostwire", "iron_vultures", "civic_shield", "black_orchid"]
SECTORS = ["docklands", "data_spire", "undergrid", "citadel_gate"]
RESOURCES = ["credits", "intel", "influence", "energy"]
OPS = ["OP-LANTERN", "OP-PRISM", "OP-NIGHTLOCK", "OP-HALO", "OP-OBSIDIAN", "OP-DAWNFALL"]


@dataclass
class EpisodeMetric:
    policy: str
    episode: int
    task_id: str
    total_reward: float
    final_task_score: float
    success: bool


class OpenEnvClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def tasks(self) -> List[Dict[str, Any]]:
        resp = self.client.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()["tasks"]

    def reset(self, task_id: str) -> Dict[str, Any]:
        resp = self.client.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()["observation"]

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.client.post(f"{self.base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()


def random_policy(_obs: Dict[str, Any]) -> Dict[str, Any]:
    action_type = random.choice(
        [
            "scout_sector",
            "negotiate_pact",
            "trade_resources",
            "deploy_asset",
            "run_operation",
            "secure_extraction",
            "noop",
        ]
    )
    if action_type == "scout_sector":
        return {"action_type": action_type, "sector": random.choice(SECTORS)}
    if action_type == "negotiate_pact":
        return {"action_type": action_type, "faction": random.choice(FACTIONS)}
    if action_type == "trade_resources":
        return {
            "action_type": action_type,
            "resource": random.choice(RESOURCES),
            "amount": random.randint(1, 25),
        }
    if action_type == "deploy_asset":
        return {"action_type": action_type, "sector": random.choice(SECTORS)}
    if action_type == "run_operation":
        return {"action_type": action_type, "operation_code": random.choice(OPS)}
    if action_type == "secure_extraction":
        return {
            "action_type": action_type,
            "sector": random.choice(SECTORS),
            "message": "window relay clean exit",
        }
    return {"action_type": "noop"}


def heuristic_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    alliances = obs.get("alliances", [])
    resources = obs.get("resources", {})
    operation_ready = bool(obs.get("operation_ready"))
    operation_executed = bool(obs.get("operation_executed"))

    if len(alliances) < 1:
        return {"action_type": "negotiate_pact", "faction": "ghostwire"}
    if len(alliances) < 2 and resources.get("influence", 0) >= 8:
        return {"action_type": "negotiate_pact", "faction": "civic_shield"}
    if resources.get("intel", 0) < 55:
        return {"action_type": "scout_sector", "sector": "undergrid"}
    if resources.get("energy", 0) < 30:
        return {"action_type": "trade_resources", "resource": "energy", "amount": 20}
    if not operation_ready:
        return {"action_type": "deploy_asset", "sector": "undergrid"}
    if not operation_executed:
        return {"action_type": "run_operation", "operation_code": "OP-NIGHTLOCK"}
    return {
        "action_type": "secure_extraction",
        "sector": "undergrid",
        "message": "stabilized fallback undergrid clean exit",
    }


def run_policy(
    client: OpenEnvClient,
    policy_name: str,
    episodes: int,
    max_steps: int,
) -> List[EpisodeMetric]:
    task_list = client.tasks()
    metrics: List[EpisodeMetric] = []

    for episode in range(episodes):
        task = task_list[episode % len(task_list)]
        task_id = task["task_id"]
        obs = client.reset(task_id)

        total_reward = 0.0
        final_task_score = 0.0
        success = False

        for _ in range(max_steps):
            action = random_policy(obs) if policy_name == "random" else heuristic_policy(obs)
            payload = client.step(action)
            total_reward += float(payload["reward"]["score"])
            obs = payload["observation"]
            if payload["done"]:
                info = payload.get("info", {})
                final_task_score = float(info.get("task_score", 0.0))
                success = bool(info.get("success", False))
                break

        metrics.append(
            EpisodeMetric(
                policy=policy_name,
                episode=episode,
                task_id=task_id,
                total_reward=total_reward,
                final_task_score=final_task_score,
                success=success,
            )
        )

    return metrics


def save_metrics(metrics: List[EpisodeMetric], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row.__dict__) + "\n")


def plot_metrics(metrics: List[EpisodeMetric], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[EpisodeMetric]] = {"random": [], "heuristic": []}
    for m in metrics:
        grouped[m.policy].append(m)

    plt.figure(figsize=(10, 6))
    for policy, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r.episode)
        xs = [r.episode for r in rows]
        ys = [r.total_reward for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"{policy} total reward")

    plt.title("Neon Syndicate: Reward Curves by Policy")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)


def print_summary(metrics: List[EpisodeMetric]) -> None:
    for policy in ["random", "heuristic"]:
        rows = [m for m in metrics if m.policy == policy]
        avg_reward = sum(r.total_reward for r in rows) / max(1, len(rows))
        avg_score = sum(r.final_task_score for r in rows) / max(1, len(rows))
        success_rate = sum(1 for r in rows if r.success) / max(1, len(rows))
        print(
            f"[EVAL] policy={policy} episodes={len(rows)} "
            f"avg_total_reward={avg_reward:.3f} avg_task_score={avg_score:.3f} success_rate={success_rate:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies and generate judge-friendly reward plots.")
    parser.add_argument("--env-base-url", type=str, default="http://localhost:7860")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output-jsonl", type=str, default="artifacts/eval_metrics.jsonl")
    parser.add_argument("--output-png", type=str, default="artifacts/reward_curves.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    client = OpenEnvClient(args.env_base_url)
    try:
        random_metrics = run_policy(client, "random", args.episodes, args.max_steps)
        heuristic_metrics = run_policy(client, "heuristic", args.episodes, args.max_steps)
        all_metrics = random_metrics + heuristic_metrics

        save_metrics(all_metrics, Path(args.output_jsonl))
        plot_metrics(all_metrics, Path(args.output_png))
        print_summary(all_metrics)
        print(f"[ARTIFACT] metrics={args.output_jsonl}")
        print(f"[ARTIFACT] plot={args.output_png}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
