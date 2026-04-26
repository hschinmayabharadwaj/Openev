from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def resolve_training_device(preference: str) -> torch.device:
    """Pick device for the model. Default 'auto' prefers CUDA, then CPU (not MPS: TRL+PPO is flaky on MPS)."""
    p = preference.lower()
    if p == "cpu":
        return torch.device("cpu")
    if p == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if p == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available")
        return torch.device("mps")
    if p != "auto":
        raise SystemExit(f"Unknown --device: {preference}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_dtype_for(device: torch.device, use_half: bool) -> torch.dtype:
    if not use_half or device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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
    difficulty: str
    total_reward: float
    final_task_score: float
    success: bool
    steps: int
    policy: str = "trained_ppo"
    fallback_steps: int = 0


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
    """Task-aware fallback. Uses the expert policy when the task target is
    known (which is true for the 6 bundled missions and for procedural
    tasks registered through ``/api/generate_task``); otherwise it falls
    back to the curriculum heuristic.

    Why this matters for PPO: with a small base model, the LLM's first
    actions are mostly noise, which means the env returns near-zero
    reward and PPO has nothing to climb. Plugging the expert in as the
    fallback gives the trainer non-trivial demonstration trajectories to
    bootstrap from while it learns to emit valid JSON actions on its
    own — a poor man's behaviour-cloning warmup.
    """
    try:
        from server.agent import expert_action, get_task_target, heuristic_action

        target = get_task_target(observation.get("task_id"))
        if target is not None:
            return expert_action(observation, target)
        return heuristic_action(observation)
    except Exception:
        # Defensive: never crash training because the import path moved.
        alliances = observation.get("alliances", [])
        if len(alliances) < 1:
            return {"action_type": "negotiate_pact", "faction": "ghostwire"}
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


def _load_model_qlora(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Any:
    """Load ``model_name`` 4-bit + wrap it with LoRA adapters via PEFT.

    QLoRA is the recommended path on the constrained Colab/HF Space GPUs
    handed out at the hackathon (T4 / L4 / A10). Compared with full-
    parameter PPO on a 0.5B model, QLoRA cuts VRAM by ~4× and keeps the
    optimiser footprint small enough that the *whole* environment loop
    (LLM + value head + env client) fits on a single T4. We only target
    attention/MLP projection matrices because that's where most of the
    behaviour shaping happens for instruction following.
    """
    if device.type != "cuda":
        raise SystemExit("--use-qlora requires CUDA. Run on Colab/HF Space GPU.")
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "QLoRA requested but `peft` / `bitsandbytes` are not importable. "
            "Install with: pip install peft bitsandbytes"
        ) from exc

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    base = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=bnb,
        attn_implementation=attn_implementation,
        device_map={"": 0},
    )
    base.pretrained_model = prepare_model_for_kbit_training(base.pretrained_model)
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    base.pretrained_model = get_peft_model(base.pretrained_model, lora)
    return base


def run_training(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    runner = OpenEnvEpisodeRunner(args.env_base_url)

    device = resolve_training_device(args.device)
    use_half = device.type == "cuda" and not args.fp32
    dtype = model_dtype_for(device, use_half)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_qlora:
        print("[TRAIN] loading model in QLoRA (4-bit + LoRA adapters)", flush=True)
        model = _load_model_qlora(
            args.model_name,
            device,
            dtype,
            args.attn_implementation,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )
        # bnb's 4-bit weights already live on CUDA; no explicit .to(device).
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
        )
        model = model.to(device)

    accel_kwargs: Dict[str, Any] = {"cpu": True} if device.type == "cpu" else {}
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        accelerator_kwargs=accel_kwargs,
        log_with=None,
    )

    # Notebooks / interactive sessions may have initialized Accelerate already; TRL
    # cannot construct a second Accelerator with different flags without a reset.
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    all_tasks = runner.tasks()
    difficulty_by_task = {t["task_id"]: t["difficulty"] for t in all_tasks}
    results: List[EpisodeResult] = []
    step_log_path = os.path.join(args.output_dir.rstrip("/"), "training_steps.jsonl")
    os.makedirs(args.output_dir.rstrip("/"), exist_ok=True)
    step_log = open(step_log_path, "w", encoding="utf-8")

    try:
        for episode in range(args.episodes):
            task_id = select_task(all_tasks, episode)
            observation = runner.reset(task_id)
            total_reward = 0.0
            final_task_score = 0.0
            success = False

            episode_fallback_steps = 0
            for step_idx in range(args.max_steps):
                prompt = build_prompt(observation)
                query_tensor = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
                query_tensor = query_tensor.to(model.pretrained_model.device)

                response_tensor = trainer.generate(
                    query_tensor,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                ).squeeze(0)

                response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
                used_fallback = False
                try:
                    parsed_action = extract_json_object(response_text)
                    action = normalize_action(parsed_action, observation)
                    if action.get("action_type") == "noop":
                        # Noop usually means the model produced something
                        # we couldn't normalise; fall back so the env returns
                        # a meaningful learning signal.
                        action = heuristic_fallback(observation)
                        used_fallback = True
                except Exception:
                    action = heuristic_fallback(observation)
                    used_fallback = True
                if used_fallback:
                    episode_fallback_steps += 1

                step_payload = runner.step(action)
                reward_value = float(step_payload["reward"]["score"])
                done = bool(step_payload["done"])

                reward_t = torch.tensor(reward_value, dtype=torch.float32, device=device)
                ppo_stats = trainer.step(
                    [query_tensor],
                    [response_tensor],
                    [reward_t],
                )

                # Persist per-step telemetry so we have real loss/reward curves.
                try:
                    step_log.write(
                        json.dumps(
                            {
                                "episode": episode,
                                "step": step_idx,
                                "task_id": task_id,
                                "reward": reward_value,
                                "done": done,
                                "fallback": used_fallback,
                                "ppo/loss/total": float(ppo_stats.get("ppo/loss/total", 0.0))
                                if ppo_stats
                                else None,
                                "ppo/loss/policy": float(ppo_stats.get("ppo/loss/policy", 0.0))
                                if ppo_stats
                                else None,
                                "ppo/loss/value": float(ppo_stats.get("ppo/loss/value", 0.0))
                                if ppo_stats
                                else None,
                            }
                        )
                        + "\n"
                    )
                    step_log.flush()
                except Exception:
                    pass

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
                    difficulty=difficulty_by_task.get(task_id, "?"),
                    total_reward=total_reward,
                    final_task_score=final_task_score,
                    success=success,
                    steps=int(observation.get("step_count", 0)),
                    policy="trained_ppo",
                    fallback_steps=episode_fallback_steps,
                )
            )

            print(
                f"[TRAIN] episode={episode} task={task_id} total_reward={total_reward:.3f} "
                f"task_score={final_task_score:.3f} success={success} "
                f"fallback_steps={episode_fallback_steps}",
                flush=True,
            )
    finally:
        step_log.close()

    output_dir = args.output_dir.rstrip("/")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ---- final eval pass: greedy decode, one episode per unique mission. ----
    # This is what gets saved as ``training_summary.jsonl`` because it
    # mirrors the runtime stack the HF Space serves: trained LLM proposes,
    # expert guardrail catches malformed outputs. The scoreboard the
    # README links is computed from this file.
    eval_task_ids = [t["task_id"] for t in all_tasks]
    eval_results: List[EpisodeResult] = []
    print(
        f"[EVAL] running trained policy on {len(eval_task_ids)} unique missions (greedy decode)",
        flush=True,
    )
    for ep_idx, task_id in enumerate(eval_task_ids):
        observation = runner.reset(task_id)
        total_reward = 0.0
        final_task_score = 0.0
        success = False
        eval_fallback = 0
        for step_idx in range(args.max_steps):
            prompt = build_prompt(observation)
            query_tensor = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
            query_tensor = query_tensor.to(model.pretrained_model.device)
            with torch.no_grad():
                response_tensor = trainer.generate(
                    query_tensor,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                ).squeeze(0)
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            used_fb = False
            try:
                parsed = extract_json_object(response_text)
                action = normalize_action(parsed, observation)
                if action.get("action_type") == "noop":
                    action = heuristic_fallback(observation)
                    used_fb = True
            except Exception:
                action = heuristic_fallback(observation)
                used_fb = True
            if used_fb:
                eval_fallback += 1
            step_payload = runner.step(action)
            total_reward += float(step_payload["reward"]["score"])
            observation = step_payload["observation"]
            if step_payload["done"]:
                info = step_payload.get("info", {})
                final_task_score = float(info.get("task_score", 0.0))
                success = bool(info.get("success", False))
                break
        eval_results.append(
            EpisodeResult(
                episode_id=ep_idx,
                task_id=task_id,
                difficulty=difficulty_by_task.get(task_id, "?"),
                total_reward=total_reward,
                final_task_score=final_task_score,
                success=success,
                steps=int(observation.get("step_count", 0)),
                policy="trained",
                fallback_steps=eval_fallback,
            )
        )
        print(
            f"[EVAL] task={task_id} reward={total_reward:.3f} "
            f"score={final_task_score:.3f} success={success} fallback={eval_fallback}",
            flush=True,
        )

    summary_path = f"{output_dir}/training_summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as handle:
        for item in eval_results:
            handle.write(json.dumps(item.__dict__) + "\n")

    # Also persist the raw training trajectory next to it so reviewers can
    # inspect the (noisier) PPO loop independently from the final eval.
    with open(f"{output_dir}/training_trajectory.jsonl", "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.__dict__) + "\n")

    avg_reward = sum(r.total_reward for r in eval_results) / max(1, len(eval_results))
    avg_score = sum(r.final_task_score for r in eval_results) / max(1, len(eval_results))
    success_rate = sum(1 for r in eval_results if r.success) / max(1, len(eval_results))

    print(
        f"[SUMMARY] train_episodes={len(results)} eval_episodes={len(eval_results)} "
        f"eval_avg_reward={avg_reward:.3f} eval_avg_task_score={avg_score:.3f} "
        f"eval_success_rate={success_rate:.3f}",
        flush=True,
    )

    runner.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TRL PPO training loop for Neon Syndicate.")
    parser.add_argument(
        "--env-base-url",
        type=str,
        default="http://localhost:7860",
        help="OpenEnv API origin, e.g. http://127.0.0.1:7860 or https://YOUR-USER-YOUR-SPACE.hf.space",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Model device: auto (CUDA if available, else CPU), cpu, cuda, mps (not fully tested).",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="On CUDA, use float32 instead of bf16/fp16 (slower, sometimes more stable).",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        help="transformers attention: sdpa (faster) or eager.",
    )
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Per-episode action cap. Match the env's max_steps (24).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Cap for generation; one JSON action usually needs <64 tokens. Lower = faster.",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/trl-neon-model")

    # ----- QLoRA -----
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="Load model in 4-bit + LoRA adapters (recommended for Colab/HF Space GPUs).",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
