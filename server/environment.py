from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

try:
	from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
except Exception:
	# Fallback keeps local execution working even when openenv-core is unavailable.
	class OpenEnvEnvironment:  # type: ignore[too-many-ancestors]
		pass

from models import (
	Action,
	EnvironmentState,
	MissionBrief,
	Observation,
	Reward,
	StepResponse,
	TaskDefinition,
	TaskSummary,
	TaskTarget,
)


def _clamp01(value: float) -> float:
	return max(0.0, min(1.0, value))


class NeonSyndicateEnvironment(OpenEnvEnvironment):
	"""
	Neon Syndicate is a partially observable, long-horizon strategy environment.
	The agent must negotiate alliances, manage resources, and execute a clean extraction.
	"""

	def __init__(self) -> None:
		super().__init__()
		# 24 steps so 3-faction "hard" missions are actually solvable end-to-end
		# (6 negotiates + interleaved trades + deploy + run + extract). Easy
		# missions terminate as soon as ``state.resolved`` flips, so the longer
		# clock only helps the difficult ones.
		self.max_steps = 24
		self._tasks = self._build_tasks()
		self._task_order = [
			"task_easy_docklands_relay",
			"task_easy_data_spire_broker",
			"task_medium_undergrid_blackout",
			"task_medium_citadel_convoy",
			"task_hard_orchid_coup",
			"task_hard_citywide_failsafe",
		]
		self._task_index = 0
		self._state: Optional[EnvironmentState] = None

	def _build_tasks(self) -> Dict[str, TaskDefinition]:
		return {
			"task_easy_docklands_relay": TaskDefinition(
				task_id="task_easy_docklands_relay",
				difficulty="easy",
				title="Docklands Relay Hijack",
				objective=(
					"Gain one local alliance, deploy at Docklands, execute OP-LANTERN, then extract with a coded message."
				),
				mission=MissionBrief(
					mission_id="NS-1101",
					city="Neon Meridian",
					client="Freelance Union",
					stakes="Intercept a surveillance relay before midnight curfew.",
					initial_threat="medium",
					rumors=[
						"Ghostwire hates loud operations.",
						"Docklands patrol swaps every 15 minutes.",
					],
				),
				target=TaskTarget(
					required_allies=["ghostwire"],
					required_operation_code="OP-LANTERN",
					extraction_sector="docklands",
					min_resources={"credits": 30, "intel": 35, "influence": 20, "energy": 20},
					required_message_keywords=["window", "relay", "clean exit"],
				),
			),
			"task_easy_data_spire_broker": TaskDefinition(
				task_id="task_easy_data_spire_broker",
				difficulty="easy",
				title="Data Spire Broker Run",
				objective=(
					"Secure broker trust and run OP-PRISM in Data Spire without triggering critical threat."
				),
				mission=MissionBrief(
					mission_id="NS-1133",
					city="Neon Meridian",
					client="Archive Cartel",
					stakes="Extract ledger keys from a broker node.",
					initial_threat="medium",
					rumors=[
						"Civic Shield monitors all public channels in the Spire.",
						"A precise operation is worth more than a fast one.",
					],
				),
				target=TaskTarget(
					required_allies=["civic_shield"],
					required_operation_code="OP-PRISM",
					extraction_sector="data_spire",
					min_resources={"credits": 25, "intel": 40, "influence": 30, "energy": 15},
					required_message_keywords=["broker", "keys", "exfil"],
				),
			),
			"task_medium_undergrid_blackout": TaskDefinition(
				task_id="task_medium_undergrid_blackout",
				difficulty="medium",
				title="Undergrid Blackout Play",
				objective=(
					"Form a two-faction coalition, stage undergrid assets, and execute OP-NIGHTLOCK under pressure."
				),
				mission=MissionBrief(
					mission_id="NS-2208",
					city="Neon Meridian",
					client="Circuit Assembly",
					stakes="Cut outage spillover before hospitals lose emergency routing.",
					initial_threat="high",
					rumors=[
						"Iron Vultures only cooperate when paid in credits.",
						"Ghostwire shares intel when reputation is positive.",
					],
				),
				target=TaskTarget(
					required_allies=["ghostwire", "iron_vultures"],
					required_operation_code="OP-NIGHTLOCK",
					extraction_sector="undergrid",
					min_resources={"credits": 40, "intel": 55, "influence": 30, "energy": 35},
					required_message_keywords=["stabilized", "fallback", "undergrid"],
				),
			),
			"task_medium_citadel_convoy": TaskDefinition(
				task_id="task_medium_citadel_convoy",
				difficulty="medium",
				title="Citadel Convoy Diversion",
				objective=(
					"Broker competing interests to reroute a convoy, then run OP-HALO and leave no trace."
				),
				mission=MissionBrief(
					mission_id="NS-2270",
					city="Neon Meridian",
					client="Blue Lantern Syndicate",
					stakes="Divert convoy hardware before cartel takeover.",
					initial_threat="high",
					rumors=[
						"Black Orchid wants influence, not money.",
						"Civic Shield demands minimal civilian disruption.",
					],
				),
				target=TaskTarget(
					required_allies=["black_orchid", "civic_shield"],
					required_operation_code="OP-HALO",
					extraction_sector="citadel_gate",
					min_resources={"credits": 35, "intel": 50, "influence": 45, "energy": 30},
					required_message_keywords=["convoy", "handoff", "silent"],
				),
			),
			"task_hard_orchid_coup": TaskDefinition(
				task_id="task_hard_orchid_coup",
				difficulty="hard",
				title="Orchid Coup Containment",
				objective=(
					"Neutralize a faction coup by balancing three alliances and executing OP-OBSIDIAN in one chain."
				),
				mission=MissionBrief(
					mission_id="NS-9012",
					city="Neon Meridian",
					client="City Continuity Office",
					stakes="Prevent market collapse from a hostile leadership swap.",
					initial_threat="critical",
					rumors=[
						"Trust breaks instantly when promises are inconsistent.",
						"Undergrid channels can reduce threat if scouted early.",
					],
				),
				target=TaskTarget(
					required_allies=["ghostwire", "civic_shield", "black_orchid"],
					required_operation_code="OP-OBSIDIAN",
					extraction_sector="undergrid",
					min_resources={"credits": 45, "intel": 70, "influence": 65, "energy": 45},
					required_message_keywords=["containment", "chain of command", "stabilized"],
				),
			),
			"task_hard_citywide_failsafe": TaskDefinition(
				task_id="task_hard_citywide_failsafe",
				difficulty="hard",
				title="Citywide Failsafe Cascade",
				objective=(
					"Assemble a mixed coalition and execute OP-DAWNFALL while preserving enough resources for extraction."
				),
				mission=MissionBrief(
					mission_id="NS-9990",
					city="Neon Meridian",
					client="Emergency Grid Authority",
					stakes="Stop cascading failsafe triggers across the whole city.",
					initial_threat="critical",
					rumors=[
						"Iron Vultures can brute-force energy nodes for a price.",
						"Ghostwire and Civic Shield distrust each other by default.",
					],
				),
				target=TaskTarget(
					required_allies=["ghostwire", "iron_vultures", "civic_shield"],
					required_operation_code="OP-DAWNFALL",
					extraction_sector="data_spire",
					min_resources={"credits": 50, "intel": 80, "influence": 60, "energy": 55},
					required_message_keywords=["failsafe", "recovered", "city grid"],
				),
			),
		}

	def tasks(self) -> List[TaskSummary]:
		return [
			TaskSummary(
				task_id=task.task_id,
				difficulty=task.difficulty,
				title=task.title,
				objective=task.objective,
			)
			for task in self._tasks.values()
		]

	# ------------------------------------------------------------------
	# Procedural task generator (RLVE upgrade path)
	# ------------------------------------------------------------------

	def generate_procedural_task(
		self,
		difficulty: int = 3,
		seed: Optional[int] = None,
	) -> TaskDefinition:
		"""Synthesize a brand-new mission at the requested difficulty.

		The five axes that scale with difficulty:

		* number of required allies (1, 1, 2, 3, 3)
		* number of required keywords in the extraction message (2, 2, 3, 3, 4)
		* operation-code length / format
		* initial threat level
		* resource thresholds

		This is the on-demand RLVE knob exposed at /lab and described in
		``docs/walkthrough.md`` §6. Plug it into the TRL sampler to get
		an unbounded curriculum.
		"""
		import random
		rng = random.Random(seed)

		level = max(1, min(5, int(difficulty)))

		factions = ["ghostwire", "iron_vultures", "civic_shield", "black_orchid"]
		sectors = ["docklands", "data_spire", "undergrid", "citadel_gate"]
		keyword_pool = [
			"ghostwire", "lantern", "midnight", "convoy", "broker", "extract",
			"mirage", "blackout", "failsafe", "city grid", "recovered",
			"dawnfall", "veil", "nighthaven", "sunfall",
		]
		op_pool_short  = ["OP-LANTERN", "OP-MIRAGE", "OP-VEIL", "OP-EMBER"]
		op_pool_medium = ["OP-SUNFALL", "OP-NIGHTHAVEN", "OP-PHANTOM", "OP-AURORA"]
		op_pool_long   = ["OP-DAWNFALL-23", "OP-FAILSAFE-9", "OP-PRISM-77", "OP-OBSIDIAN-44"]

		ally_count_by_level = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
		kw_count_by_level   = {1: 2, 2: 2, 3: 3, 4: 3, 5: 4}
		threat_by_level     = {1: "low", 2: "low", 3: "medium", 4: "high", 5: "critical"}

		ally_count = ally_count_by_level[level]
		kw_count   = kw_count_by_level[level]
		op_pool    = op_pool_short if level <= 2 else (op_pool_medium if level <= 4 else op_pool_long)

		base_thresholds = {"credits": 30, "intel": 30, "influence": 25, "energy": 25}
		scale = 1.0 + 0.2 * (level - 1)

		required_allies = rng.sample(factions, k=ally_count)
		required_kws    = rng.sample(keyword_pool, k=kw_count)
		op_code         = rng.choice(op_pool)
		extraction_sec  = rng.choice(sectors)
		initial_threat  = threat_by_level[level]

		min_resources = {k: int(round(v * scale)) for k, v in base_thresholds.items()}

		task_id = f"task_proc_lvl{level}_{rng.randint(1000, 9999)}"
		mission_id = f"NS-PROC-{rng.randint(1000, 9999)}"
		titles = {
			1: "Relay Skim",
			2: "Spire Intercept",
			3: "Undergrid Bargain",
			4: "Orchid Sabotage",
			5: "Citywide Failsafe",
		}
		stake = (
			"Procedurally generated mission. "
			f"Coalition target: {', '.join(required_allies)}. "
			f"Operation: {op_code}. Extraction: {extraction_sec}."
		)

		return TaskDefinition(
			task_id=task_id,
			difficulty=("easy" if level <= 2 else ("medium" if level <= 4 else "hard")),
			title=titles[level],
			objective=(
				f"Gain {ally_count} alliance(s), deploy at {extraction_sec}, "
				f"execute {op_code}, then secure extraction with the right keywords."
			),
			mission=MissionBrief(
				mission_id=mission_id,
				city="Neon Meridian",
				client="Procedural Sampler",
				stakes=stake,
				initial_threat=initial_threat,
				rumors=[
					f"Rumor: {required_kws[0]} is the password for tonight.",
					f"Rumor: extraction window opens near {extraction_sec}.",
				],
			),
			target=TaskTarget(
				required_allies=required_allies,
				required_operation_code=op_code,
				extraction_sector=extraction_sec,
				min_resources=min_resources,
				required_message_keywords=required_kws,
			),
		)

	def register_procedural_task(self, task: TaskDefinition) -> None:
		"""Register a generated task into the live env so /reset can use it."""
		self._tasks[task.task_id] = task

	def reward_design(self) -> Dict[str, object]:
		"""Machine-readable summary of the env's reward design.

		Used by /lab to render the 5-gate rubric and the penalty table
		without re-implementing them in JS. Keeps the page in lockstep
		with whatever ``_apply_action`` actually does.
		"""
		return {
			"grader_weights": {
				"alliance":   0.30,
				"resources":  0.25,
				"operation":  0.20,
				"message":    0.15,
				"extraction": 0.10,
			},
			"dense_weights": {
				"alliances":       0.30,
				"resources":       0.25,
				"operation":       0.20,
				"message_quality": 0.15,
				"extraction":      0.10,
			},
			"penalties": [
				{"trigger": "Identical action repeated back-to-back", "penalty": 0.08, "defends_against": "Spam-best-action exploit"},
				{"trigger": "scout_sector without sector",            "penalty": 0.08, "defends_against": "Malformed-JSON spam"},
				{"trigger": "negotiate_pact without faction",         "penalty": 0.08, "defends_against": "Malformed-JSON spam"},
				{"trigger": "trade_resources without resource/amount","penalty": 0.08, "defends_against": "Malformed-JSON spam"},
				{"trigger": "Insufficient influence to negotiate",    "penalty": 0.07, "defends_against": "Free-recruit attempt"},
				{"trigger": "Insufficient energy to deploy",          "penalty": 0.08, "defends_against": "Free-deploy attempt"},
				{"trigger": "run_operation before deploy",            "penalty": 0.10, "defends_against": "Skip the deploy gate"},
				{"trigger": "run_operation with zero alliances",      "penalty": 0.10, "defends_against": "Skip the alliance gate"},
				{"trigger": "run_operation with wrong code",          "penalty": 0.05, "defends_against": "Brute-forcing op codes"},
				{"trigger": "secure_extraction without sector/message", "penalty": 0.10, "defends_against": "Empty-extraction shortcut"},
				{"trigger": "noop while mission is live",             "penalty": 0.06, "defends_against": "Stalling to drain the clock"},
				{"trigger": "Critical-threat passive play",           "penalty": 0.04, "defends_against": "Ignoring threat escalation"},
			],
			"action_types": [
				"scout_sector", "negotiate_pact", "trade_resources",
				"deploy_asset", "run_operation", "secure_extraction", "noop",
			],
			"max_steps": self.max_steps,
			"factions": ["ghostwire", "iron_vultures", "civic_shield", "black_orchid"],
		}

	def reset(
		self,
		seed: Optional[int] = None,
		episode_id: Optional[str] = None,
		task_id: Optional[str] = None,
		**kwargs,
	) -> Observation:
		_ = seed, episode_id, kwargs
		if task_id is None:
			task_id = self._task_order[self._task_index]
			self._task_index = (self._task_index + 1) % len(self._task_order)

		if task_id not in self._tasks:
			raise ValueError(f"Unknown task_id: {task_id}")

		task = deepcopy(self._tasks[task_id])
		self._state = EnvironmentState(
			active_task=task,
			step_count=0,
			max_steps=self.max_steps,
			known_threat=task.mission.initial_threat,
			resources={"credits": 30, "intel": 20, "influence": 20, "energy": 20},
			reputation={
				"ghostwire": 0,
				"iron_vultures": 0,
				"civic_shield": 0,
				"black_orchid": 0,
			},
		)
		return self._observation()

	def _require_state(self) -> EnvironmentState:
		if self._state is None:
			raise RuntimeError("Environment has not been reset.")
		return self._state

	@property
	def state(self) -> EnvironmentState:
		return self._require_state()

	def _observation(self) -> Observation:
		state = self._require_state()
		return Observation(
			task_id=state.active_task.task_id,
			difficulty=state.active_task.difficulty,
			objective=state.active_task.objective,
			step_count=state.step_count,
			max_steps=state.max_steps,
			mission=state.active_task.mission,
			known_threat=state.known_threat,
			resources=dict(state.resources),
			reputation=dict(state.reputation),
			alliances=list(state.alliances),
			deployed_sector=state.deployed_sector,
			operation_ready=state.operation_ready,
			operation_executed=state.operation_executed,
			extraction_ready=state.extraction_ready,
			intel_log=list(state.intel_log),
			last_action=None,
			action_history=list(state.action_history),
		)

	def _resource_progress(self, current: Dict[str, int], required: Dict[str, int]) -> float:
		scores = []
		for key, threshold in required.items():
			if threshold <= 0:
				scores.append(1.0)
				continue
			scores.append(min(1.0, current.get(key, 0) / threshold))
		return sum(scores) / max(1, len(scores))

	def _message_quality(self, message: Optional[str], keywords: List[str]) -> float:
		if not message:
			return 0.0
		msg = message.lower()
		matched = sum(1 for kw in keywords if kw.lower() in msg)
		return matched / max(1, len(keywords))

	def _grader_score(self, state: EnvironmentState) -> float:
		target = state.active_task.target
		alliance_score = sum(1.0 for f in target.required_allies if f in state.alliances) / max(
			1, len(target.required_allies)
		)
		resource_score = self._resource_progress(state.resources, target.min_resources)
		operation_score = 1.0 if state.operation_executed else 0.0
		extraction_score = (
			1.0
			if state.success and state.extraction_sector == target.extraction_sector
			else 0.0
		)
		message_score = self._message_quality(state.extraction_message, target.required_message_keywords)

		weighted = (
			0.30 * alliance_score
			+ 0.25 * resource_score
			+ 0.20 * operation_score
			+ 0.15 * message_score
			+ 0.10 * extraction_score
		)
		return _clamp01(weighted)

	def _progress_signals(self, state: EnvironmentState) -> Tuple[float, Dict[str, float]]:
		target = state.active_task.target
		alliance_signal = sum(1.0 for f in target.required_allies if f in state.alliances) / max(
			1, len(target.required_allies)
		)
		resource_signal = self._resource_progress(state.resources, target.min_resources)
		operation_signal = 1.0 if state.operation_executed else (0.5 if state.operation_ready else 0.0)
		message_signal = self._message_quality(state.extraction_message, target.required_message_keywords)
		extraction_signal = 1.0 if state.success else (0.5 if state.extraction_ready else 0.0)

		components = {
			"alliances": 0.30 * alliance_signal,
			"resources": 0.25 * resource_signal,
			"operation": 0.20 * operation_signal,
			"message_quality": 0.15 * message_signal,
			"extraction": 0.10 * extraction_signal,
		}
		return sum(components.values()), components

	def _adjust_threat(self, state: EnvironmentState, delta: int) -> None:
		levels = ["low", "medium", "high", "critical"]
		current_idx = levels.index(state.known_threat)
		new_idx = max(0, min(len(levels) - 1, current_idx + delta))
		state.known_threat = levels[new_idx]

	def _apply_action(self, state: EnvironmentState, action: Action) -> Tuple[float, str]:
		penalty = 0.0
		reason = "Action accepted."
		target = state.active_task.target

		action_signature = action.model_dump_json(exclude_none=True)
		if state.action_history and state.action_history[-1] == action_signature:
			penalty += 0.08
			reason = "Penalty: repeated identical action."

		if action.action_type == "scout_sector":
			if not action.sector:
				penalty += 0.08
				reason = "Scout requires sector."
			else:
				state.resources["intel"] = min(100, state.resources["intel"] + 12)
				state.resources["energy"] = max(0, state.resources["energy"] - 2)
				intel_note = f"Scout report from {action.sector}: patrol rhythm mapped."
				state.intel_log.append(intel_note)
				if action.sector == target.extraction_sector:
					self._adjust_threat(state, -1)
					reason = "High-value intel found at target extraction sector."
		elif action.action_type == "negotiate_pact":
			if not action.faction:
				penalty += 0.08
				reason = "Negotiation requires faction."
			else:
				cost = 8
				if state.resources["influence"] < cost:
					penalty += 0.07
					reason = "Not enough influence to negotiate."
				else:
					state.resources["influence"] -= cost
					state.reputation[action.faction] = min(100, state.reputation[action.faction] + 18)
					if state.reputation[action.faction] >= 35 and action.faction not in state.alliances:
						state.alliances.append(action.faction)
						reason = f"Alliance formed with {action.faction}."
					else:
						reason = f"Negotiation advanced with {action.faction}."
		elif action.action_type == "trade_resources":
			if not action.resource or action.amount is None:
				penalty += 0.08
				reason = "Trade requires resource and amount."
			else:
				amount = max(1, min(25, action.amount))
				if action.resource == "credits":
					state.resources["credits"] = min(120, state.resources["credits"] + amount)
					state.resources["energy"] = max(0, state.resources["energy"] - 4)
				elif action.resource == "intel":
					state.resources["intel"] = min(120, state.resources["intel"] + amount)
					state.resources["credits"] = max(0, state.resources["credits"] - 5)
				elif action.resource == "influence":
					state.resources["influence"] = min(120, state.resources["influence"] + amount)
					state.resources["credits"] = max(0, state.resources["credits"] - 6)
				elif action.resource == "energy":
					state.resources["energy"] = min(120, state.resources["energy"] + amount)
					state.resources["credits"] = max(0, state.resources["credits"] - 4)
				reason = f"Resource lane shifted toward {action.resource}."
		elif action.action_type == "deploy_asset":
			if not action.sector:
				penalty += 0.08
				reason = "Deploy requires sector."
			elif state.resources["energy"] < 10:
				penalty += 0.08
				reason = "Not enough energy to deploy asset."
			else:
				state.resources["energy"] -= 10
				state.deployed_sector = action.sector
				state.operation_ready = True
				reason = f"Asset deployed to {action.sector}."
		elif action.action_type == "run_operation":
			if not action.operation_code:
				penalty += 0.08
				reason = "Operation requires operation_code."
			elif not state.operation_ready:
				penalty += 0.10
				reason = "Cannot run operation before deploying assets."
			elif len(state.alliances) == 0:
				penalty += 0.10
				reason = "Operation is unstable without at least one alliance."
			else:
				state.resources["intel"] = min(130, state.resources["intel"] + 15)
				state.resources["credits"] = min(130, state.resources["credits"] + 10)
				state.operation_executed = action.operation_code == target.required_operation_code
				state.extraction_ready = state.operation_executed
				if state.operation_executed:
					self._adjust_threat(state, -1)
					reason = "Operation matched target plan and succeeded."
				else:
					penalty += 0.05
					reason = "Operation ran, but code mismatch lowered mission quality."
		elif action.action_type == "secure_extraction":
			if not action.sector or not action.message:
				penalty += 0.10
				reason = "Extraction requires sector and message."
			else:
				state.extraction_sector = action.sector
				state.extraction_message = action.message.strip()
				allies_ok = all(f in state.alliances for f in target.required_allies)
				resources_ok = all(
					state.resources.get(k, 0) >= threshold for k, threshold in target.min_resources.items()
				)
				message_ok = self._message_quality(
					state.extraction_message, target.required_message_keywords
				) >= 0.66
				state.success = (
					state.operation_executed
					and allies_ok
					and resources_ok
					and message_ok
					and action.sector == target.extraction_sector
				)
				state.resolved = True
				if state.success:
					reason = "Clean extraction. Coalition objective achieved."
				else:
					reason = "Extraction attempted, but objective constraints were not fully met."
		elif action.action_type == "noop":
			penalty += 0.06
			reason = "No-op penalty to discourage stalling."
		else:
			penalty += 0.12
			reason = "Unknown action type."

		if state.known_threat == "critical" and action.action_type in {
			"noop",
			"trade_resources",
		}:
			penalty += 0.04
			reason = "Threat is critical; passive play is penalized."

		state.action_history.append(action_signature)
		return penalty, reason

	def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs) -> StepResponse:
		_ = timeout_s, kwargs
		state = self._require_state()
		if state.resolved or state.step_count >= state.max_steps:
			score = self._grader_score(state)
			obs = self._observation()
			obs.last_action = action
			return StepResponse(
				observation=obs,
				reward=Reward(score=score, components={}, reason="Episode already complete."),
				done=True,
				info={
					"task_score": score,
					"success": state.success,
					"max_steps_reached": state.step_count >= state.max_steps,
				},
			)

		old_progress, _ = self._progress_signals(state)
		penalty, reason = self._apply_action(state, action)

		state.step_count += 1
		new_progress, components = self._progress_signals(state)
		progress_delta = new_progress - old_progress
		completion_bonus = 0.10 if state.resolved and state.success else (0.03 if state.resolved else 0.0)
		dense_reward = _clamp01(progress_delta + completion_bonus - penalty)

		state.cumulative_penalty += penalty
		state.cumulative_reward = _clamp01(state.cumulative_reward + dense_reward)

		done = state.resolved or state.step_count >= state.max_steps
		task_score = self._grader_score(state)

		observation = self._observation()
		observation.last_action = action
		reward = Reward(score=dense_reward, components=components, reason=reason)

		info = {
			"task_score": task_score,
			"success": state.success,
			"progress": new_progress,
			"penalty": penalty,
			"cumulative_reward": state.cumulative_reward,
			"done_reason": "resolved" if state.resolved else ("max_steps" if done else "ongoing"),
		}

		return StepResponse(observation=observation, reward=reward, done=done, info=info)
