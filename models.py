from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
	"scout_sector",
	"negotiate_pact",
	"trade_resources",
	"deploy_asset",
	"run_operation",
	"secure_extraction",
	"noop",
]

Difficulty = Literal["easy", "medium", "hard"]
ThreatLevel = Literal["low", "medium", "high", "critical"]
FactionName = Literal["ghostwire", "iron_vultures", "civic_shield", "black_orchid"]
SectorName = Literal["docklands", "data_spire", "undergrid", "citadel_gate"]


class Action(BaseModel):
	action_type: ActionType
	faction: Optional[FactionName] = None
	sector: Optional[SectorName] = None
	resource: Optional[Literal["credits", "intel", "influence", "energy"]] = None
	amount: Optional[int] = Field(default=None, ge=1, le=100)
	message: Optional[str] = None
	operation_code: Optional[str] = None


class MissionBrief(BaseModel):
	mission_id: str
	city: str
	client: str
	stakes: str
	initial_threat: ThreatLevel
	rumors: List[str]


class Observation(BaseModel):
	task_id: str
	difficulty: Difficulty
	objective: str
	step_count: int
	max_steps: int
	mission: MissionBrief
	known_threat: ThreatLevel
	resources: Dict[str, int]
	reputation: Dict[str, int]
	alliances: List[str] = Field(default_factory=list)
	deployed_sector: Optional[SectorName] = None
	operation_ready: bool = False
	operation_executed: bool = False
	extraction_ready: bool = False
	intel_log: List[str] = Field(default_factory=list)
	last_action: Optional[Action] = None
	action_history: List[str] = Field(default_factory=list)


class Reward(BaseModel):
	score: float = Field(ge=0.0, le=1.0)
	components: Dict[str, float] = Field(default_factory=dict)
	reason: str


class StepResponse(BaseModel):
	observation: Observation
	reward: Reward
	done: bool
	info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
	task_id: Optional[str] = None


class TaskSummary(BaseModel):
	task_id: str
	difficulty: Difficulty
	title: str
	objective: str


class TaskTarget(BaseModel):
	required_allies: List[FactionName]
	required_operation_code: str
	extraction_sector: SectorName
	min_resources: Dict[str, int]
	required_message_keywords: List[str]


class TaskDefinition(BaseModel):
	task_id: str
	difficulty: Difficulty
	title: str
	objective: str
	mission: MissionBrief
	target: TaskTarget


class EnvironmentState(BaseModel):
	active_task: TaskDefinition
	step_count: int
	max_steps: int
	known_threat: ThreatLevel
	resources: Dict[str, int] = Field(default_factory=dict)
	reputation: Dict[str, int] = Field(default_factory=dict)
	alliances: List[str] = Field(default_factory=list)
	deployed_sector: Optional[SectorName] = None
	operation_ready: bool = False
	operation_executed: bool = False
	extraction_ready: bool = False
	extraction_sector: Optional[SectorName] = None
	extraction_message: Optional[str] = None
	intel_log: List[str] = Field(default_factory=list)
	action_history: List[str] = Field(default_factory=list)
	resolved: bool = False
	success: bool = False
	cumulative_reward: float = 0.0
	cumulative_penalty: float = 0.0
