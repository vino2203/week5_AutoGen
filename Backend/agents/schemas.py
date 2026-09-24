"""Pydantic models shared by all agents."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

ALL_OPTIONS = ["rag", "n8n", "crewai", "autogen"]
Skill = Literal["no-code", "low-code", "code"]
WEIGHT_KEYS = ["fit_to_requirements", "cost", "setup_effort", "latency", "long_term_benefit"]


# ---------------------------------------------------------------- user input
class Requirements(BaseModel):
    """Everything is optional: whatever the user leaves empty, the manager (AI) infers or assumes."""
    channels: list[str] = Field(default_factory=list)
    document_upload: Optional[bool] = None
    messages_per_day: Optional[int] = Field(default=None, ge=0)
    human_approval_of_replies: Optional[bool] = None
    multi_step_reasoning: Optional[bool] = None

    @field_validator("channels")
    @classmethod
    def _clean_channels(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(c.strip().lower() for c in v if c.strip()))


class Constraints(BaseModel):
    budget_per_day_usd: Optional[float] = Field(default=None, gt=0)
    max_latency_ms: Optional[int] = Field(default=None, gt=0)
    min_recall: Optional[float] = Field(default=None, ge=0, le=1)
    user_skill: Optional[Skill] = None
    time_available_hours: Optional[float] = Field(default=None, gt=0)
    time_available: Optional[str] = None  # free text such as "1 week"; the manager (AI) converts it to hours


class Weights(BaseModel):
    fit_to_requirements: Optional[float] = Field(default=None, ge=0)
    cost: Optional[float] = Field(default=None, ge=0)
    setup_effort: Optional[float] = Field(default=None, ge=0)
    latency: Optional[float] = Field(default=None, ge=0)
    long_term_benefit: Optional[float] = Field(default=None, ge=0)

    def resolved(self, defaults: dict) -> "Weights":
        """Fill missing weights from the defaults and normalise to sum 1."""
        merged = {k: (getattr(self, k) if getattr(self, k) is not None else defaults[k]) for k in WEIGHT_KEYS}
        total = sum(merged.values())
        if total <= 0:
            merged, total = dict(defaults), sum(defaults.values())
        return Weights(**{k: round(v / total, 6) for k, v in merged.items()})


def clean_options(v: list[str]) -> list[str]:
    out: list[str] = []
    for o in v:
        o = o.strip().lower().replace("-", "").replace(" ", "")
        if o not in ALL_OPTIONS:
            raise ValueError(f"unknown option '{o}'; allowed: {ALL_OPTIONS}")
        if o not in out:
            out.append(o)
    return out


class ProjectInput(BaseModel):
    project_description: str = ""
    options_to_compare: list[str] = Field(default_factory=list)  # empty: the manager decides
    requirements: Requirements = Field(default_factory=Requirements)
    constraints: Constraints = Field(default_factory=Constraints)
    weights: Optional[Weights] = None

    @field_validator("options_to_compare")
    @classmethod
    def _clean_options(cls, v: list[str]) -> list[str]:
        return clean_options(v)


def required_keys(project: ProjectInput) -> list[str]:
    """The requirement vocabulary that the agents and the recommender share."""
    r = project.requirements
    keys = [f"channel:{c}" for c in r.channels]
    if r.document_upload:
        keys.append("document_upload")
    if r.human_approval_of_replies:
        keys.append("human_approval")
    if r.multi_step_reasoning:
        keys.append("multi_step_reasoning")
    return keys


# ------------------------------------------------ manager (AI) intake output
class Brief(BaseModel):
    """What the manager understood from the user's text and form fields."""
    options_to_compare: list[str]
    channels: list[str]
    document_upload: bool
    messages_per_day: int = Field(ge=0)
    human_approval_of_replies: bool
    multi_step_reasoning: bool
    constraints: Constraints
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("options_to_compare")
    @classmethod
    def _clean_options(cls, v: list[str]) -> list[str]:
        return clean_options(v) or list(ALL_OPTIONS)


# ------------------------------------------------------------------- reports
class FitInfo(BaseModel):
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class SetupStep(BaseModel):
    step: str
    hours: float = Field(ge=0)


class SetupEffort(BaseModel):
    hours: float = Field(ge=0)
    skill_required: Skill
    steps: list[SetupStep] = Field(default_factory=list)


class ReportBody(BaseModel):
    """What a knowledge agent (AI) writes about its framework for this project."""
    summary: str
    fit_to_requirements: FitInfo
    setup_effort: SetupEffort
    estimated_daily_cost_usd: float = Field(ge=0)
    estimated_latency_ms: int = Field(ge=0)
    long_term_score: float = Field(ge=0, le=10)
    expected_recall: Optional[float] = Field(default=None, ge=0, le=1)  # an estimate to validate, not a measurement
    risks: list[str] = Field(default_factory=list)
    works_well_with: list[str] = Field(default_factory=list)
    simple_design: str  # Mermaid flowchart
    how_it_works: list[str]
    project_specific_notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class FrameworkReport(ReportBody):
    agent_id: str
    framework: str


# ------------------------------------------------------------ recommendation
class StackScore(BaseModel):
    rank: int = 0
    stack: list[str]
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    estimated_daily_cost_usd: float = Field(ge=0)
    estimated_latency_ms: int = Field(ge=0)
    setup_hours: float = Field(ge=0)
    skill_required: Skill = "low-code"
    expected_recall: Optional[float] = None
    criteria: dict[str, float]  # 0-10 per criterion, given by the recommender (AI)
    weighted_score: float = 0  # recomputed in code from criteria and weights so the numbers add up
    flags: list[str] = Field(default_factory=list)  # hard-limit failures
    passes: bool = True
    reasoning: str = ""

    @field_validator("stack")
    @classmethod
    def _stack(cls, v: list[str]) -> list[str]:
        return clean_options(v)

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: dict[str, float]) -> dict[str, float]:
        missing = [k for k in WEIGHT_KEYS if k not in v]
        if missing:
            raise ValueError(f"criteria must contain all of {WEIGHT_KEYS}; missing {missing}")
        if any(not 0 <= v[k] <= 10 for k in WEIGHT_KEYS):
            raise ValueError("every criterion score must be between 0 and 10")
        return {k: v[k] for k in WEIGHT_KEYS}


class RecommendationBody(BaseModel):
    """What the recommender (AI) writes."""
    ranking: list[StackScore] = Field(min_length=1)
    rationale: str
    simple_design: str
    how_it_works: list[str]
    phased_plan: list[str]
    upgrade_triggers: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recall_checklist: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    recommended_stack: list[str]
    headline: str
    rationale: str
    ranking: list[StackScore]
    comparison_table_md: str
    simple_design: str
    how_it_works: list[str]
    phased_plan: list[str]
    estimated_daily_cost_usd: float
    estimated_monthly_cost_usd: float
    upgrade_triggers: list[str]
    risks: list[str]
    recall_checklist: list[str]
    assumptions: list[str]
    warnings: list[str]


class AgentError(BaseModel):
    agent_id: str
    error: str


class Session(BaseModel):
    session_id: str
    status: Literal["needs_input", "awaiting_approval", "approved", "rejected", "error"]
    revision: int = 0
    project: Optional[ProjectInput] = None
    questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    reports: list[FrameworkReport] = Field(default_factory=list)
    errors: list[AgentError] = Field(default_factory=list)
    recommendation: Optional[Recommendation] = None
