from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(BaseModel):
    id: int = Field(..., description="Sequential task number.")

    title: str = Field(
        ...,
        description="Short title describing the task."
    )

    objective: str = Field(
        ...,
        description="What this task accomplishes."
    )

    reasoning: str = Field(
        ...,
        description="Why this task is necessary."
    )

    implementation_notes: List[str] = Field(
        default_factory=list,
        description="Important implementation considerations."
    )

    dependencies: List[int] = Field(
        default_factory=list,
        description="Task IDs that must complete first."
    )

    affected_components: List[str] = Field(
        default_factory=list,
        description="Files, modules, services or components likely to change."
    )

    risks: List[str] = Field(
        default_factory=list,
        description="Potential issues while implementing this task."
    )

    priority: Priority = Priority.MEDIUM

    complexity: Complexity = Complexity.MEDIUM


class ValidationPlan(BaseModel):
    unit_tests: List[str] = Field(default_factory=list)

    integration_tests: List[str] = Field(default_factory=list)

    manual_validation: List[str] = Field(default_factory=list)

    edge_cases: List[str] = Field(default_factory=list)


class ImplementationPlan(BaseModel):
    goal: str

    strategy: str

    assumptions: List[str] = Field(default_factory=list)

    missing_information: List[str] = Field(default_factory=list)

    affected_components: List[str] = Field(default_factory=list)

    tasks: List[Task]

    overall_risks: List[str] = Field(default_factory=list)

    validation: ValidationPlan

    out_of_scope: List[str] = Field(default_factory=list)

    estimated_complexity: Complexity

    estimated_risk: RiskLevel