from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(BaseModel):
    id: int = Field(
        description="Sequential task number starting from 1."
    )

    title: str = Field(
        description="Short descriptive task title."
    )

    objective: str = Field(
        description="What this task accomplishes."
    )

    implementation_notes: List[str] = Field(
        default_factory=list,
        description="Important implementation details for the executor."
    )

    dependencies: List[int] = Field(
        default_factory=list,
        description="IDs of prerequisite tasks."
    )

    affected_components: List[str] = Field(
        default_factory=list,
        description="Files, modules, services, packages or components likely to change."
    )


class ValidationPlan(BaseModel):
    unit_tests: List[str] = Field(default_factory=list)

    integration_tests: List[str] = Field(default_factory=list)

    manual_validation: List[str] = Field(default_factory=list)

    edge_cases: List[str] = Field(default_factory=list)


class ImplementationPlan(BaseModel):
    goal: str = Field(
        description="Overall objective of the implementation."
    )

    strategy: str = Field(
        description="High-level implementation strategy."
    )

    tasks: List[Task] = Field(
        description="Ordered implementation tasks."
    )

    affected_components: List[str] = Field(
        default_factory=list,
        description="Overall list of affected files, modules or services."
    )

    missing_information: List[str] = Field(
        default_factory=list,
        description="Unknown information that could not be determined."
    )

    assumptions: List[str] = Field(
        default_factory=list,
        description="Explicit assumptions made while planning."
    )

    validation: ValidationPlan = Field(
        default_factory=ValidationPlan
    )

    estimated_complexity: Complexity = Complexity.MEDIUM

    estimated_risk: RiskLevel = RiskLevel.LOW