"""
LangGraph state and Value Objects for the compliance validation workflow.

GoF Patterns:
- Value Object: ValidationResult and RuleViolation are immutable frozen
  dataclasses. They represent compliance outcomes with no identity —
  two results with the same violations are equal. This prevents mutation
  of audit data after the fact.

The ComplianceState TypedDict is the single shared state record passed
between all LangGraph nodes. The `messages` field uses operator.add so
LangGraph merges new messages rather than replacing the full list.
"""

import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ComplianceStatus(Enum):
    """Lifecycle states of a compliance validation pass."""
    PENDING = auto()
    PASSED = auto()
    FAILED = auto()
    MAX_RETRIES_EXCEEDED = auto()


@dataclass(frozen=True)
class RuleViolation:
    """
    Immutable record of a single compliance rule violation (Value Object).

    Attributes:
        rule_name:   Identifier of the rule that fired
        description: Human-readable explanation of the violation
        severity:    "ERROR" blocks the response; "WARNING" is logged only
    """
    rule_name: str
    description: str
    severity: str  # "ERROR" | "WARNING"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.rule_name}: {self.description}"


@dataclass(frozen=True)
class ValidationResult:
    """
    Immutable outcome of a full compliance validation pass (Value Object).

    Attributes:
        status:       Overall compliance status
        violations:   All rule violations found (errors and warnings)
        validated_at: UTC timestamp of the validation
    """
    status: ComplianceStatus
    violations: tuple  # tuple[RuleViolation, ...]
    validated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def passed(self) -> bool:
        """True if the status is PASSED (set by the ValidationStrategy)."""
        return self.status == ComplianceStatus.PASSED

    @property
    def errors(self) -> List[RuleViolation]:
        """All ERROR-severity violations."""
        return [v for v in self.violations if v.severity == "ERROR"]

    @property
    def warnings(self) -> List[RuleViolation]:
        """All WARNING-severity violations."""
        return [v for v in self.violations if v.severity == "WARNING"]

    def to_correction_prompt(self) -> str:
        """
        Produce a correction instruction for the LLM.

        Called when the graph routes to the correction node so the LLM
        knows exactly what compliance requirements it failed to meet.
        """
        lines = [
            "Your previous response failed compliance checks. "
            "Please rewrite it to satisfy the following requirements:"
        ]
        for v in self.errors:
            lines.append(f"  - {v.description}")
        lines.append(
            "\nYour corrected response must include: "
            "the exact numeric rate returned by the tool, "
            "an explicit reference to the Bank of Canada as the data source, "
            "and the correct date and currencies from the original query."
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        violation_lines = "\n  ".join(str(v) for v in self.violations)
        return (
            f"ValidationResult(status={self.status.name}, "
            f"violations=[\n  {violation_lines}\n])"
        )


class ComplianceState(TypedDict):
    """
    Shared state record passed between all nodes in the compliance LangGraph.

    LangGraph merges state between nodes. The `messages` field is annotated
    with operator.add so new messages are appended, not replaced.

    Fields:
        messages:            Full LangChain message list (append-only)
        user_message:        Original user input string
        tool_results:        Formatted tool output strings, keyed by tool name
        tool_args:           Raw args passed to each tool (for validation)
        llm_response:        Latest LLM text response
        validation_result:   Result of the most recent compliance validation
        correction_attempts: Number of correction iterations so far
        final_response:      The response delivered to the user
        compliance_passed:   Whether the final response passed compliance
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_message: str
    tool_results: Dict[str, str]
    tool_args: Dict[str, Any]
    llm_response: str
    validation_result: Optional[ValidationResult]
    correction_attempts: int
    final_response: str
    compliance_passed: bool
