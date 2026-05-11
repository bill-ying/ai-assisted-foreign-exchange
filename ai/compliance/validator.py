"""
Validation strategies for executing compliance rule chains (GoF Strategy pattern).

GoF Patterns:
- Strategy: ValidationStrategy is the abstract strategy interface. Concrete
  implementations define how a rule chain is assembled and how violation
  severity is interpreted. The strategy is injected into ComplianceGraph,
  allowing the graph's validation behaviour to be swapped without modifying
  any graph or rule code.

Two built-in strategies:
- LenientValidationStrategy: only ERROR violations block the response
- StrictValidationStrategy:  WARNING violations are also treated as blocking

The module also provides build_default_rule_chain() — a factory function
that assembles the standard four-rule chain in priority order.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .rules import (
    ComplianceRule,
    CurrencyConsistencyRule,
    DateConsistencyRule,
    RateValuePresentRule,
    SourceAttributionRule,
)
from .state import ComplianceStatus, ValidationResult


def build_default_rule_chain() -> ComplianceRule:
    """
    Assemble the default compliance rule chain in priority order.

    Order rationale:
      1. RateValuePresentRule    — most critical: catch hallucinated rates first
      2. SourceAttributionRule   — regulatory provenance requirement
      3. DateConsistencyRule     — factual accuracy of the queried date
      4. CurrencyConsistencyRule — factual accuracy of the queried currencies

    Returns:
        Head of the assembled ComplianceRule chain
    """
    head = RateValuePresentRule()
    head.set_next(SourceAttributionRule()) \
        .set_next(DateConsistencyRule()) \
        .set_next(CurrencyConsistencyRule())
    return head


class ValidationStrategy(ABC):
    """
    Abstract strategy for response compliance validation (GoF Strategy).

    Defines the interface for running compliance checks on an LLM response
    against the raw tool data. Concrete strategies differ in which rules
    they apply and how they interpret violation severity.
    """

    @abstractmethod
    def validate(
        self,
        response: str,
        tool_results: Dict[str, str],
        tool_args: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate an LLM response against tool-sourced ground truth.

        Args:
            response:     The LLM's generated response text
            tool_results: Formatted tool output strings, keyed by tool name
            tool_args:    Raw args passed to each tool, keyed by tool name

        Returns:
            ValidationResult with status and all found violations
        """
        ...

    def _build_result(
        self, violations: list, treat_warnings_as_errors: bool
    ) -> ValidationResult:
        """Shared logic for assembling a ValidationResult from a violation list."""
        if treat_warnings_as_errors:
            has_failure = len(violations) > 0
        else:
            has_failure = any(v.severity == "ERROR" for v in violations)

        status = ComplianceStatus.FAILED if has_failure else ComplianceStatus.PASSED
        return ValidationResult(status=status, violations=tuple(violations))


class LenientValidationStrategy(ValidationStrategy):
    """
    Validation strategy that only fails on ERROR-severity violations.

    WARNING violations are recorded in the ValidationResult for audit
    purposes but do not trigger a correction loop. Suitable for conversational
    POC use where strict citation format cannot always be guaranteed.

    This is the recommended default for the FX assistant POC.
    """

    def __init__(self, rule_chain: Optional[ComplianceRule] = None) -> None:
        """
        Args:
            rule_chain: Custom rule chain head. Defaults to build_default_rule_chain().
        """
        self._chain = rule_chain or build_default_rule_chain()

    def validate(self, response, tool_results, tool_args) -> ValidationResult:
        violations = self._chain.check(response, tool_results, tool_args)
        return self._build_result(violations, treat_warnings_as_errors=False)


class StrictValidationStrategy(ValidationStrategy):
    """
    Validation strategy that promotes WARNING violations to blocking failures.

    Used when the response must fully conform to all attribution, date-citing,
    and currency requirements — for example, when generating audit reports
    or regulatory submissions where every field must be explicitly present.
    """

    def __init__(self, rule_chain: Optional[ComplianceRule] = None) -> None:
        """
        Args:
            rule_chain: Custom rule chain head. Defaults to build_default_rule_chain().
        """
        self._chain = rule_chain or build_default_rule_chain()

    def validate(self, response, tool_results, tool_args) -> ValidationResult:
        violations = self._chain.check(response, tool_results, tool_args)
        return self._build_result(violations, treat_warnings_as_errors=True)
