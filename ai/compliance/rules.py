"""
Compliance rules using the Chain of Responsibility pattern.

GoF Patterns:
- Chain of Responsibility: ComplianceRule is the abstract Handler. Concrete
  rules are linked via set_next(). Each rule checks one concern and delegates
  down the chain, accumulating all violations rather than short-circuiting.
  This ensures every rule is independently evaluated on every response.

All rules are stateless — they receive context via method arguments, not
constructor injection. This makes them trivially composable and reorderable
without any rule knowing about another (Open/Closed Principle).

Rule severity guide:
  ERROR   — factual failure; the response must not be delivered as-is
  WARNING — compliance concern; logged for audit but does not block delivery
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .state import RuleViolation


class ComplianceRule(ABC):
    """
    Abstract handler in the Chain of Responsibility (GoF).

    Each concrete subclass checks one compliance concern and delegates
    to the next rule in the chain. All violations are accumulated across
    the full chain — no short-circuiting.

    Usage:
        head = RateValuePresentRule()
        head.set_next(SourceAttributionRule()).set_next(DateConsistencyRule())
        violations = head.check(response, tool_results, tool_args)
    """

    def __init__(self) -> None:
        self._next: Optional['ComplianceRule'] = None

    def set_next(self, rule: 'ComplianceRule') -> 'ComplianceRule':
        """
        Link the next rule and return it to allow fluent chaining.

            rule_a.set_next(rule_b).set_next(rule_c)

        Args:
            rule: The next ComplianceRule handler

        Returns:
            The next rule (enables fluent chaining)
        """
        self._next = rule
        return rule

    def check(
        self,
        response: str,
        tool_results: Dict[str, str],
        tool_args: Dict[str, Any],
    ) -> List[RuleViolation]:
        """
        Run this rule, then delegate to the rest of the chain.

        Args:
            response:     The LLM's generated response text
            tool_results: Formatted tool output strings, keyed by tool name
            tool_args:    Raw arguments passed to each tool, keyed by tool name

        Returns:
            All violations found by this rule and all subsequent rules
        """
        violations = self._check(response, tool_results, tool_args)
        if self._next:
            violations.extend(self._next.check(response, tool_results, tool_args))
        return violations

    @abstractmethod
    def _check(
        self,
        response: str,
        tool_results: Dict[str, str],
        tool_args: Dict[str, Any],
    ) -> List[RuleViolation]:
        """
        Check this rule's single compliance concern.

        Returns:
            Empty list if compliant; list of RuleViolation if not
        """
        ...


class RateValuePresentRule(ComplianceRule):
    """
    Verifies the exact numeric rate from the tool result appears in the response.

    Severity: ERROR — an LLM that fabricates a rate it never received from
    the tool is a hard compliance failure in a financial context. The rate
    value is the single most critical piece of data in this system.
    """

    _RATE_PATTERN = re.compile(r'\d+\.\d{2,6}')

    def _check(self, response, tool_results, tool_args) -> List[RuleViolation]:
        if not tool_results:
            return []

        tool_rates: set = set()
        for result in tool_results.values():
            for match in self._RATE_PATTERN.findall(result):
                tool_rates.add(match)

        if not tool_rates:
            return []

        for rate in tool_rates:
            if rate in response:
                return []

        return [RuleViolation(
            rule_name="RateValuePresent",
            description=(
                "Response does not contain the rate value returned by the tool. "
                f"Expected one of: {sorted(tool_rates)}"
            ),
            severity="ERROR",
        )]


class SourceAttributionRule(ComplianceRule):
    """
    Verifies the response attributes data to the Bank of Canada.

    Severity: WARNING — important for audit provenance but does not
    constitute a factual error. The LLM is prompted to include attribution
    but this rule catches when it omits it.
    """

    _ATTRIBUTION_PHRASES = [
        "bank of canada",
        "bankofcanada.ca",
    ]

    def _check(self, response, tool_results, tool_args) -> List[RuleViolation]:
        if not tool_results:
            return []

        response_lower = response.lower()
        if any(phrase in response_lower for phrase in self._ATTRIBUTION_PHRASES):
            return []

        return [RuleViolation(
            rule_name="SourceAttribution",
            description=(
                "Response does not cite the Bank of Canada as the data source. "
                "All exchange rate figures must be explicitly attributed."
            ),
            severity="WARNING",
        )]


class DateConsistencyRule(ComplianceRule):
    """
    Verifies the response date is consistent with the queried date.

    Severity: WARNING — a date mismatch is a factual error in an audit trail
    but may occur because the LLM rephrases the date (e.g. "January 15").
    Checks only for year presence to avoid false positives from date reformatting.
    """

    def _check(self, response, tool_results, tool_args) -> List[RuleViolation]:
        if not tool_args:
            return []

        violations = []
        for _, args in tool_args.items():
            expected_date = args.get('date', '')
            if not expected_date:
                continue
            year = expected_date[:4]
            if year and year not in response:
                violations.append(RuleViolation(
                    rule_name="DateConsistency",
                    description=(
                        f"Response does not mention the queried year '{year}' "
                        f"(queried date: {expected_date}). "
                        "The response may reference an incorrect date."
                    ),
                    severity="WARNING",
                ))

        return violations


class CurrencyConsistencyRule(ComplianceRule):
    """
    Verifies the queried currencies appear in the response.

    Severity: WARNING — if a user asks about USD/CAD but the LLM discusses
    a different pair, that is a factual error that must be surfaced in the
    audit trail even if it does not block delivery alone.
    """

    def _check(self, response, tool_results, tool_args) -> List[RuleViolation]:
        if not tool_args:
            return []

        violations = []
        response_upper = response.upper()
        for _, args in tool_args.items():
            for currency_key in ('from_currency', 'to_currency'):
                currency = args.get(currency_key, '').upper()
                if currency and currency not in response_upper:
                    violations.append(RuleViolation(
                        rule_name="CurrencyConsistency",
                        description=(
                            f"Response does not mention {currency}, "
                            f"which was the {currency_key.replace('_', ' ')} "
                            "in the original query."
                        ),
                        severity="WARNING",
                    ))

        return violations
