"""
Compliance package — LangGraph-powered compliance validation for FX responses.

Exports the core public surface:
- ComplianceGraphBuilder  : builds the LangGraph StateGraph (Builder)
- ValidationStrategy      : abstract validation strategy (Strategy)
- LenientValidationStrategy / StrictValidationStrategy : concrete strategies
- ComplianceState         : LangGraph state TypedDict
- ValidationResult        : immutable validation outcome (Value Object)
- RuleViolation           : immutable single-rule violation (Value Object)
- ComplianceStatus        : validation lifecycle enum
- ComplianceRule          : abstract Chain of Responsibility handler
- Individual rules        : RateValuePresentRule, SourceAttributionRule,
                            DateConsistencyRule, CurrencyConsistencyRule
"""

from .graph import ComplianceGraphBuilder
from .rules import (
    ComplianceRule,
    CurrencyConsistencyRule,
    DateConsistencyRule,
    RateValuePresentRule,
    SourceAttributionRule,
)
from .state import ComplianceState, ComplianceStatus, RuleViolation, ValidationResult
from .validator import (
    LenientValidationStrategy,
    StrictValidationStrategy,
    ValidationStrategy,
    build_default_rule_chain,
)

__all__ = [
    # Builder
    "ComplianceGraphBuilder",
    # Strategies
    "ValidationStrategy",
    "LenientValidationStrategy",
    "StrictValidationStrategy",
    "build_default_rule_chain",
    # State / Value Objects
    "ComplianceState",
    "ComplianceStatus",
    "ValidationResult",
    "RuleViolation",
    # Rules
    "ComplianceRule",
    "RateValuePresentRule",
    "SourceAttributionRule",
    "DateConsistencyRule",
    "CurrencyConsistencyRule",
]
