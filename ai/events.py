"""
Observer pattern implementation for assistant lifecycle events.

GoF Observer Pattern:
- Subject: EventBus — maintains a list of observers and notifies them of events
- Observer: EventObserver — abstract interface for event handlers
- ConcreteObserver: AuditLogger — logs events for regulatory compliance

In regulated financial environments, all AI tool-calls and decisions must be
logged and traceable. This event system enables that without coupling the
assistant logic to specific logging implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import logging


class EventType(Enum):
    """Types of events emitted during assistant operations."""
    QUERY_RECEIVED = auto()
    TOOL_CALLED = auto()
    TOOL_RESULT = auto()
    RESPONSE_GENERATED = auto()
    ERROR_OCCURRED = auto()
    HISTORY_CLEARED = auto()


@dataclass
class AssistantEvent:
    """
    Immutable event emitted during assistant operations.

    Attributes:
        event_type: The category of event
        timestamp: When the event occurred (UTC)
        data: Arbitrary payload with event-specific details
    """
    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.timestamp.isoformat()}] {self.event_type.name}: {self.data}"


class EventObserver(ABC):
    """
    Abstract observer interface (GoF Observer pattern).

    Subclasses implement on_event() to react to assistant lifecycle events.
    """

    @abstractmethod
    def on_event(self, event: AssistantEvent) -> None:
        """
        Handle an assistant event.

        Args:
            event: The event to process
        """
        ...


class EventBus:
    """
    Central event dispatcher — the Subject in GoF Observer pattern.

    Manages observer subscriptions and dispatches events to all
    registered observers. Thread-safe for single-threaded async use.

    Usage:
        bus = EventBus()
        bus.subscribe(AuditLogger())
        bus.publish(AssistantEvent(EventType.QUERY_RECEIVED, data={"query": "..."}))
    """

    def __init__(self):
        self._observers: List[EventObserver] = []

    def subscribe(self, observer: EventObserver) -> None:
        """Register an observer to receive events."""
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe(self, observer: EventObserver) -> None:
        """Remove an observer from the subscription list."""
        self._observers = [o for o in self._observers if o is not observer]

    def publish(self, event: AssistantEvent) -> None:
        """Dispatch an event to all registered observers."""
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception:
                # Observers must not break the main flow
                logging.getLogger(__name__).exception(
                    f"Observer {type(observer).__name__} failed on {event.event_type.name}"
                )

    @property
    def observer_count(self) -> int:
        """Number of currently registered observers."""
        return len(self._observers)


class AuditLogger(EventObserver):
    """
    Concrete Observer: logs all assistant events for audit compliance.

    In regulated financial environments, this provides the traceability
    required for AI-assisted decisions. Each tool call, result, and error
    is logged with timestamps for post-hoc analysis.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger("audit")

    def on_event(self, event: AssistantEvent) -> None:
        """Log the event with appropriate severity and formatting."""
        handlers = {
            EventType.QUERY_RECEIVED: self._log_query,
            EventType.TOOL_CALLED: self._log_tool_call,
            EventType.TOOL_RESULT: self._log_tool_result,
            EventType.RESPONSE_GENERATED: self._log_response,
            EventType.ERROR_OCCURRED: self._log_error,
            EventType.HISTORY_CLEARED: self._log_clear,
        }
        handler = handlers.get(event.event_type)
        if handler:
            handler(event)

    def _log_query(self, event: AssistantEvent) -> None:
        self._logger.info(
            "AUDIT | Query received: %s", event.data.get('query', '')[:200]
        )

    def _log_tool_call(self, event: AssistantEvent) -> None:
        self._logger.info(
            "AUDIT | Tool called: %s with args: %s",
            event.data.get('tool_name'),
            event.data.get('args')
        )

    def _log_tool_result(self, event: AssistantEvent) -> None:
        self._logger.info(
            "AUDIT | Tool result: %s", event.data.get('result', '')[:200]
        )

    def _log_response(self, event: AssistantEvent) -> None:
        self._logger.info(
            "AUDIT | Response generated (%d chars)",
            len(event.data.get('response', ''))
        )

    def _log_error(self, event: AssistantEvent) -> None:
        self._logger.error(
            "AUDIT | Error: %s", event.data.get('error')
        )

    def _log_clear(self, event: AssistantEvent) -> None:
        self._logger.info("AUDIT | Chat history cleared")
