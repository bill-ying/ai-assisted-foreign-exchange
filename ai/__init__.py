"""
AI Package — AI assistant components for FX rate queries.

Provides the Facade (FxAssistant), Observer (EventBus), Strategy (formatters,
chat history), and Template Method (tools) implementations.
"""

from .fx_assistant import FxAssistant
from .events import EventBus, EventType, AssistantEvent, EventObserver, AuditLogger
from .result_formatter import ResultFormatter, LLMResultFormatter, HumanResultFormatter
from .chat_history import ChatHistory, InMemoryChatHistory
from .tools import ToolRegistry, FxRateTool

__all__ = [
    'FxAssistant',
    'EventBus',
    'EventType',
    'AssistantEvent',
    'EventObserver',
    'AuditLogger',
    'ResultFormatter',
    'LLMResultFormatter',
    'HumanResultFormatter',
    'ChatHistory',
    'InMemoryChatHistory',
    'ToolRegistry',
    'FxRateTool',
]
