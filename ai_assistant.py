"""
Backward-compatibility shim for ai_assistant.py.

The FxAIAssistant class has been refactored into the ai/ package
using GoF design patterns. This module preserves the original import
path so that main.py and any other clients continue to work unchanged.

    from ai_assistant import FxAIAssistant  # still works
"""

from ai.fx_assistant import FxAssistant
from fx_service import FxRateService


class FxAIAssistant:
    """
    Backward-compatible wrapper around the refactored FxAssistant.

    Preserves the original constructor signature and public API.
    Delegates all behavior to FxAssistant (GoF Facade).
    """

    def __init__(self, fx_service=None):
        """
        Initialize the AI assistant.

        Args:
            fx_service: Optional FxRateService instance. Creates a new one if not provided.
        """
        if fx_service is not None:
            # Wrap the provided service — the FxAssistant needs its own wiring
            self._assistant = FxAssistant.create()
            # Replace the service inside the assistant
            self._assistant._fx_service = fx_service
            self._assistant._owns_service = False
        else:
            self._assistant = FxAssistant.create()

    def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant's response."""
        return self._assistant.chat(user_message)

    def clear_history(self):
        """Clear the conversation history."""
        self._assistant.clear_history()

    def close(self):
        """Clean up resources."""
        self._assistant.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
