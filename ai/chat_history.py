"""
Chat history management (GoF Strategy pattern for storage backend).

The ChatHistory abstraction allows swapping storage implementations
(in-memory, database, file-based) without changing the assistant logic.
"""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import BaseMessage


class ChatHistory(ABC):
    """
    Abstract chat history storage (GoF Strategy pattern).

    Defines the interface for storing and retrieving conversation messages.
    Concrete implementations determine the storage backend.
    """

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the history.

        Args:
            message: A LangChain message (Human, AI, Tool, System)
        """
        ...

    @abstractmethod
    def get_messages(self) -> List[BaseMessage]:
        """
        Retrieve all messages in chronological order.

        Returns:
            List of messages
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the history."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of messages currently stored."""
        ...


class InMemoryChatHistory(ChatHistory):
    """
    In-memory chat history (default implementation).

    Simple list-based storage suitable for single-session conversations.
    For production use with persistence, implement a database-backed
    ChatHistory subclass.
    """

    def __init__(self):
        self._messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def get_messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    @property
    def size(self) -> int:
        return len(self._messages)
