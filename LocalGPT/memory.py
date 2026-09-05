"""
LocalGPT: Memory Management
Manages multi-turn conversation history, context window trimming,
and message state serialization.
"""

from typing import List, Dict, Any, Optional


class ConversationMemory:
    """
    Manages in-memory and persisted conversation state for multi-turn chats.
    """

    def __init__(self, system_prompt: Optional[str] = None, max_history_turns: int = 10):
        self.default_system_prompt = system_prompt or "You are LocalGPT, an intelligent, helpful, and concise AI assistant running 100% locally on this device."
        self.max_history_turns = max_history_turns
        self.messages: List[Dict[str, str]] = []
        self.reset()

    def reset(self, new_system_prompt: Optional[str] = None) -> None:
        """
        Clears the conversation and resets the system prompt.
        """
        if new_system_prompt:
            self.default_system_prompt = new_system_prompt
        self.messages = [{"role": "system", "content": self.default_system_prompt}]

    def add_user_message(self, content: str) -> None:
        """
        Appends a user message to the conversation history.
        """
        if content.strip():
            self.messages.append({"role": "user", "content": content.strip()})

    def add_assistant_message(self, content: str) -> None:
        """
        Appends an assistant message to the conversation history.
        """
        if content.strip():
            self.messages.append({"role": "assistant", "content": content.strip()})

    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Returns the conversation messages list.
        """
        if include_system:
            return self.messages
        return [m for m in self.messages if m.get("role") != "system"]

    def get_trimmed_context(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Returns the conversation history trimmed to the last N turns to avoid context overflow.
        Always preserves the initial system prompt.
        """
        turns = max_turns or self.max_history_turns
        if len(self.messages) <= 1:
            return self.messages

        system_msg = self.messages[0] if self.messages and self.messages[0].get("role") == "system" else None
        conversation_msgs = self.messages[1:] if system_msg else self.messages

        # Each turn is user + assistant (2 messages)
        max_msgs = turns * 2
        trimmed = conversation_msgs[-max_msgs:] if len(conversation_msgs) > max_msgs else conversation_msgs

        if system_msg:
            return [system_msg] + trimmed
        return trimmed

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes memory state to a dictionary.
        """
        return {
            "system_prompt": self.default_system_prompt,
            "max_history_turns": self.max_history_turns,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """
        Restores memory state from a dictionary.
        """
        mem = cls(
            system_prompt=data.get("system_prompt"),
            max_history_turns=data.get("max_history_turns", 10),
        )
        mem.messages = data.get("messages", [])
        return mem
