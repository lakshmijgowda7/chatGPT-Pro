"""
Conversation Memory Service
Manages context window trimming and message history formatting.
Reuses proven multi-turn sliding window logic from Project 2.
"""

from typing import List, Dict, Any
from app.models.message import Message


class MemoryService:
    @staticmethod
    def format_history_for_llm(
        messages: List[Message],
        system_prompt: str,
        max_turns: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Formats message list into standard OpenAI role/content dictionary format
        with sliding-window context trimming to prevent token overflow.
        """
        formatted = [{"role": "system", "content": system_prompt}]
        
        # Max turns = user + assistant pairs (2 messages per turn)
        max_msgs = max_turns * 2
        recent_msgs = messages[-max_msgs:] if len(messages) > max_msgs else messages

        for m in recent_msgs:
            if m.role in ["user", "assistant", "system"]:
                formatted.append({"role": m.role, "content": m.content})

        return formatted
