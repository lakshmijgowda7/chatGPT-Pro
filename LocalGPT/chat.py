"""
LocalGPT: Chat Engine
Coordinates multi-turn conversational chat sessions, persona configurations,
and local response generation.
"""

from typing import Dict, Any, List, Optional, Iterator
from model import (
    load_model_and_tokenizer,
    generate_chat_response,
    stream_chat_response,
    clear_memory_cache,
)
from tokenizer import format_chat_prompt
from memory import ConversationMemory

# Preset personas
DEFAULT_PERSONAS: Dict[str, str] = {
    "General Assistant": "You are LocalGPT, a helpful, intelligent, and friendly AI assistant. Give clear, accurate, and concise answers.",
    "Python Tutor": "You are an expert Python programming instructor. Provide clear explanations, clean code examples, and best practices.",
    "Concise Analyst": "You are a concise data and logic analyst. Deliver direct, bullet-pointed, fluff-free responses.",
    "Creative Writer": "You are an imaginative creative writer. Respond with engaging narratives, vivid descriptions, and expressive tone.",
}


class ChatSession:
    """
    Manages an active conversational chat session with local inference.
    """

    def __init__(
        self,
        persona: str = "General Assistant",
        custom_system_prompt: Optional[str] = None,
        max_history_turns: int = 10,
    ):
        system_prompt = custom_system_prompt or DEFAULT_PERSONAS.get(persona, DEFAULT_PERSONAS["General Assistant"])
        self.memory = ConversationMemory(system_prompt=system_prompt, max_history_turns=max_history_turns)
        self.persona = persona
        self.model = None
        self.tokenizer = None

    def _ensure_model_loaded(self):
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_model_and_tokenizer()

    def set_persona(self, persona_name: str) -> None:
        """
        Updates the active persona and resets the system prompt.
        """
        prompt = DEFAULT_PERSONAS.get(persona_name, DEFAULT_PERSONAS["General Assistant"])
        self.persona = persona_name
        self.memory.reset(new_system_prompt=prompt)

    def set_custom_system_prompt(self, system_prompt: str) -> None:
        """
        Sets a custom system prompt and resets conversation history.
        """
        self.persona = "Custom"
        self.memory.reset(new_system_prompt=system_prompt)

    def ask(
        self,
        user_message: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Executes a single conversational turn synchronously.
        """
        self._ensure_model_loaded()
        self.memory.add_user_message(user_message)

        messages_to_send = self.memory.get_trimmed_context()
        formatted_prompt = format_chat_prompt(messages_to_send, tokenizer=self.tokenizer)

        result = generate_chat_response(
            formatted_prompt=formatted_prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=(temperature > 0),
        )

        if not result.get("error"):
            self.memory.add_assistant_message(result["response"])

        clear_memory_cache()
        return result

    def stream_ask(
        self,
        user_message: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 256,
    ) -> Iterator[str]:
        """
        Streams response tokens iteratively.
        """
        self._ensure_model_loaded()
        self.memory.add_user_message(user_message)

        messages_to_send = self.memory.get_trimmed_context()
        formatted_prompt = format_chat_prompt(messages_to_send, tokenizer=self.tokenizer)

        full_response = ""
        for chunk in stream_chat_response(
            formatted_prompt=formatted_prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=(temperature > 0),
        ):
            full_response += chunk
            yield chunk

        self.memory.add_assistant_message(full_response.strip())
        clear_memory_cache()
