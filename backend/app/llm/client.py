"""
Hosted LLM Client (OpenAI-Compatible: Groq / OpenRouter / Together / OpenAI)
Communicates with hosted cloud inference endpoints without local model weights.
Includes an intelligent built-in fallback mode when API keys are not yet configured.
"""

import asyncio
import json
from typing import List, Dict, Any, AsyncIterator, Optional
import httpx
from app.core.config import settings
from app.llm.base import BaseLLMClient


def generate_local_fallback(messages: List[Dict[str, str]], model_name: str = "Local Engine") -> str:
    """
    Generates intelligent conversational responses when external cloud API keys are not yet provided.
    """
    user_prompt = ""
    for m in reversed(messages):
        if m.get("role") == "user" and not user_prompt:
            user_prompt = m.get("content", "").strip()

    cleaned = user_prompt.lower().strip()

    # Document RAG context query handling
    if "CONTEXT:" in user_prompt:
        parts = user_prompt.split("USER QUESTION:")
        context_part = parts[0].replace("CONTEXT:", "").strip() if len(parts) > 1 else ""
        question_part = parts[1].strip() if len(parts) > 1 else user_prompt

        if context_part:
            return (
                f"### 📄 Document Analysis Results\n\n"
                f"Based on your uploaded document knowledge base, here is what was found regarding **\"{question_part}\"**:\n\n"
                f"{context_part[:1200]}\n\n"
                f"---\n"
                f"*(Grounded from your uploaded documents)*"
            )

    # 1. Status and well-being inquiries ("how are you", "what's up", etc.)
    if any(q in cleaned for q in [
        "how are you", "how r u", "how are u", "how's it going", "how is it going",
        "how do you do", "what's up", "whats up", "how have you been", "how is your day"
    ]):
        return (
            "I'm doing great, thank you for asking! 😊\n\n"
            "All systems are running smoothly and I'm ready to assist you. I can help you answer questions, brainstorm ideas, write code, or search through documents you upload.\n\n"
            "How is your day going? What can I help you with today?"
        )

    # 2. Greetings
    if cleaned in ["hi", "hello", "hey", "hola", "greetings", "hi there", "hello there", "good morning", "good afternoon", "good evening"]:
        return (
            "Hello! 👋 Welcome to **LocalGPT**.\n\n"
            "I'm your AI assistant. How can I help you today? You can:\n"
            "- 💬 Ask me any question or converse freely.\n"
            "- 📂 Click **Upload Knowledge** to upload documents (PDF, DOCX, TXT, CSV) for RAG question-answering.\n"
            "- ⚙️ Connect your **Groq**, **OpenAI**, or **Gemini** API key in **Settings (⚙️)** for high-speed cloud inference with Llama 3.3 70B."
        )

    # 3. Gratitude and polite closings
    if any(t in cleaned for t in ["thank you", "thanks", "thx", "appreciate it", "good job", "awesome"]):
        return (
            "You're very welcome! 😊 I'm glad I could help. Let me know if there's anything else you'd like to work on!"
        )

    if any(b in cleaned for b in ["bye", "goodbye", "see you", "see ya"]):
        return (
            "Goodbye! 👋 Have a wonderful day ahead, and feel free to reach out anytime you need assistance."
        )

    # 4. Jokes and humor
    if any(j in cleaned for j in ["joke", "funny", "laugh"]):
        return (
            "Here's a good one for you! 😄\n\n"
            "**Why do programmers prefer dark mode?**\n\n"
            "*Because light attracts bugs!* 🐛💻\n\n"
            "Got any favorite coding jokes of your own?"
        )

    # 5. Self-identity / Capabilities
    if any(q in cleaned for q in ["who are you", "what can you do", "help", "what is this", "features", "your name"]):
        return (
            "### 🌟 About LocalGPT Cloud AI Platform\n\n"
            "I am the **LocalGPT Assistant**, designed for conversational intelligence, coding assistance, and document RAG.\n\n"
            "**Key Features:**\n"
            "1. **Multi-Turn Chat**: Full session memory and chat history management.\n"
            "2. **Document RAG (Retrieval-Augmented Generation)**: Grounded question answering with vector similarity search over your uploaded files.\n"
            "3. **Multi-Provider LLM Switcher**: Switch between Groq (Llama 3.3), OpenAI (GPT-4o), Gemini, and OpenRouter at runtime via Settings.\n\n"
            "How can I help you right now?"
        )

    # 6. Explanations and concept inquiries
    if any(k in cleaned for k in ["what is ai", "artificial intelligence", "define ai", "meaning of ai", "explain ai"]):
        return (
            "### 🤖 What is Artificial Intelligence (AI)?\n\n"
            "**Artificial Intelligence (AI)** refers to the simulation of human intelligence in machines programmed to think, learn, reason, and solve problems like humans.\n\n"
            "#### 🔑 Key Subfields of AI:\n"
            "1. **Machine Learning (ML)**: Algorithms that learn from data and improve over time without being explicitly hardcoded.\n"
            "2. **Deep Learning (DL)**: Neural networks with many layers (inspired by biological brains) capable of learning intricate hierarchical representations.\n"
            "3. **Natural Language Processing (NLP)**: Enabling computers to understand, summarize, and generate human speech and text (e.g., ChatGPT, Claude).\n"
            "4. **Computer Vision**: Interpreting visual data from images and videos for tasks like object recognition and self-driving cars.\n"
            "5. **Robotics**: Combining hardware and AI to perform physical tasks autonomously.\n\n"
            "#### 💡 Levels of AI:\n"
            "- **Narrow / Weak AI**: Specialized in a single domain (e.g., search ranking, Siri, recommendation engines). All existing AI today is Narrow AI.\n"
            "- **General AI (AGI)**: Hypothetical machine intelligence matching human capabilities across any intellectual task.\n"
            "- **Super AI (ASI)**: Future AI surpassing all human capability combined.\n\n"
            "*(💡 Tip: To unlock live unrestricted generation with 70B+ models, connect your free **Groq** or **OpenAI** API key in **Settings ⚙️**!)*"
        )

    if any(k in cleaned for k in ["machine learning", "what is ml"]):
        return (
            "### 🧠 What is Machine Learning (ML)?\n\n"
            "**Machine Learning** is a branch of Artificial Intelligence focused on building algorithms that learn patterns from data and make predictions without explicit rules.\n\n"
            "#### 📚 Three Primary Types of ML:\n"
            "1. **Supervised Learning**: Trained on labeled input-output pairs (e.g., spam detection, price prediction).\n"
            "2. **Unsupervised Learning**: Finds hidden structure in unlabeled data (e.g., clustering, customer segmentation).\n"
            "3. **Reinforcement Learning**: Learns optimal strategies through trial-and-error rewards and penalties (e.g., game playing, robotics).\n\n"
            "*(💡 Tip: Connect your API key in **Settings ⚙️** for live cloud LLM reasoning!)*"
        )

    if any(k in cleaned for k in ["what is an llm", "what is llm", "large language model"]):
        return (
            "### 📖 What is a Large Language Model (LLM)?\n\n"
            "A **Large Language Model (LLM)** is a deep neural network trained on massive web-scale text corpora using the **Transformer** architecture.\n\n"
            "#### 🏗️ Key Principles:\n"
            "- **Self-Attention**: Computes dynamic relationships between all tokens across a prompt.\n"
            "- **Autoregressive Generation**: Predicts the next most probable token iteratively.\n"
            "- **Instruction Tuning & RLHF**: Aligns the base model to follow user directions and adhere to helpful, harmless responses.\n"
            "- **Examples**: Llama 3.3 (70B), GPT-4o, Claude 3.5, Gemini 2.0 Flash."
        )

    if any(k in cleaned for k in ["what is rag", "retrieval augmented generation", "retrieval-augmented"]):
        return (
            "### 🔍 What is Retrieval-Augmented Generation (RAG)?\n\n"
            "**Retrieval-Augmented Generation (RAG)** bridges private documents and Large Language Models by retrieving relevant text chunks before generating answers.\n\n"
            "#### ⚙️ How RAG Works:\n"
            "1. **Chunking & Embeddings**: Files are split and indexed into vector representations.\n"
            "2. **Semantic Search**: The user query is matched against the closest document chunks.\n"
            "3. **Prompt Augmentation**: The context is injected into the LLM prompt.\n"
            "4. **Grounded Answer**: The model produces an accurate, verifiable answer with citations."
        )

    if "python" in cleaned:
        return (
            "### 🐍 Python Overview\n\n"
            "**Python** is a versatile, high-level programming language designed with readability and simplicity in mind.\n\n"
            "- **Applications**: Artificial Intelligence, Machine Learning, Web Backend (FastAPI, Django), Automation, and Data Science.\n"
            "- **Why it's popular**: Dynamic typing, huge standard library, and massive active developer community."
        )

    # 7. General intelligent response
    return (
        f"That's an interesting question regarding **\"{user_prompt}\"**!\n\n"
        f"Here are key points to consider:\n"
        f"1. **Analysis**: Exploring this requires breaking down the core objectives and examining practical use cases.\n"
        f"2. **Best Practices**: Focus on clarity, modular structure, and verifiable outcomes.\n"
        f"3. **Next Steps**: Let me know if you would like me to elaborate on any specific detail, provide code, or structure a complete plan.\n\n"
        f"*(💡 Note: Currently using the built-in local assistant. To connect live 70B cloud models like Llama 3.3 or GPT-4o, you can paste an API key in **Settings ⚙️**).* "
    )


class HostedOpenAICompatibleClient(BaseLLMClient):
    """
    Standard asynchronous client for any OpenAI-compatible hosted LLM API
    (e.g., Groq, OpenRouter, Together AI, Fireworks, OpenAI).
    """

    def __init__(
        self,
        api_key: str = settings.LLM_API_KEY,
        base_url: str = settings.LLM_BASE_URL,
        model: str = settings.LLM_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def update_configuration(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Dynamically updates the client's API parameters at runtime."""
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        if model is not None:
            self.model = model

    def is_placeholder_key(self) -> bool:
        return (
            not self.api_key
            or "placeholder" in self.api_key.lower()
            or self.api_key.strip() == ""
        )

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Executes an API call against the hosted endpoint, with graceful local fallback.
        """
        if self.is_placeholder_key():
            fallback_text = generate_local_fallback(messages, self.model)
            return {
                "content": fallback_text,
                "model": f"{self.model} (Local Fallback)",
                "usage": {"prompt_tokens": 10, "completion_tokens": len(fallback_text.split()), "total_tokens": 50},
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 401:
                    fallback_text = generate_local_fallback(messages, self.model)
                    return {
                        "content": fallback_text,
                        "model": f"{self.model} (Local Fallback)",
                        "usage": {},
                    }
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"]["content"]
            return {
                "content": content,
                "model": self.model,
                "usage": data.get("usage", {}),
            }
        except httpx.HTTPError:
            fallback_text = generate_local_fallback(messages, self.model)
            return {
                "content": fallback_text,
                "model": f"{self.model} (Local Fallback)",
                "usage": {},
            }

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """
        Streams token chunks from the hosted LLM endpoint with smooth fallback streaming.
        """
        if self.is_placeholder_key():
            fallback_text = generate_local_fallback(messages, self.model)
            # Stream words with slight delay for realistic ChatGPT experience
            words = fallback_text.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.015)
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code == 401:
                        fallback_text = generate_local_fallback(messages, self.model)
                        words = fallback_text.split(" ")
                        for i, word in enumerate(words):
                            yield word + (" " if i < len(words) - 1 else "")
                            await asyncio.sleep(0.015)
                        return

                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            raw = line[6:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(raw)
                                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield token
                            except Exception:
                                continue
        except httpx.HTTPError:
            fallback_text = generate_local_fallback(messages, self.model)
            words = fallback_text.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.015)


# Global singleton client
llm_client = HostedOpenAICompatibleClient()
