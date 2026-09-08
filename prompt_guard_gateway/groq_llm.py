"""Groq LLM integration for downstream processing."""

import os
import re
from groq import Groq


def groq_llm(user_input: str, system_prompt: str | None = None) -> str:
    """Call Groq API with the user's message. Optionally pass a system prompt to constrain the model."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set."

    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        content = completion.choices[0].message.content or ""
        # Some models (e.g. qwen) emit <think>...</think> reasoning blocks;
        # strip them so users never see internal reasoning.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        return f"Groq API error: {str(e)}"
