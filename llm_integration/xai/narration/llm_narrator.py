# Copyright 2025 - LLM Narrator for Attention-Grounded XRL Pipeline
"""
Module 11: LLMNarrator

Makes an HTTP API call to OpenRouter (or compatible endpoint) and returns
the narration string.  This is the only module that makes a network call.
"""

import json
import os
import re
from typing import Any, Dict, Optional

import requests


def narrate(
    system_prompt: str,
    user_prompt: str,
    config: Dict[str, Any],
) -> tuple[str, float]:
    """
    Call the LLM API and return the narration string.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User-level prompt with report data.
        config: Pipeline config dict — uses the ``llm`` sub-dict:
            - ``provider``: currently only ``"openrouter"``
            - ``base_url``: API endpoint
            - ``model``: model identifier (e.g. ``"qwen/qwen3-4b"``)
            - ``max_tokens``: max response length
            - ``api_key_env``: environment variable name holding the API key

    Returns:
        Tuple of (Narration string, Response time in seconds).
    """
    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
    model = llm_cfg.get("model", "qwen/qwen3-4b")
    max_tokens = llm_cfg.get("max_tokens", 256)
    api_key_env = llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")
    enable_thinking = llm_cfg.get("enable_thinking", False)

    # Fallback to the literal string if it looks like an API key, so raw keys work directly.
    api_key = os.environ.get(api_key_env, None)
    if not api_key:
        api_key = api_key_env if api_key_env.startswith(("sk-", "gsk_")) else ""

    if not api_key:
        return (
            f"[LLMNarrator] ERROR: API key not found in env var '{api_key_env}' and doesn't look like a raw key. "
            f"Set it with: export {api_key_env}=sk-...",
            0.0
        )

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        resp = requests.post(
            base_url,
            json=payload,
            headers=headers,
            timeout=120
        )
        resp.raise_for_status()

        body = resp.json()
        

        # 1. Parse local HTTP time (fallback)
        api_time = resp.elapsed.total_seconds()

        # 2. Prefer Groq's exact server-side inference time if available
        if "x_groq" in body and "usage" in body["x_groq"]:
            if "total_time" in body["x_groq"]["usage"]:
                api_time = body["x_groq"]["usage"]["total_time"]

        choices = body.get("choices", [])
        if choices:
            choice = choices[0]
            msg = choice.get("message", {})
            finish_reason = choice.get("finish_reason")

            if finish_reason == "length":
                print(f"[LLMNarrator WARNING] Generation cut off by max_tokens limit ({max_tokens}). Model ran out of token budget during thinking!")

            # Some thinking models put content in 'reasoning_content' and
            # leave 'content' as null when thinking is disabled.
            text = msg.get("content") or ""

            # Fallback: check for reasoning_content if content is empty
            if not text.strip():
                text = msg.get("reasoning_content") or ""

            # Strip <think>...</think> blocks that some models include
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            if not text:
                return "[LLMNarrator] WARNING: Model returned empty content after stripping think tags.", round(api_time, 4)

            return text, round(api_time, 4)
        
        print(f"[LLMNarrator DEBUG] Raw API response body (no choices):\n{json.dumps(body, indent=2)}")
        return "[LLMNarrator] WARNING: Empty response from API.", round(api_time, 4)

    except requests.exceptions.HTTPError as e:
        return f"[LLMNarrator] HTTP Error: {e.response.status_code} - {e.response.text[:200]}", 0.0
    except Exception as e:
        return f"[LLMNarrator] ERROR: {e}", 0.0
