"""
LLM Client for the AI Trading System, using OpenRouter.
"""
import hashlib
from typing import Any, Optional

import httpx
from openai import OpenAI

from src.config.settings import settings
from src.data.cache import CacheManager


class LLMClient:
    """
    A client for interacting with various large language models via OpenRouter.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the LLMClient for OpenRouter.

        Args:
            api_key: The OpenRouter API key. If not provided, it will be read
                      from the OPENROUTER_API_KEY environment variable.
        """
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        # Configure httpx client with connection pooling limits for performance
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0,
            )
        )

        self.client = OpenAI(
            base_url=settings.llm.BASE_URL,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": settings.llm.SITE_URL,
                "X-Title": settings.llm.APP_NAME,
            },
            http_client=http_client,
        )
        self.cache = CacheManager()
        self.last_usage = None

    async def generate(self, model: str, prompt: str, system_prompt: str) -> str:
        """
        Generates a response from the language model via OpenRouter.

        Args:
            model: The name of the language model to use (e.g., "anthropic/claude-3-haiku").
            prompt: The user-level prompt for the LLM.
            system_prompt: The system-level prompt for the LLM.

        Returns:
            The textual response from the language model.
        """
        # Create cache key from inputs
        cache_key_data = f"{model}:{system_prompt}:{prompt}"
        cache_key = hashlib.sha256(cache_key_data.encode()).hexdigest()

        # Check cache first
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            print(f"LLM Cache HIT for key: {cache_key}")
            return cached_response

        print(f"LLM Cache MISS for key: {cache_key}")

        # Generate response from LLM
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=model,
        )
        response = chat_completion.choices[0].message.content
        self.last_usage = chat_completion.usage

        # Cache the response
        self.cache.set(cache_key, response, settings.llm.CACHE_TTL_SECONDS)

        return response
