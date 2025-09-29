"""
LLM Client for the AI Trading System, using OpenRouter.
"""
import asyncio
import hashlib
from typing import Any, List, Optional, Tuple

import httpx
from openai import OpenAI

from src.config.settings import settings
from src.data.cache import CacheManager
from src.utils.logging import get_logger
from src.utils.performance import time_function

logger = get_logger(__name__)


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

    @time_function(operation_name="llm_generate_batch")
    async def generate_batch(self, requests: List[Tuple[str, str, str]]) -> List[str]:
        """
        Generates responses for multiple requests concurrently.

        Args:
            requests: List of tuples (model, prompt, system_prompt)

        Returns:
            List of response strings in the same order as requests.
        """
        logger.info(f"Processing batch of {len(requests)} LLM requests concurrently")

        # Create tasks for concurrent execution
        tasks = [
            self.generate(model, prompt, system_prompt)
            for model, prompt, system_prompt in requests
        ]

        try:
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in responses
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Error in batch request {i}: {response}")
                    processed_responses.append(f"Error: {str(response)}")
                else:
                    processed_responses.append(response)

            logger.info(f"Successfully processed batch of {len(requests)} requests")
            return processed_responses

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return error messages for all requests
            return [f"Batch error: {str(e)}"] * len(requests)
