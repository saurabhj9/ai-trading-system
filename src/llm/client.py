"""
LLM Client for the AI Trading System, using OpenRouter.
"""
import os
from typing import Any, Optional

from openai import OpenAI

class LLMClient:
    """
    A client for interacting with various large language models via OpenRouter.
    """
    def __init__(self, api_key: Optional[str] = None, site_url: Optional[str] = "https://my-site.com", app_name: Optional[str] = "AI Trading System"):
        """
        Initializes the LLMClient for OpenRouter.

        Args:
            api_key: The OpenRouter API key. If not provided, it will be read
                     from the OPENROUTER_API_KEY environment variable.
            site_url: Your site's URL, for OpenRouter analytics.
            app_name: Your app's name, for OpenRouter analytics.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
        )

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
        return chat_completion.choices[0].message.content
