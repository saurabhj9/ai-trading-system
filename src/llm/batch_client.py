"""
Batch Request Manager for handling multiple LLM requests efficiently.
"""
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.utils.logging import get_logger
from src.utils.performance import time_function

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """Represents a single request in a batch."""
    request_id: str
    model: str
    prompt: str
    system_prompt: str


class BatchRequestManager:
    """
    Manages queuing and batching of LLM requests for optimized processing.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the BatchRequestManager.

        Args:
            llm_client: The LLM client instance to use for processing batches.
        """
        self.llm_client = llm_client
        self.request_queue: List[BatchRequest] = []
        self.lock = asyncio.Lock()

    async def add_request(
        self,
        request_id: str,
        model: str,
        prompt: str,
        system_prompt: str
    ) -> None:
        """
        Adds a request to the batch queue.

        Args:
            request_id: Unique identifier for the request.
            model: The LLM model to use.
            prompt: The user prompt.
            system_prompt: The system prompt.
        """
        async with self.lock:
            self.request_queue.append(BatchRequest(
                request_id=request_id,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt
            ))
            logger.debug(f"Added request {request_id} to batch queue")

    @time_function(operation_name="batch_process_requests")
    async def process_batch(self) -> Dict[str, str]:
        """
        Processes all queued requests in a batch and returns the results.

        Returns:
            Dictionary mapping request_id to response string.
        """
        async with self.lock:
            if not self.request_queue:
                logger.debug("No requests in queue to process")
                return {}

            requests = self.request_queue.copy()
            self.request_queue.clear()

        logger.info(f"Processing batch of {len(requests)} requests")

        # Prepare batch data for LLM client
        batch_data = [
            (req.model, req.prompt, req.system_prompt)
            for req in requests
        ]

        try:
            # Process batch through LLM client
            responses = await self.llm_client.generate_batch(batch_data)

            # Map responses back to request IDs
            result = {}
            for req, response in zip(requests, responses):
                result[req.request_id] = response

            logger.info(f"Successfully processed batch of {len(requests)} requests")
            return result

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return error responses for all requests in batch
            error_msg = f"Batch processing failed: {str(e)}"
            return {req.request_id: error_msg for req in requests}

    async def get_queue_size(self) -> int:
        """
        Returns the current number of requests in the queue.

        Returns:
            Number of queued requests.
        """
        async with self.lock:
            return len(self.request_queue)
