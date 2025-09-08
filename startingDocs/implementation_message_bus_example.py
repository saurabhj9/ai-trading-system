import asyncio
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Callable, Any
from datetime import datetime

@dataclass
class Message:
    from_agent: str
    to_agent: str
    message_type: str
    payload: Dict[str, Any]
    correlation_id: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MessageBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        self.message_queue = asyncio.Queue()
        self.running = False

    async def start(self):
        self.running = True
        asyncio.create_task(self._process_messages())

    async def stop(self):
        self.running = False

    def subscribe(self, agent_name: str, callback: Callable):
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)

    async def publish(self, message: Message):
        await self.message_queue.put(message)

    async def _process_messages(self):
        while self.running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                await self._route_message(message)
                self.message_history.append(message)
            except asyncio.TimeoutError:
                continue

    async def _route_message(self, message: Message):
        if message.to_agent in self.subscribers:
            for callback in self.subscribers[message.to_agent]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)

# Usage example
async def main():
    bus = MessageBus()
    await bus.start()

    # Subscribe to messages
    def handle_message(msg):
        print(f"Received: {msg.from_agent} -> {msg.to_agent}: {msg.payload}")

    bus.subscribe("portfolio_manager", handle_message)

    # Send a message
    await bus.publish(Message(
        from_agent="technical_agent",
        to_agent="portfolio_manager",
        message_type="analysis_complete",
        payload={"signal": "BUY", "confidence": 0.8},
        correlation_id="trade_123"
    ))

    await asyncio.sleep(1)  # Let message process
    await bus.stop()

if __name__ == "__main__":
    asyncio.run(main())
