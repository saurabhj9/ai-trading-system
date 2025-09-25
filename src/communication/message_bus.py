"""
Simple in-memory message bus for the AI Trading System.
"""
from typing import Any, Callable, Dict, List


class MessageBus:
    """
    A simple in-memory message bus for event-driven communication.
    """

    def __init__(self):
        """Initializes the MessageBus."""
        self.listeners: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribes a callback to an event type.

        Args:
            event_type: The type of event to subscribe to.
            callback: The function to call when the event is published.
        """
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)

    def publish(self, event_type: str, message: Any):
        """
        Publishes an event to all subscribed listeners.

        Args:
            event_type: The type of event being published.
            message: The message to send to the listeners.
        """
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                callback(message)
