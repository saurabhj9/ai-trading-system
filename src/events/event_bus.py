"""
Event bus system for distributing trigger events throughout the system.

This module provides a centralized event distribution system that allows
multiple components to subscribe to and handle trigger events efficiently.
"""
import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from .trigger_detector import TriggerEvent, TriggerType, TriggerSeverity

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for event processing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SubscriberType(Enum):
    """Types of event subscribers."""
    TRIGGER_HANDLER = "trigger_handler"
    NOTIFIER = "notifier"
    LOGGER = "logger"
    ANALYZER = "analyzer"
    TRADER = "trader"


@dataclass
class EventSubscription:
    """Represents an event subscription."""
    id: str
    subscriber_id: str
    subscriber_type: SubscriberType
    trigger_types: Set[TriggerType]
    severity_filter: Optional[Set[TriggerSeverity]] = None
    symbol_filter: Optional[Set[str]] = None
    max_queue_size: int = 1000
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)

    def should_handle_event(self, event: TriggerEvent) -> bool:
        """Check if this subscription should handle the given event."""
        # Check trigger type filter
        if event.trigger_type not in self.trigger_types:
            return False

        # Check severity filter
        if self.severity_filter and event.severity not in self.severity_filter:
            return False

        # Check symbol filter
        if self.symbol_filter and event.symbol not in self.symbol_filter:
            return False

        return True


@dataclass
class EventBusStats:
    """Event bus statistics."""
    total_events_published: int = 0
    total_events_delivered: int = 0
    events_dropped: int = 0
    active_subscriptions: int = 0
    subscribers_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    avg_processing_time_ms: float = 0.0
    last_event_time: Optional[datetime] = None


class EventSubscriber(ABC):
    """Abstract base class for event subscribers."""

    def __init__(self, subscriber_id: str, subscriber_type: SubscriberType):
        self.subscriber_id = subscriber_id
        self.subscriber_type = subscriber_type
        self.subscription_id: Optional[str] = None
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time: Optional[datetime] = None

    @abstractmethod
    async def handle_event(self, event: TriggerEvent) -> bool:
        """
        Handle an incoming trigger event.

        Args:
            event: The trigger event to handle

        Returns:
            True if event was handled successfully, False otherwise
        """
        pass

    async def on_subscription_created(self, subscription_id: str):
        """Called when subscription is created."""
        self.subscription_id = subscription_id
        logger.info(f"{self.subscriber_type.value} {self.subscriber_id} subscribed with ID {subscription_id}")

    async def on_subscription_removed(self):
        """Called when subscription is removed."""
        self.subscription_id = None
        logger.info(f"{self.subscriber_type.value} {self.subscriber_id} unsubscribed")


class EventBus:
    """
    Central event bus for distributing trigger events.

    Provides pub/sub functionality with filtering, priority handling,
    and performance monitoring.
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        worker_count: int = 5,
        enable_persistence: bool = False
    ):
        """
        Initialize the event bus.

        Args:
            max_queue_size: Maximum size of the event queue
            worker_count: Number of worker processes for handling events
            enable_persistence: Whether to persist events to storage
        """
        self.max_queue_size = max_queue_size
        self.worker_count = worker_count
        self.enable_persistence = enable_persistence

        # Event queue and worker management
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.workers: List[asyncio.Task] = []
        self.is_running = False

        # Subscription management
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.subscribers: Dict[str, EventSubscriber] = {}

        # Statistics
        self.stats = EventBusStats()
        self.processing_times: List[float] = []

        logger.info(f"Event bus initialized with {worker_count} workers")

    def subscribe(
        self,
        subscriber: EventSubscriber,
        trigger_types: List[TriggerType],
        severity_filter: Optional[List[TriggerSeverity]] = None,
        symbol_filter: Optional[List[str]] = None,
        max_queue_size: int = 1000
    ) -> str:
        """
        Subscribe to trigger events.

        Args:
            subscriber: Event subscriber instance
            trigger_types: List of trigger types to subscribe to
            severity_filter: Optional severity level filter
            symbol_filter: Optional symbol filter
            max_queue_size: Maximum queue size for this subscriber

        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())

        subscription = EventSubscription(
            id=subscription_id,
            subscriber_id=subscriber.subscriber_id,
            subscriber_type=subscriber.subscriber_type,
            trigger_types=set(trigger_types),
            severity_filter=set(severity_filter) if severity_filter else None,
            symbol_filter=set(symbol_filter) if symbol_filter else None,
            max_queue_size=max_queue_size,
            callback=subscriber.handle_event
        )

        self.subscriptions[subscription_id] = subscription
        self.subscribers[subscriber.subscriber_id] = subscriber

        # Update statistics
        self.stats.active_subscriptions = len(self.subscriptions)
        subscriber_type_name = subscriber.subscriber_type.value
        self.stats.subscribers_by_type[subscriber_type_name] = \
            self.stats.subscribers_by_type.get(subscriber_type_name, 0) + 1

        # Notify subscriber
        asyncio.create_task(subscriber.on_subscription_created(subscription_id))

        logger.info(f"Subscribed {subscriber.subscriber_type.value} {subscriber.subscriber_id} to {len(trigger_types)} trigger types")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from trigger events.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if subscription was removed, False if not found
        """
        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]
        subscriber = self.subscribers.get(subscription.subscriber_id)

        # Remove subscription
        del self.subscriptions[subscription_id]

        # Update statistics
        self.stats.active_subscriptions = len(self.subscriptions)
        if subscriber:
            subscriber_type_name = subscriber.subscriber_type.value
            self.stats.subscribers_by_type[subscriber_type_name] = max(
                0, self.stats.subscribers_by_type.get(subscriber_type_name, 0) - 1
            )

        # Notify subscriber
        if subscriber:
            asyncio.create_task(subscriber.on_subscription_removed())

        logger.info(f"Unsubscribed {subscription.subscriber_type.value} {subscription.subscriber_id}")
        return True

    async def publish(self, event: TriggerEvent) -> bool:
        """
        Publish a trigger event to all relevant subscribers.

        Args:
            event: Trigger event to publish

        Returns:
            True if event was queued successfully, False if queue is full
        """
        try:
            # Update statistics
            self.stats.total_events_published += 1
            self.stats.events_by_type[event.trigger_type.value] = \
                self.stats.events_by_type.get(event.trigger_type.value, 0) + 1
            self.stats.events_by_severity[event.severity.value] = \
                self.stats.events_by_severity.get(event.severity.value, 0) + 1
            self.stats.last_event_time = datetime.now()

            # Add to queue
            await self.event_queue.put(event)
            return True

        except asyncio.QueueFull:
            self.stats.events_dropped += 1
            logger.warning(f"Event queue full, dropping event for {event.symbol}")
            return False

    async def publish_batch(self, events: List[TriggerEvent]) -> int:
        """
        Publish multiple events at once.

        Args:
            events: List of trigger events to publish

        Returns:
            Number of events successfully queued
        """
        success_count = 0
        for event in events:
            if await self.publish(event):
                success_count += 1

        return success_count

    async def _worker_loop(self, worker_id: int):
        """
        Main worker loop for processing events.

        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Event bus worker {worker_id} started")

        while self.is_running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                # Process event
                await self._process_event(event, worker_id)

            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")

        logger.info(f"Event bus worker {worker_id} stopped")

    async def _process_event(self, event: TriggerEvent, worker_id: int):
        """
        Process a single event and deliver to relevant subscribers.

        Args:
            event: Event to process
            worker_id: Worker identifier for logging
        """
        start_time = datetime.now()
        delivered_count = 0

        try:
            # Find relevant subscriptions
            relevant_subscriptions = [
                sub for sub in self.subscriptions.values()
                if sub.should_handle_event(event)
            ]

            if not relevant_subscriptions:
                logger.debug(f"No subscribers for {event.trigger_type.value} event for {event.symbol}")
                return

            logger.debug(f"Worker {worker_id} processing {event.trigger_type.value} event for {event.symbol}")

            # Create delivery tasks for each subscriber
            delivery_tasks = []
            for subscription in relevant_subscriptions:
                subscriber = self.subscribers.get(subscription.subscriber_id)
                if subscriber:
                    task = asyncio.create_task(
                        self._deliver_event(subscription, subscriber, event)
                    )
                    delivery_tasks.append(task)

            # Wait for all deliveries to complete
            if delivery_tasks:
                results = await asyncio.gather(*delivery_tasks, return_exceptions=True)

                # Count successful deliveries
                delivered_count = sum(1 for result in results if result is True)

                # Log any exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        subscription = relevant_subscriptions[i]
                        logger.error(f"Error delivering to {subscription.subscriber_id}: {result}")

                        subscriber = self.subscribers.get(subscription.subscriber_id)
                        if subscriber:
                            subscriber.events_failed += 1

            # Update statistics
            self.stats.total_events_delivered += delivered_count

            # Track processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.processing_times.append(processing_time)

            # Keep only last 1000 processing times
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]

            # Update average processing time
            self.stats.avg_processing_time_ms = sum(self.processing_times) / len(self.processing_times)

        except Exception as e:
            logger.error(f"Error processing event for {event.symbol}: {e}")

    async def _deliver_event(
        self,
        subscription: EventSubscription,
        subscriber: EventSubscriber,
        event: TriggerEvent
    ) -> bool:
        """
        Deliver an event to a specific subscriber.

        Args:
            subscription: Event subscription
            subscriber: Event subscriber
            event: Event to deliver

        Returns:
            True if delivery was successful
        """
        try:
            # Call subscriber's handle method
            success = await subscription.callback(event)

            if success:
                subscriber.events_processed += 1
                subscriber.last_event_time = datetime.now()
            else:
                subscriber.events_failed += 1

            return success

        except Exception as e:
            subscriber.events_failed += 1
            logger.error(f"Error delivering event to {subscriber.subscriber_id}: {e}")
            return False

    async def start(self):
        """Start the event bus and worker processes."""
        if self.is_running:
            logger.warning("Event bus is already running")
            return

        self.is_running = True

        # Start worker processes
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker_loop(i + 1))
            self.workers.append(worker)

        logger.info(f"Event bus started with {self.worker_count} workers")

    async def stop(self):
        """Stop the event bus and worker processes."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()

        logger.info("Event bus stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        stats = {
            'is_running': self.is_running,
            'total_events_published': self.stats.total_events_published,
            'total_events_delivered': self.stats.total_events_delivered,
            'events_dropped': self.stats.events_dropped,
            'active_subscriptions': self.stats.active_subscriptions,
            'queue_size': self.event_queue.qsize(),
            'avg_processing_time_ms': self.stats.avg_processing_time_ms,
            'last_event_time': self.stats.last_event_time.isoformat() if self.stats.last_event_time else None,
            'subscribers_by_type': dict(self.stats.subscribers_by_type),
            'events_by_type': dict(self.stats.events_by_type),
            'events_by_severity': dict(self.stats.events_by_severity)
        }

        # Calculate delivery rate
        if self.stats.total_events_published > 0:
            stats['delivery_rate'] = self.stats.total_events_delivered / self.stats.total_events_published
        else:
            stats['delivery_rate'] = 0.0

        # Calculate drop rate
        if self.stats.total_events_published > 0:
            stats['drop_rate'] = self.stats.events_dropped / self.stats.total_events_published
        else:
            stats['drop_rate'] = 0.0

        return stats

    def get_subscriber_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for individual subscribers."""
        subscriber_stats = []

        for subscriber_id, subscriber in self.subscribers.items():
            stats = {
                'subscriber_id': subscriber_id,
                'subscriber_type': subscriber.subscriber_type.value,
                'subscription_id': subscriber.subscription_id,
                'events_processed': subscriber.events_processed,
                'events_failed': subscriber.events_failed,
                'last_event_time': subscriber.last_event_time.isoformat() if subscriber.last_event_time else None
            }

            # Calculate success rate
            total_events = subscriber.events_processed + subscriber.events_failed
            if total_events > 0:
                stats['success_rate'] = subscriber.events_processed / total_events
            else:
                stats['success_rate'] = 0.0

            subscriber_stats.append(stats)

        return subscriber_stats

    def get_subscription_details(self) -> List[Dict[str, Any]]:
        """Get details of all subscriptions."""
        subscriptions = []

        for subscription in self.subscriptions.values():
            details = {
                'subscription_id': subscription.id,
                'subscriber_id': subscription.subscriber_id,
                'subscriber_type': subscription.subscriber_type.value,
                'trigger_types': [tt.value for tt in subscription.trigger_types],
                'severity_filter': [s.value for s in subscription.severity_filter] if subscription.severity_filter else None,
                'symbol_filter': list(subscription.symbol_filter) if subscription.symbol_filter else None,
                'max_queue_size': subscription.max_queue_size,
                'created_at': subscription.created_at.isoformat()
            }
            subscriptions.append(details)

        return subscriptions

    async def clear_queue(self) -> int:
        """Clear the event queue and return number of events cleared."""
        cleared_count = 0

        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break

        logger.info(f"Cleared {cleared_count} events from queue")
        return cleared_count

    async def flush_events(self, timeout_seconds: float = 30.0) -> bool:
        """
        Wait for all queued events to be processed.

        Args:
            timeout_seconds: Maximum time to wait for events to flush

        Returns:
            True if all events were processed, False if timeout occurred
        """
        start_time = datetime.now()

        while not self.event_queue.empty():
            await asyncio.sleep(0.1)

            if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                logger.warning(f"Event flush timeout: {self.event_queue.qsize()} events remaining")
                return False

        logger.info("All events flushed successfully")
        return True


class TriggerHandlerSubscriber(EventSubscriber):
    """
    Base subscriber for handling trigger events with basic filtering.
    """

    def __init__(self, subscriber_id: str, handler_func: Callable[[TriggerEvent], bool]):
        super().__init__(subscriber_id, SubscriberType.TRIGGER_HANDLER)
        self.handler_func = handler_func

    async def handle_event(self, event: TriggerEvent) -> bool:
        """Handle trigger event using provided handler function."""
        try:
            return self.handler_func(event)
        except Exception as e:
            logger.error(f"Error in trigger handler for {event.symbol}: {e}")
            return False


class LoggingSubscriber(EventSubscriber):
    """
    Subscriber that logs trigger events.
    """

    def __init__(self, subscriber_id: str = "logger"):
        super().__init__(subscriber_id, SubscriberType.LOGGER)

    async def handle_event(self, event: TriggerEvent) -> bool:
        """Log trigger event."""
        logger.info(f"TRIGGER: {event.trigger_type.value} - {event.severity.value} - {event.symbol} - {event.description}")
        return True
