from enum import Enum
from typing import Protocol
from mixpanel import Mixpanel
from knowlang.configs.chat_config import ChatbotAnalyticsConfig, AnalyticsProvider

class ChatFeedback(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class AnalyticsProviderProtocol(Protocol):
    """Protocol defining what an analytics provider must implement"""
    def track_event(self, event_name: str, distinct_id: str, properties: dict) -> None:
        """Track an event with the analytics provider"""
        ...

class NoopAnalyticsProvider:
    """Provider that does nothing - used when analytics are disabled"""
    def track_event(self, event_name: str, distinct_id: str, properties: dict) -> None:
        pass

class MixpanelProvider:
    """Concrete implementation for Mixpanel"""
    def __init__(self, api_key: str):
        self._mp = Mixpanel(api_key)
        
    def track_event(self, event_name: str, distinct_id: str, properties: dict) -> None:
        self._mp.track(
            distinct_id=distinct_id,
            event_name=event_name,
            properties=properties
        )

def create_analytics_provider(config: ChatbotAnalyticsConfig) -> AnalyticsProviderProtocol:
    """Factory function to create the appropriate analytics provider"""
    if not config.enabled or not config.api_key:
        return NoopAnalyticsProvider()
        
    if config.provider == AnalyticsProvider.MIXPANEL:
        return MixpanelProvider(config.api_key)
        
    raise ValueError(f"Unsupported analytics provider: {config.provider}")

class ChatAnalytics:
    def __init__(self, config: ChatbotAnalyticsConfig):
        self._provider = create_analytics_provider(config)
        
    def track_query(self, query: str, client_ip: str) -> None:
        """Track query event"""
        self._provider.track_event(
            event_name="chat_query",
            distinct_id=str(hash(client_ip)),  # Hash for privacy
            properties={"query": query}
        )

    def track_feedback(self, like: bool, query: str, client_ip: str) -> None:
        """Track feedback event"""
        self._provider.track_event(
            event_name="chat_feedback",
            distinct_id=str(hash(client_ip)),  # Hash for privacy
            properties={
                "feedback": ChatFeedback.POSITIVE.value if like else ChatFeedback.NEGATIVE.value,
                "query": query,
            }
        )