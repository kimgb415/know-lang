from mixpanel import Mixpanel
from datetime import datetime
from enum import Enum
from know_lang_bot.configs.chat_config import ChatbotAnalyticsConfig

class ChatFeedback(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ChatAnalytics:
    def __init__(self, config: ChatbotAnalyticsConfig):
        self.mp = Mixpanel(config.api_key)
        
    def track_query(
        self, 
        query: str, 
        client_ip: str
    ):
        """Track query event in Mixpanel"""
        self.mp.track(
            distinct_id=hash(client_ip),  # Hash for privacy
            event_name="chat_query",
            properties={
                "query": query,
            }
        )

    def track_feedback(
        self, 
        like: bool,
        query: str,
        client_ip: str
    ):
        """Track feedback event in Mixpanel"""
        self.mp.track(
            distinct_id=hash(client_ip),  # Hash for privacy
            event_name="chat_feedback",
            properties={
                "feedback": ChatFeedback.POSITIVE.value if like else ChatFeedback.NEGATIVE.value,
                "query": query,
            }
        )