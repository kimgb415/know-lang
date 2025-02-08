from pydantic import Field
from pydantic_settings import BaseSettings
from enum import Enum


class AnalyticsProvider(str, Enum):
    MIXPANEL = "mixpanel"


class ChatbotAnalyticsConfig(BaseSettings):
    enabled: bool = Field(
        default=False,
        description="Enable analytics tracking"
    )
    provider: AnalyticsProvider = Field(
        default=AnalyticsProvider.MIXPANEL,
        description="Analytics provider to use for tracking feedback"
    )

    api_key: str = Field(
        default=None,
        description="api key for feedback tracking"
    )