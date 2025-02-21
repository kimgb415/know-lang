# github_app/config.py
from pydantic import Field
from pydantic_settings import BaseSettings

from knowlang.configs import generate_model_config


class GitHubAppConfig(BaseSettings):
    model_config = generate_model_config(
        env_file=".env.github",
    )
    GITHUB_APP_ID: str = Field(
        description="GitHub App ID"
    )
    GITHUB_PRIVATE_KEY: str = Field(
        description="GitHub App private key"
    )
    GITHUB_WEBHOOK_SECRET: str = Field(
        description="GitHub webhook secret"
    )
    AWS_ACCESS_KEY_ID: str = Field(
        description="AWS access key ID"
    )
    AWS_SECRET_ACCESS_KEY: str = Field(
        description="AWS secret access key"
    )
    AWS_REGION: str = Field(
        default="us-west-2",
        description="AWS region"
    )