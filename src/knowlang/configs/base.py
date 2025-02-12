from pydantic_settings import SettingsConfigDict
from pathlib import Path

def generate_model_config(env_dir : Path = Path('settings'), env_file: Path = '.env', env_prefix : str = '') -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=str(env_dir / env_file),
        env_prefix=env_prefix,
        env_file_encoding='utf-8',
        env_nested_delimiter='__'
    )