import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = (
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
)


def missing_required_env_vars() -> list[str]:
    return [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]


def has_database_url() -> bool:
    return bool(os.getenv("DATABASE_URL"))
