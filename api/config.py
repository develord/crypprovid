"""
Configuration Settings
======================
"""
import os
from pathlib import Path
from typing import List


class Settings:
    """Application settings"""

    # Server configuration
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    # Support both PORT (Render.com) and API_PORT (local dev)
    PORT: int = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:8081",  # React Native Metro
        "http://localhost:19000",  # Expo
        "http://localhost:19001",
        "http://localhost:19002",
        "*"  # Allow all in dev - RESTRICT IN PRODUCTION
    ]

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models" / "xgboost_v6"

    # Model configuration
    MODEL_VERSION: str = "v6"
    TIMEFRAME: str = "1d"

    # API configuration
    CACHE_TTL: int = 60  # seconds
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 10  # seconds

    # Binance API (optionnel - pour prix en temps réel)
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")


settings = Settings()
