"""
Pydantic Models for API
=======================
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class Probabilities(BaseModel):
    """Prediction probabilities"""
    buy: float = Field(..., description="Probabilité BUY (0-1)", ge=0, le=1)
    sell: float = Field(..., description="Probabilité SELL (0-1)", ge=0, le=1)
    hold: float = Field(..., description="Probabilité HOLD (0-1)", ge=0, le=1)


class RiskManagement(BaseModel):
    """Risk management metrics"""
    target_price: Optional[float] = Field(None, description="Prix cible", gt=0)
    stop_loss: Optional[float] = Field(None, description="Prix stop loss", gt=0)
    take_profit: Optional[float] = Field(None, description="Prix take profit", gt=0)
    risk_reward_ratio: Optional[float] = Field(None, description="Ratio Risk:Reward", gt=0)
    potential_gain_percent: Optional[float] = Field(None, description="Gain potentiel en %")
    potential_loss_percent: Optional[float] = Field(None, description="Perte potentielle en %")


class PredictionResponse(BaseModel):
    """Response model for single crypto prediction"""
    crypto: str = Field(..., description="Crypto ID (bitcoin, ethereum, etc.)")
    symbol: str = Field(..., description="Trading symbol (BTCUSDT, ETHUSDT, etc.)")
    name: str = Field(..., description="Nom complet (Bitcoin, Ethereum, etc.)")
    signal: str = Field(..., description="Signal de trading: BUY, SELL, ou HOLD")
    confidence: float = Field(..., description="Confiance (probabilité P(TP) pour V11)", ge=0, le=1)
    probabilities: Optional[Probabilities] = Field(None, description="Probabilités détaillées (V6 uniquement)")
    threshold: Optional[float] = Field(None, description="Threshold optimal utilisé (V11)", ge=0, le=1)
    current_price: float = Field(..., description="Prix actuel en USDT", gt=0)
    risk_management: Optional[RiskManagement] = Field(None, description="Gestion de risque (target, stop loss, take profit, R:R)")
    model_version: Optional[str] = Field(None, description="Version du modèle (v11_temporal, v6, etc.)")
    timestamp: str = Field(..., description="Timestamp ISO de la prédiction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "crypto": "bitcoin",
                "symbol": "BTCUSDT",
                "name": "Bitcoin",
                "signal": "BUY",
                "confidence": 0.65,
                "probabilities": {
                    "buy": 0.65,
                    "sell": 0.15,
                    "hold": 0.20
                },
                "current_price": 45000.50,
                "risk_management": {
                    "target_price": 49500.0,
                    "stop_loss": 44100.0,
                    "take_profit": 49500.0,
                    "risk_reward_ratio": 5.0,
                    "potential_gain_percent": 10.0,
                    "potential_loss_percent": 2.0
                },
                "timestamp": "2025-01-01T12:00:00"
            }
        }
    }


class AllPredictionsResponse(BaseModel):
    """Response model for all predictions"""
    predictions: Dict[str, PredictionResponse] = Field(..., description="Prédictions par crypto")
    timestamp: str = Field(..., description="Timestamp ISO")
    count: int = Field(..., description="Nombre de cryptos")


class CryptoInfo(BaseModel):
    """Crypto information"""
    id: str = Field(..., description="Crypto ID")
    symbol: str = Field(..., description="Trading symbol")
    name: str = Field(..., description="Nom complet")


class CryptoListResponse(BaseModel):
    """Response model for crypto list"""
    cryptos: Dict[str, CryptoInfo] = Field(..., description="Liste des cryptos supportées")
    count: int = Field(..., description="Nombre de cryptos")


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="État du service (healthy/unhealthy)")
    timestamp: str = Field(..., description="Timestamp ISO")
    models_loaded: int = Field(..., description="Nombre de modèles chargés")
    cryptos_available: List[str] = Field(..., description="Liste des cryptos disponibles")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Message d'erreur")
    detail: Optional[str] = Field(None, description="Détails supplémentaires")
    timestamp: str = Field(..., description="Timestamp ISO")
