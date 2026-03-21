"""
FastAPI Server - Crypto Predictions API
========================================
Serveur API pour les prédictions de cryptomonnaies avec modèles V11 TEMPORAL
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))
sys.path.insert(0, str(project_root / 'data'))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from config import settings
from models import (
    PredictionResponse,
    AllPredictionsResponse,
    CryptoListResponse,
    HealthCheckResponse,
    ErrorResponse
)
from predictions_v11 import get_prediction_service_v11

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Predictions API",
    description="""
    API de prédictions pour cryptomonnaies utilisant des modèles V11 TEMPORAL.

    ## Features
    - Prédictions en temps réel (BUY/HOLD)
    - Modèles multi-timeframe (1d + 4h + 1h)
    - Triple Barrier: TP=+1.5%, SL=-0.75%
    - Thresholds optimaux (BTC=0.37, ETH=0.35, SOL=0.35)
    - Performance validée: +43.38% ROI portfolio
    - Documentation Swagger interactive

    ## Cryptos supportées
    - Bitcoin (BTC) - ROI +22.56%
    - Ethereum (ETH) - ROI +45.07%
    - Solana (SOL) - ROI +64.48%
    """,
    version="11.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Crypto Adviser",
        "url": "https://github.com/crypto-adviser",
    },
    license_info={
        "name": "MIT",
    }
)

# CORS configuration - Pour l'app Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = None


@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global prediction_service
    try:
        logger.info("Starting API server with V11 TEMPORAL models...")
        prediction_service = get_prediction_service_v11()
        await prediction_service.load_models()
        logger.info(f"✅ API ready - {len(prediction_service.models)} V11 models loaded")
        logger.info(f"   Thresholds: BTC={prediction_service.thresholds['bitcoin']}, ETH={prediction_service.thresholds['ethereum']}, SOL={prediction_service.thresholds['solana']}")
    except Exception as e:
        logger.error(f"Failed to load V11 models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server...")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get(
    "/",
    response_model=dict,
    summary="API Root",
    description="Point d'entrée de l'API avec informations de base"
)
async def root():
    """Welcome endpoint"""
    return {
        "message": "Crypto Predictions API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "cryptos": "/api/cryptos"
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Vérifier l'état du serveur et des modèles"
)
async def health_check():
    """Health check endpoint"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(prediction_service.models),
        "cryptos_available": list(prediction_service.models.keys())
    }


@app.get(
    "/api/cryptos",
    response_model=CryptoListResponse,
    summary="Liste des Cryptos",
    description="Obtenir la liste de toutes les cryptomonnaies supportées"
)
async def get_cryptos():
    """Get list of supported cryptocurrencies"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    # V11 supported cryptos with metadata
    cryptos_with_id = {
        "bitcoin": {
            "id": "bitcoin",
            "symbol": "BTCUSDT",
            "name": "Bitcoin"
        },
        "ethereum": {
            "id": "ethereum",
            "symbol": "ETHUSDT",
            "name": "Ethereum"
        },
        "solana": {
            "id": "solana",
            "symbol": "SOLUSDT",
            "name": "Solana"
        }
    }

    return {
        "cryptos": cryptos_with_id,
        "count": len(cryptos_with_id)
    }


@app.get(
    "/api/predictions/all",
    response_model=AllPredictionsResponse,
    summary="Toutes les Prédictions",
    description="Obtenir les prédictions pour toutes les cryptomonnaies en une seule requête"
)
async def get_all_predictions():
    """Get predictions for all cryptocurrencies"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    try:
        predictions = await prediction_service.predict_all()
        return {
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Error getting all predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate predictions: {str(e)}"
        )


@app.get(
    "/api/predictions/{crypto}",
    response_model=PredictionResponse,
    summary="Prédiction Crypto",
    description="Obtenir la prédiction pour une cryptomonnaie spécifique",
    responses={
        200: {
            "description": "Prédiction générée avec succès",
            "content": {
                "application/json": {
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
                        "timestamp": "2025-01-01T12:00:00"
                    }
                }
            }
        },
        404: {
            "description": "Crypto non trouvée",
            "model": ErrorResponse
        }
    }
)
async def get_prediction(crypto: str):
    """Get prediction for specific cryptocurrency"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    crypto = crypto.lower()
    if crypto not in prediction_service.models:
        available = list(prediction_service.models.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crypto '{crypto}' not found. Available: {available}"
        )

    try:
        prediction = await prediction_service.predict_one(crypto)
        return prediction
    except Exception as e:
        logger.error(f"Error predicting {crypto}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate prediction: {str(e)}"
        )


@app.get(
    "/api/price/{crypto}",
    summary="Prix Actuel",
    description="Obtenir le prix actuel d'une cryptomonnaie (inclus dans la prédiction)"
)
async def get_current_price(crypto: str):
    """Get current price for cryptocurrency (from latest CSV data)"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    crypto = crypto.lower()
    if crypto not in prediction_service.models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crypto '{crypto}' not found"
        )

    try:
        # Get price from latest features
        _, current_price = prediction_service.get_latest_features(crypto)

        symbols = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'solana': 'SOLUSDT'
        }

        return {
            "crypto": crypto,
            "symbol": symbols[crypto],
            "price": round(current_price, 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting price for {crypto}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get price: {str(e)}"
        )


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
