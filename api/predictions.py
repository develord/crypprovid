"""
Prediction Service
==================
Service pour générer les prédictions avec les modèles XGBoost V6
"""
import sys
import os
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))
sys.path.insert(0, str(project_root / 'data'))

from config import settings
from models import PredictionResponse, Probabilities, RiskManagement
from train_models_xgboost import download_historical_data, calculate_indicators, prepare_features
from xgboost_features import calculate_all_xgboost_features
from xgboost_features_v6 import calculate_all_xgboost_features_v6
from feature_selection_v5 import SELECTED_FEATURES_V5

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for ML predictions"""

    CRYPTO_INFO = {
        'bitcoin': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'ethereum': {'symbol': 'ETHUSDT', 'name': 'Ethereum'},
        'bnb': {'symbol': 'BNBUSDT', 'name': 'BNB'},
        'xrp': {'symbol': 'XRPUSDT', 'name': 'XRP'},
        'cardano': {'symbol': 'ADAUSDT', 'name': 'Cardano'},
        'avalanche': {'symbol': 'AVAXUSDT', 'name': 'Avalanche'},
        'polkadot': {'symbol': 'DOTUSDT', 'name': 'Polkadot'},
        'solana': {'symbol': 'SOLUSDT', 'name': 'Solana'}
    }

    def __init__(self):
        self.models = {}
        self.indicators_history = {}

    async def load_models(self):
        """Load all XGBoost V6 models"""
        logger.info("Loading XGBoost V6 models...")

        for crypto_id in self.CRYPTO_INFO.keys():
            model_path = settings.MODELS_DIR / f"{crypto_id}_1d_xgboost_v6.pkl"

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            try:
                with open(model_path, 'rb') as f:
                    self.models[crypto_id] = pickle.load(f)
                logger.info(f"✅ Loaded {crypto_id} model")
            except Exception as e:
                logger.error(f"Failed to load {crypto_id} model: {e}")

        logger.info(f"Loaded {len(self.models)}/{len(self.CRYPTO_INFO)} models")

        if len(self.models) == 0:
            raise Exception("No models loaded!")

    def select_features_from_vector(self, features_69, selected_features_names):
        """Extract selected features from full 69-feature vector"""
        return features_69[:41]

    def prepare_features_v6(self, crypto_id: str, klines, klines_btc=None):
        """Prepare V6 features (41 V5 + 16 V6 = 57 features)"""
        if len(klines) < 201:
            raise ValueError("Need at least 201 candles")

        # Initialize indicators history if needed
        if crypto_id not in self.indicators_history:
            self.indicators_history[crypto_id] = []

        # Use last 201 candles
        i = len(klines) - 1
        window_data = klines[max(0, i-200):i+1]
        indicators = calculate_indicators(window_data)
        current_price = float(klines[i][4])

        # Update indicators history
        self.indicators_history[crypto_id].append(indicators)
        if len(self.indicators_history[crypto_id]) > 10:
            self.indicators_history[crypto_id].pop(0)

        prices_history = [float(k[4]) for k in window_data]
        features_base = prepare_features(indicators, current_price, prices_history)

        # BTC data
        window_data_btc = None
        if klines_btc and len(klines_btc) >= i+1:
            window_data_btc = klines_btc[max(0, i-200):i+1]

        volumes = [float(k[5]) for k in window_data]

        # V5 features (69)
        features_xgb_v5 = calculate_all_xgboost_features(
            window_data,
            indicators,
            volumes,
            crypto_symbol=self.CRYPTO_INFO[crypto_id]['symbol'],
            klines_btc=window_data_btc,
            indicators_history=self.indicators_history[crypto_id]
        )

        # V6 features (16)
        features_xgb_v6 = calculate_all_xgboost_features_v6(
            window_data,
            indicators,
            volumes,
            crypto_symbol=self.CRYPTO_INFO[crypto_id]['symbol'],
            klines_btc=window_data_btc,
            indicators_history=self.indicators_history[crypto_id]
        )

        # Combine: base + V5 + V6
        features_69_v5 = features_base + features_xgb_v5
        features_selected_v5 = self.select_features_from_vector(features_69_v5, SELECTED_FEATURES_V5)
        features_all_v6 = features_selected_v5 + features_xgb_v6

        return np.array(features_all_v6), current_price

    def calculate_risk_management(self, signal: str, current_price: float, confidence: float) -> RiskManagement:
        """
        Calculate risk management metrics based on signal and confidence

        Logic:
        - BUY: Target above current, Stop Loss below
        - SELL: Target below current, Stop Loss above
        - HOLD: No risk management (None)
        """
        if signal == "HOLD":
            return None

        # Parameters
        STOP_LOSS_PERCENT = 0.02  # 2% stop loss
        MAX_TARGET_PERCENT = 0.15  # Maximum 15% target

        if signal == "BUY":
            # Calculate target based on confidence (higher confidence = higher target)
            target_percent = confidence * MAX_TARGET_PERCENT
            target_price = current_price * (1 + target_percent)
            stop_loss = current_price * (1 - STOP_LOSS_PERCENT)
            take_profit = target_price

            # Calculate percentages
            potential_gain_percent = ((take_profit - current_price) / current_price) * 100
            potential_loss_percent = ((current_price - stop_loss) / current_price) * 100

            # Risk:Reward = Gain / Loss
            risk_reward_ratio = (take_profit - current_price) / (current_price - stop_loss)

        else:  # SELL
            # Calculate target based on confidence (higher confidence = lower target)
            target_percent = confidence * MAX_TARGET_PERCENT
            target_price = current_price * (1 - target_percent)
            stop_loss = current_price * (1 + STOP_LOSS_PERCENT)
            take_profit = target_price

            # Calculate percentages
            potential_gain_percent = ((current_price - take_profit) / current_price) * 100
            potential_loss_percent = ((stop_loss - current_price) / current_price) * 100

            # Risk:Reward = Gain / Loss
            risk_reward_ratio = (current_price - take_profit) / (stop_loss - current_price)

        return RiskManagement(
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            potential_gain_percent=round(potential_gain_percent, 2),
            potential_loss_percent=round(potential_loss_percent, 2)
        )

    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Binance"""
        # Use cached data for now
        klines = download_historical_data(symbol, interval='1d', limit=1)
        if klines and len(klines) > 0:
            return float(klines[-1][4])
        raise Exception(f"Failed to get price for {symbol}")

    async def predict_one(self, crypto_id: str) -> PredictionResponse:
        """Generate prediction for one crypto"""
        if crypto_id not in self.models:
            raise ValueError(f"Model not loaded for {crypto_id}")

        model = self.models[crypto_id]
        crypto_info = self.CRYPTO_INFO[crypto_id]
        symbol = crypto_info['symbol']

        # Download data
        klines = download_historical_data(symbol, interval='1d', limit=300)

        klines_btc = None
        if symbol != 'BTCUSDT':
            klines_btc = download_historical_data('BTCUSDT', interval='1d', limit=300)

        # Prepare features
        features, current_price = self.prepare_features_v6(crypto_id, klines, klines_btc)

        # Predict
        features_reshaped = features.reshape(1, -1)
        prediction = model.predict(features_reshaped)[0]
        probabilities = model.predict_proba(features_reshaped)[0]

        # Map prediction to signal
        signal_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        signal = signal_map[prediction]
        confidence = float(probabilities.max())

        # Calculate risk management
        risk_mgmt = self.calculate_risk_management(signal, current_price, confidence)
        logger.info(f"Risk management for {crypto_id}: {risk_mgmt}")

        return PredictionResponse(
            crypto=crypto_id,
            symbol=symbol,
            name=crypto_info['name'],
            signal=signal,
            confidence=confidence,
            probabilities=Probabilities(
                buy=float(probabilities[0]),
                sell=float(probabilities[1]),
                hold=float(probabilities[2])
            ),
            current_price=current_price,
            risk_management=risk_mgmt,
            timestamp=datetime.now().isoformat()
        )

    async def predict_all(self) -> Dict[str, PredictionResponse]:
        """Generate predictions for all cryptos"""
        predictions = {}

        for crypto_id in self.models.keys():
            try:
                prediction = await self.predict_one(crypto_id)
                predictions[crypto_id] = prediction
            except Exception as e:
                logger.error(f"Error predicting {crypto_id}: {e}")
                continue

        return predictions

