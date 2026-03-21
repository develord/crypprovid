"""
V11 TEMPORAL PREDICTION SERVICE
================================
Service de prédiction utilisant les modèles V11 TEMPORAL multi-timeframe.

Architecture:
- Modèles: Binary classifiers XGBoost (P(TP) prediction)
- Features: Multi-timeframe (1d + 4h + 1h) déjà calculées dans CSV merged
- Thresholds optimaux: BTC=0.37, ETH=0.35, SOL=0.35
- Triple Barrier: TP=+1.5%, SL=-0.75%, 7 days lookahead

Performance validée (Walk-Forward):
- Portfolio ROI: +43.38%
- BTC: +22.56%, ETH: +45.07%, SOL: +64.48%

Date: 21 Mars 2026
"""

import pandas as pd
import numpy as np
import joblib
import json
import requests
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

class PredictionServiceV11:
    """Service de prédiction V11 TEMPORAL"""

    def __init__(self):
        """Initialize V11 prediction service"""
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / 'models' / 'v11'
        self.data_dir = self.base_dir / 'data' / 'v11_cache'

        # Load config
        self.config = self._load_config()
        self.thresholds = self._load_thresholds()

        # Models storage
        self.models = {}

        print("[V11] Prediction Service initialized")

    def _load_config(self) -> Dict:
        """Load V11 configuration"""
        config_file = self.data_dir / 'v11_config.json'

        if not config_file.exists():
            raise FileNotFoundError(f"V11 config not found: {config_file}")

        with open(config_file) as f:
            config = json.load(f)

        print(f"[V11] Loaded config: {config['version']}")
        return config

    def _load_thresholds(self) -> Dict[str, float]:
        """Load optimal thresholds"""
        threshold_file = self.data_dir / 'optimal_thresholds_v11.json'

        if not threshold_file.exists():
            raise FileNotFoundError(f"Thresholds not found: {threshold_file}")

        with open(threshold_file) as f:
            thresholds = json.load(f)

        print(f"[V11] Loaded thresholds: {thresholds}")
        return thresholds

    def _get_binance_symbol(self, crypto_id: str) -> str:
        """Convert crypto ID to Binance symbol"""
        symbol_map = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'solana': 'SOLUSDT'
        }
        return symbol_map.get(crypto_id, f'{crypto_id.upper()}USDT')

    def get_live_price(self, crypto_id: str) -> Optional[float]:
        """
        Get live price from Binance API

        Args:
            crypto_id: 'bitcoin', 'ethereum', or 'solana'

        Returns:
            Live price from Binance, or None if error
        """
        try:
            binance_symbol = self._get_binance_symbol(crypto_id)
            url = f'https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}'
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                live_price = float(data['price'])
                print(f"[V11] Live price for {crypto_id}: ${live_price}")
                return live_price
            else:
                print(f"[V11] Binance API error for {crypto_id}: {response.status_code}")
                return None
        except Exception as e:
            print(f"[V11] Failed to fetch live price for {crypto_id}: {e}")
            return None

    async def load_models(self):
        """Load all V11 models"""
        print("\n[V11] Loading models...")

        for crypto_id, crypto_config in self.config['cryptos'].items():
            model_file = self.models_dir / crypto_config['model']

            if not model_file.exists():
                print(f"  [!!] Model not found: {model_file}")
                continue

            try:
                self.models[crypto_id] = joblib.load(model_file)
                print(f"  [OK] Loaded {crypto_id}: {crypto_config['model']}")
            except Exception as e:
                print(f"  [ERROR] Failed to load {crypto_id}: {e}")

        print(f"\n[V11] Loaded {len(self.models)}/3 models")

    def get_latest_features(self, crypto_id: str) -> Tuple[np.ndarray, float]:
        """
        Get latest features from merged CSV (dernière ligne = données les plus récentes)

        Args:
            crypto_id: 'bitcoin', 'ethereum', or 'solana'

        Returns:
            features: Feature vector (numpy array)
            current_price: Current close price
        """
        # Load merged data
        data_file = self.data_dir / f'{crypto_id}_multi_tf_merged.csv'

        if not data_file.exists():
            raise FileNotFoundError(f"Data not found: {data_file}")

        df = pd.read_csv(data_file, index_col=0, parse_dates=True)

        # Get last row (most recent data)
        latest = df.iloc[-1]

        # Exclude non-feature columns
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric',
            'price_target_pct', 'future_price',
            'triple_barrier_label'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Extract features
        features = latest[feature_cols].values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle feature selection for Solana (Top 50)
        if crypto_id == 'solana' and self.config['cryptos'][crypto_id].get('feature_selection'):
            # Load selected features
            feature_file = self.data_dir / f'{crypto_id}_selected_features_top50.json'

            if feature_file.exists():
                with open(feature_file) as f:
                    feature_data = json.load(f)
                    selected_features = feature_data['selected_feature_names']

                # Get indices of selected features
                feature_indices = [feature_cols.index(f) for f in selected_features if f in feature_cols]
                features = features[feature_indices]

                print(f"[V11] Applied feature selection for {crypto_id}: {len(features)} features")

        # Current price
        current_price = latest['close']

        return features, current_price

    async def predict_one(self, crypto_id: str) -> Dict:
        """
        Make prediction for one crypto

        Args:
            crypto_id: 'bitcoin', 'ethereum', or 'solana'

        Returns:
            Prediction dictionary with signal, confidence, etc.
        """
        if crypto_id not in self.models:
            raise ValueError(f"Model not loaded for {crypto_id}")

        # Get model and threshold
        model = self.models[crypto_id]
        threshold = self.thresholds.get(crypto_id, 0.35)

        # Get latest features
        features, csv_price = self.get_latest_features(crypto_id)

        # Get live price from Binance (fallback to CSV price if unavailable)
        live_price = self.get_live_price(crypto_id)
        current_price = live_price if live_price is not None else csv_price

        # Predict P(TP)
        prob_tp = model.predict_proba(features.reshape(1, -1))[0, 1]

        # Apply threshold
        signal = "BUY" if prob_tp >= threshold else "HOLD"

        # Calculate risk management (if BUY)
        if signal == "BUY":
            tp_pct = 1.5  # Take Profit +1.5%
            sl_pct = 0.75  # Stop Loss -0.75%

            target_price = current_price * (1 + tp_pct / 100)
            stop_loss = current_price * (1 - sl_pct / 100)
            risk_reward_ratio = tp_pct / sl_pct

            risk_management = {
                "target_price": round(target_price, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(target_price, 2),
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "potential_gain_percent": tp_pct,
                "potential_loss_percent": sl_pct
            }
        else:
            risk_management = None

        # Build response
        result = {
            "crypto": crypto_id,
            "symbol": self._get_symbol(crypto_id),
            "name": crypto_id.capitalize(),
            "signal": signal,
            "confidence": round(prob_tp, 4),
            "threshold": threshold,
            "current_price": round(current_price, 2),
            "risk_management": risk_management,
            "model_version": "v11_temporal",
            "timestamp": datetime.now().isoformat()
        }

        return result

    async def predict_all(self) -> Dict:
        """
        Get predictions for all cryptos

        Returns:
            Dictionary with predictions for BTC, ETH, SOL
        """
        results = {}

        for crypto_id in ['bitcoin', 'ethereum', 'solana']:
            try:
                prediction = await self.predict_one(crypto_id)
                results[crypto_id] = prediction
            except Exception as e:
                print(f"[ERROR] Failed to predict {crypto_id}: {e}")
                results[crypto_id] = {
                    "error": str(e),
                    "crypto": crypto_id
                }

        return results

    def _get_symbol(self, crypto_id: str) -> str:
        """Get trading symbol"""
        symbols = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'solana': 'SOLUSDT'
        }
        return symbols.get(crypto_id, 'UNKNOWN')


# Global service instance
_service_v11 = None

def get_prediction_service_v11() -> PredictionServiceV11:
    """Get or create V11 prediction service instance"""
    global _service_v11
    if _service_v11 is None:
        _service_v11 = PredictionServiceV11()
    return _service_v11
