"""
LIVE FEATURE ENGINEERING FOR V11 PREDICTIONS
==============================================
Calcule les features multi-timeframe en temps réel depuis Binance API.

Architecture:
- Récupère les klines Binance (1d, 4h, 1h)
- Calcule les indicateurs techniques pour chaque timeframe
- Merge les features comme dans le CSV d'entraînement

Date: 21 Mars 2026
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple
from datetime import datetime


class LiveFeatureEngine:
    """Calcul des features en temps réel depuis Binance"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"

    def _get_binance_symbol(self, crypto_id: str) -> str:
        """Convert crypto ID to Binance symbol"""
        symbol_map = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'solana': 'SOLUSDT'
        }
        return symbol_map.get(crypto_id, f'{crypto_id.upper()}USDT')

    def fetch_klines(self, crypto_id: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Récupère les klines depuis Binance

        Args:
            crypto_id: 'bitcoin', 'ethereum', or 'solana'
            interval: '1d', '4h', '1h'
            limit: nombre de klines (default 500 pour calcul des indicateurs)

        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        try:
            symbol = self._get_binance_symbol(crypto_id)
            url = f"{self.base_url}/klines"

            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            klines = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Keep only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df.set_index('timestamp', inplace=True)

            print(f"[LiveFeatures] Fetched {len(df)} {interval} klines for {crypto_id}")
            return df

        except Exception as e:
            print(f"[LiveFeatures] Error fetching {interval} klines for {crypto_id}: {e}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Calcule les indicateurs techniques pour un timeframe

        Args:
            df: DataFrame avec OHLCV
            prefix: préfixe pour les noms de colonnes (ex: '1d_', '4h_', '1h_')

        Returns:
            DataFrame avec tous les indicateurs
        """
        result = df.copy()

        # Returns
        result[f'{prefix}return_1'] = result['close'].pct_change(1)
        result[f'{prefix}return_5'] = result['close'].pct_change(5)
        result[f'{prefix}return_10'] = result['close'].pct_change(10)
        result[f'{prefix}return_20'] = result['close'].pct_change(20)

        # Moving Averages
        for period in [7, 14, 21, 50, 100, 200]:
            result[f'{prefix}sma_{period}'] = result['close'].rolling(period).mean()
            result[f'{prefix}ema_{period}'] = result['close'].ewm(span=period).mean()

        # Volatility
        result[f'{prefix}volatility_10'] = result['close'].pct_change().rolling(10).std()
        result[f'{prefix}volatility_20'] = result['close'].pct_change().rolling(20).std()
        result[f'{prefix}volatility_50'] = result['close'].pct_change().rolling(50).std()

        # RSI
        for period in [7, 14, 21]:
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            result[f'{prefix}rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = result['close'].ewm(span=12).mean()
        ema26 = result['close'].ewm(span=26).mean()
        result[f'{prefix}macd'] = ema12 - ema26
        result[f'{prefix}macd_signal'] = result[f'{prefix}macd'].ewm(span=9).mean()
        result[f'{prefix}macd_diff'] = result[f'{prefix}macd'] - result[f'{prefix}macd_signal']

        # Bollinger Bands
        for period in [20, 50]:
            sma = result['close'].rolling(period).mean()
            std = result['close'].rolling(period).std()
            result[f'{prefix}bb_upper_{period}'] = sma + (2 * std)
            result[f'{prefix}bb_lower_{period}'] = sma - (2 * std)
            result[f'{prefix}bb_width_{period}'] = (result[f'{prefix}bb_upper_{period}'] - result[f'{prefix}bb_lower_{period}']) / sma

        # ATR (Average True Range)
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result[f'{prefix}atr_14'] = tr.rolling(14).mean()

        # Volume indicators
        result[f'{prefix}volume_sma_20'] = result['volume'].rolling(20).mean()
        result[f'{prefix}volume_ratio'] = result['volume'] / result[f'{prefix}volume_sma_20']

        # Momentum
        result[f'{prefix}momentum_10'] = result['close'] - result['close'].shift(10)
        result[f'{prefix}momentum_20'] = result['close'] - result['close'].shift(20)

        # Rate of Change
        result[f'{prefix}roc_10'] = ((result['close'] - result['close'].shift(10)) / result['close'].shift(10)) * 100
        result[f'{prefix}roc_20'] = ((result['close'] - result['close'].shift(20)) / result['close'].shift(20)) * 100

        # Stochastic Oscillator
        low_14 = result['low'].rolling(14).min()
        high_14 = result['high'].rolling(14).max()
        result[f'{prefix}stoch_k'] = 100 * ((result['close'] - low_14) / (high_14 - low_14))
        result[f'{prefix}stoch_d'] = result[f'{prefix}stoch_k'].rolling(3).mean()

        # Williams %R
        result[f'{prefix}williams_r'] = -100 * ((high_14 - result['close']) / (high_14 - low_14))

        # Price position relative to moving averages
        for period in [7, 14, 21, 50, 100, 200]:
            result[f'{prefix}price_to_sma{period}'] = (result['close'] - result[f'{prefix}sma_{period}']) / result[f'{prefix}sma_{period}']

        return result

    def get_live_features(self, crypto_id: str) -> Tuple[np.ndarray, float]:
        """
        Génère les features multi-timeframe en temps réel

        Args:
            crypto_id: 'bitcoin', 'ethereum', or 'solana'

        Returns:
            features: array numpy avec toutes les features
            current_price: prix actuel
        """
        print(f"\n[LiveFeatures] Generating live features for {crypto_id}...")

        # Fetch klines pour chaque timeframe
        df_1d = self.fetch_klines(crypto_id, '1d', limit=500)
        df_4h = self.fetch_klines(crypto_id, '4h', limit=500)
        df_1h = self.fetch_klines(crypto_id, '1h', limit=500)

        # Calculate indicators pour chaque timeframe
        df_1d = self.calculate_technical_indicators(df_1d, prefix='1d_')
        df_4h = self.calculate_technical_indicators(df_4h, prefix='4h_')
        df_1h = self.calculate_technical_indicators(df_1h, prefix='1h_')

        # Get latest row from each timeframe
        latest_1d = df_1d.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        latest_1h = df_1h.iloc[-1]

        # Merge features (enlever OHLCV de base, garder seulement les indicateurs)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']

        features_1d = latest_1d[[col for col in latest_1d.index if col not in exclude_cols]]
        features_4h = latest_4h[[col for col in latest_4h.index if col not in exclude_cols]]
        features_1h = latest_1h[[col for col in latest_1h.index if col not in exclude_cols]]

        # Combine all features
        all_features = pd.concat([features_1d, features_4h, features_1h])

        # Convert to numpy array et remplacer NaN par 0
        features_array = all_features.values.astype(float)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Current price from 1h timeframe (le plus récent)
        current_price = float(latest_1h['close'])

        print(f"[LiveFeatures] Generated {len(features_array)} features, price=${current_price:.2f}")

        return features_array, current_price
