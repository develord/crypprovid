"""
COMPLETE LIVE FEATURE ENGINEERING FOR V11 PREDICTIONS
======================================================
Calcule TOUTES les 237 features multi-timeframe en temps réel depuis Binance API.

Architecture:
- Récupère les klines Binance (1d, 4h, 1w)
- Calcule 79 indicateurs techniques pour chaque timeframe
- Match exactement la structure du CSV d'entraînement

Date: 21 Mars 2026
"""

import pandas as pd
import numpy as np
import requests
from typing import Tuple
from datetime import datetime


class LiveFeatureEngineComplete:
    """Calcul complet des 237 features en temps réel depuis Binance"""

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
            interval: '1d', '4h', '1w'
            limit: nombre de klines (default 500)

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

    def calculate_complete_indicators(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Calcule TOUS les 79 indicateurs pour un timeframe (match exact CSV)

        Args:
            df: DataFrame avec OHLCV
            prefix: préfixe pour les noms de colonnes (ex: '1d_', '4h_', '1w_')

        Returns:
            DataFrame avec tous les 79 indicateurs
        """
        result = df.copy()

        # 1-2. RSI (14, 21)
        for period in [14, 21]:
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            result[f'{prefix}rsi_{period}'] = 100 - (100 / (1 + rs))

        # 3-4. RSI flags
        result[f'{prefix}rsi_overbought'] = (result[f'{prefix}rsi_14'] > 70).astype(int)
        result[f'{prefix}rsi_oversold'] = (result[f'{prefix}rsi_14'] < 30).astype(int)

        # 5-8. MACD
        ema12 = result['close'].ewm(span=12).mean()
        ema26 = result['close'].ewm(span=26).mean()
        result[f'{prefix}macd_line'] = ema12 - ema26
        result[f'{prefix}macd_signal'] = result[f'{prefix}macd_line'].ewm(span=9).mean()
        result[f'{prefix}macd_histogram'] = result[f'{prefix}macd_line'] - result[f'{prefix}macd_signal']
        result[f'{prefix}macd_crossover'] = ((result[f'{prefix}macd_line'] > result[f'{prefix}macd_signal']) &
                                              (result[f'{prefix}macd_line'].shift(1) <= result[f'{prefix}macd_signal'].shift(1))).astype(int)

        # 9-13. Bollinger Bands
        sma20 = result['close'].rolling(20).mean()
        std20 = result['close'].rolling(20).std()
        result[f'{prefix}bb_upper'] = sma20 + (2 * std20)
        result[f'{prefix}bb_middle'] = sma20
        result[f'{prefix}bb_lower'] = sma20 - (2 * std20)
        result[f'{prefix}bb_width'] = (result[f'{prefix}bb_upper'] - result[f'{prefix}bb_lower']) / result[f'{prefix}bb_middle']
        result[f'{prefix}bb_percent'] = (result['close'] - result[f'{prefix}bb_lower']) / (result[f'{prefix}bb_upper'] - result[f'{prefix}bb_lower'])

        # 14-18. EMAs
        for period in [12, 26, 50, 200]:
            result[f'{prefix}ema_{period}'] = result['close'].ewm(span=period).mean()
        result[f'{prefix}ema_cross_12_26'] = (result[f'{prefix}ema_12'] > result[f'{prefix}ema_26']).astype(int)

        # 19-21. SMAs
        for period in [20, 50, 200]:
            result[f'{prefix}sma_{period}'] = result['close'].rolling(period).mean()

        # 22-23. ATR
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result[f'{prefix}atr_14'] = tr.rolling(14).mean()
        result[f'{prefix}atr_pct'] = (result[f'{prefix}atr_14'] / result['close']) * 100

        # 24-27. Stochastic
        low_14 = result['low'].rolling(14).min()
        high_14 = result['high'].rolling(14).max()
        result[f'{prefix}stoch_k'] = 100 * ((result['close'] - low_14) / (high_14 - low_14 + 1e-10))
        result[f'{prefix}stoch_d'] = result[f'{prefix}stoch_k'].rolling(3).mean()
        result[f'{prefix}stoch_overbought'] = (result[f'{prefix}stoch_k'] > 80).astype(int)
        result[f'{prefix}stoch_oversold'] = (result[f'{prefix}stoch_k'] < 20).astype(int)

        # 28. ADX
        plus_dm = result['high'].diff()
        minus_dm = -result['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr_smooth = tr.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr_smooth)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        result[f'{prefix}adx_14'] = dx.rolling(14).mean()

        # 29. OBV (On-Balance Volume)
        obv = (np.sign(result['close'].diff()) * result['volume']).fillna(0).cumsum()
        result[f'{prefix}obv'] = obv

        # 30. CMF (Chaikin Money Flow)
        mfm = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (result['high'] - result['low'] + 1e-10)
        mfv = mfm * result['volume']
        result[f'{prefix}cmf_20'] = mfv.rolling(20).sum() / result['volume'].rolling(20).sum()

        # 31-40. Lags (close et volume, 1-5 periods)
        for lag in range(1, 6):
            result[f'{prefix}close_lag_{lag}'] = result['close'].shift(lag)
            result[f'{prefix}volume_lag_{lag}'] = result['volume'].shift(lag)

        # 41-52. Price vs Mean, Volatility, Trend (windows 3,5,7,14)
        for window in [3, 5, 7, 14]:
            mean = result['close'].rolling(window).mean()
            std = result['close'].rolling(window).std()
            result[f'{prefix}price_vs_mean_{window}'] = (result['close'] - mean) / (mean + 1e-10)
            result[f'{prefix}volatility_{window}'] = std / (mean + 1e-10)
            result[f'{prefix}trend_{window}'] = (result['close'] - result['close'].shift(window)) / (result['close'].shift(window) + 1e-10)

        # 53-57. Momentum (1,3,5,7,14 periods)
        for period in [1, 3, 5, 7, 14]:
            result[f'{prefix}momentum_{period}'] = result['close'] - result['close'].shift(period)

        # 58-61. Acceleration
        result[f'{prefix}accel_1_3'] = result[f'{prefix}momentum_3'] - result[f'{prefix}momentum_1']
        result[f'{prefix}accel_3_5'] = result[f'{prefix}momentum_5'] - result[f'{prefix}momentum_3']
        result[f'{prefix}accel_5_7'] = result[f'{prefix}momentum_7'] - result[f'{prefix}momentum_5']
        result[f'{prefix}accel_7_14'] = result[f'{prefix}momentum_14'] - result[f'{prefix}momentum_7']

        # 62-65. Higher highs / Lower lows ratios and range metrics
        result[f'{prefix}higher_highs_ratio'] = (result['high'] > result['high'].shift(1)).rolling(14).sum() / 14
        result[f'{prefix}lower_lows_ratio'] = (result['low'] < result['low'].shift(1)).rolling(14).sum() / 14
        high_14 = result['high'].rolling(14).max()
        low_14 = result['low'].rolling(14).min()
        result[f'{prefix}position_in_range'] = (result['close'] - low_14) / (high_14 - low_14 + 1e-10)
        result[f'{prefix}range_expansion'] = (high_14 - low_14) / (high_14.shift(14) - low_14.shift(14) + 1e-10)

        # 66-71. Volume ratios and trends (windows 3,7,14)
        for window in [3, 7, 14]:
            vol_mean = result['volume'].rolling(window).mean()
            result[f'{prefix}volume_ratio_{window}'] = result['volume'] / (vol_mean + 1e-10)
            result[f'{prefix}volume_trend_{window}'] = (result['volume'] - result['volume'].shift(window)) / (result['volume'].shift(window) + 1e-10)

        # 72-77. Indicator momentums
        result[f'{prefix}rsi_momentum'] = result[f'{prefix}rsi_14'] - result[f'{prefix}rsi_14'].shift(1)
        result[f'{prefix}macd_momentum'] = result[f'{prefix}macd_histogram'] - result[f'{prefix}macd_histogram'].shift(1)
        result[f'{prefix}stoch_momentum'] = result[f'{prefix}stoch_k'] - result[f'{prefix}stoch_k'].shift(1)
        result[f'{prefix}cmf_momentum'] = result[f'{prefix}cmf_20'] - result[f'{prefix}cmf_20'].shift(1)
        result[f'{prefix}bb_position_momentum'] = result[f'{prefix}bb_percent'] - result[f'{prefix}bb_percent'].shift(1)
        result[f'{prefix}adx_momentum'] = result[f'{prefix}adx_14'] - result[f'{prefix}adx_14'].shift(1)

        # 78-79. Price-Volume correlation and divergence
        result[f'{prefix}price_volume_corr'] = result['close'].rolling(14).corr(result['volume'])
        price_change = result['close'].pct_change()
        volume_change = result['volume'].pct_change()
        result[f'{prefix}price_volume_divergence'] = (np.sign(price_change) != np.sign(volume_change)).rolling(14).sum() / 14

        return result

    def calculate_altcoin_specific_features(self, df_asset: pd.DataFrame, df_btc: pd.DataFrame, prefix: str = "") -> pd.Series:
        """
        Calcule les 4 features spécifiques aux altcoins pour un timeframe
        - altcoin_season_7, altcoin_season_14: Binary (1 if altcoin outperforming BTC)
        - asset_price_vol_divergence: Binary (1 if price up but volume down or vice versa)
        - asset_trend: Float (trend strength)

        Args:
            df_asset: DataFrame OHLCV de l'altcoin
            df_btc: DataFrame OHLCV de Bitcoin
            prefix: Prefix pour les noms de colonnes (ex: '1d_')

        Returns:
            Series avec 4 features
        """
        features = {}

        # Align dataframes
        df_combined = pd.DataFrame({
            'asset_close': df_asset['close'],
            'asset_volume': df_asset['volume'],
            'btc_close': df_btc['close']
        })

        # 1-2. Altcoin season indicators (altcoin outperforming BTC)
        for period in [7, 14]:
            if len(df_combined) >= period:
                asset_return = df_combined['asset_close'].pct_change(period).iloc[-1]
                btc_return = df_combined['btc_close'].pct_change(period).iloc[-1]
                features[f'{prefix}altcoin_season_{period}'] = 1 if asset_return > btc_return else 0
            else:
                features[f'{prefix}altcoin_season_{period}'] = 0

        # 3. Asset price-volume divergence (price up + volume down = 1, or vice versa)
        if len(df_combined) >= 5:
            price_change_recent = df_combined['asset_close'].pct_change(5).iloc[-1]
            volume_change_recent = df_combined['asset_volume'].pct_change(5).iloc[-1]
            # Divergence if signs are opposite
            features[f'{prefix}asset_price_vol_divergence'] = 1 if (price_change_recent > 0 and volume_change_recent < 0) or (price_change_recent < 0 and volume_change_recent > 0) else 0
        else:
            features[f'{prefix}asset_price_vol_divergence'] = 0

        # 4. Asset trend (EMA ratio as trend strength)
        if len(df_combined) >= 50:
            ema_short = df_combined['asset_close'].ewm(span=12).mean().iloc[-1]
            ema_long = df_combined['asset_close'].ewm(span=50).mean().iloc[-1]
            features[f'{prefix}asset_trend'] = ema_short / (ema_long + 1e-10)
        else:
            features[f'{prefix}asset_trend'] = 1.0

        # Replace NaN with defaults
        for k, v in features.items():
            if pd.isna(v):
                features[k] = 0 if 'season' in k or 'divergence' in k else 1.0

        return pd.Series(features)

    def calculate_btc_correlation_features(self, df_asset: pd.DataFrame, df_btc: pd.DataFrame, prefix: str = "") -> pd.Series:
        """
        Calcule les 33 features de corrélation BTC pour un timeframe (pour altcoins seulement)

        Args:
            df_asset: DataFrame OHLCV de l'altcoin (ETH/SOL)
            df_btc: DataFrame OHLCV de Bitcoin
            prefix: préfixe pour les noms (ex: '1d_', '4h_', '1w_')

        Returns:
            Series avec 33 features BTC
        """
        features = {}

        # Align dataframes par timestamp
        df_combined = pd.DataFrame({
            'asset_close': df_asset['close'],
            'asset_volume': df_asset['volume'],
            'btc_close': df_btc['close'],
            'btc_volume': df_btc['volume']
        })

        # 1-7. Correlations (price + volume)
        for period in [7, 14, 30, 60]:
            if len(df_combined) >= period:
                features[f'{prefix}btc_corr_{period}'] = df_combined['asset_close'].rolling(period).corr(df_combined['btc_close']).iloc[-1]

        for period in [3, 7, 14]:
            if len(df_combined) >= period:
                features[f'{prefix}btc_volume_corr_{period}'] = df_combined['asset_volume'].rolling(period).corr(df_combined['btc_volume']).iloc[-1]

        # 8-13. Volatility ratios
        for period in [7, 14, 30]:
            asset_vol = df_combined['asset_close'].pct_change().rolling(period).std().iloc[-1]
            btc_vol = df_combined['btc_close'].pct_change().rolling(period).std().iloc[-1]
            features[f'{prefix}btc_volatility_{period}'] = btc_vol
            features[f'{prefix}btc_volatility_ratio_{period}'] = asset_vol / (btc_vol + 1e-10)

        # 14-19. Momentum ratios
        for period in [3, 7, 14]:
            asset_mom = df_combined['asset_close'].pct_change(period).iloc[-1]
            btc_mom = df_combined['btc_close'].pct_change(period).iloc[-1]
            features[f'{prefix}btc_momentum_{period}'] = btc_mom
            features[f'{prefix}btc_momentum_ratio_{period}'] = asset_mom / (btc_mom + 1e-10)

        # 20-22. Trend features
        asset_trend = (df_combined['asset_close'].iloc[-1] - df_combined['asset_close'].iloc[-14]) / (df_combined['asset_close'].iloc[-14] + 1e-10) if len(df_combined) >= 14 else 0
        btc_trend = (df_combined['btc_close'].iloc[-1] - df_combined['btc_close'].iloc[-14]) / (df_combined['btc_close'].iloc[-14] + 1e-10) if len(df_combined) >= 14 else 0
        features[f'{prefix}btc_trend'] = btc_trend
        features[f'{prefix}btc_trend_alignment'] = 1 if (asset_trend > 0 and btc_trend > 0) or (asset_trend < 0 and btc_trend < 0) else 0
        features[f'{prefix}btc_trend_diff'] = asset_trend - btc_trend

        # 23-25. Relative strength
        for period in [7, 14, 30]:
            if len(df_combined) >= period:
                asset_change = df_combined['asset_close'].pct_change(period).iloc[-1]
                btc_change = df_combined['btc_close'].pct_change(period).iloc[-1]
                features[f'{prefix}btc_relative_strength_{period}'] = asset_change - btc_change

        # 26-28. Volume ratios
        for period in [3, 7, 14]:
            if len(df_combined) >= period:
                asset_vol_avg = df_combined['asset_volume'].rolling(period).mean().iloc[-1]
                btc_vol_avg = df_combined['btc_volume'].rolling(period).mean().iloc[-1]
                features[f'{prefix}btc_volume_ratio_{period}'] = asset_vol_avg / (btc_vol_avg + 1e-10)

        # 29-30. Divergence
        asset_price_change = df_combined['asset_close'].pct_change().iloc[-14:] if len(df_combined) >= 14 else df_combined['asset_close'].pct_change()
        btc_price_change = df_combined['btc_close'].pct_change().iloc[-14:] if len(df_combined) >= 14 else df_combined['btc_close'].pct_change()
        asset_vol_change = df_combined['asset_volume'].pct_change().iloc[-14:] if len(df_combined) >= 14 else df_combined['asset_volume'].pct_change()

        features[f'{prefix}btc_price_vol_divergence'] = (np.sign(btc_price_change) != np.sign(df_combined['btc_volume'].pct_change().iloc[-14:] if len(df_combined) >= 14 else df_combined['btc_volume'].pct_change())).sum() / len(btc_price_change)
        features[f'{prefix}btc_asset_divergence'] = (np.sign(asset_price_change) != np.sign(btc_price_change)).sum() / len(asset_price_change)

        # 31-32. Dominance (BTC market strength)
        for period in [7, 14]:
            if len(df_combined) >= period:
                btc_strength = (df_combined['btc_close'].iloc[-1] > df_combined['btc_close'].rolling(period).mean().iloc[-1]).astype(int)
                features[f'{prefix}btc_dominance_{period}'] = btc_strength

        # 33. Synchronization
        features[f'{prefix}btc_asset_synchronized'] = features.get(f'{prefix}btc_trend_alignment', 0)

        return pd.Series(features)

    def get_live_features(self, crypto_id: str) -> Tuple[np.ndarray, float]:
        """
        Génère les features multi-timeframe en temps réel
        - Bitcoin: 237 features (79 × 3 timeframes)
        - Ethereum/Solana: 348 features (237 + 111 BTC correlation)

        Args:
            crypto_id: 'bitcoin', 'ethereum', or 'solana'

        Returns:
            features: array numpy avec les features (237 ou 348)
            current_price: prix actuel
        """
        is_altcoin = crypto_id in ['ethereum', 'solana']
        expected_features = 348 if is_altcoin else 237

        print(f"\n[LiveFeatures] Generating COMPLETE {expected_features} features for {crypto_id}...")

        # Fetch klines pour chaque timeframe
        df_1d = self.fetch_klines(crypto_id, '1d', limit=500)
        df_4h = self.fetch_klines(crypto_id, '4h', limit=500)
        df_1w = self.fetch_klines(crypto_id, '1w', limit=500)

        # Calculate indicators pour chaque timeframe (79 features each)
        df_1d = self.calculate_complete_indicators(df_1d, prefix='1d_')
        df_4h = self.calculate_complete_indicators(df_4h, prefix='4h_')
        df_1w = self.calculate_complete_indicators(df_1w, prefix='1w_')

        # Get latest row from each timeframe
        latest_1d = df_1d.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        latest_1w = df_1w.iloc[-1]

        # Exclude OHLCV columns, keep only the 79 indicators
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']

        features_1d = latest_1d[[col for col in latest_1d.index if col not in exclude_cols]]
        features_4h = latest_4h[[col for col in latest_4h.index if col not in exclude_cols]]
        features_1w = latest_1w[[col for col in latest_1w.index if col not in exclude_cols]]

        # Combine: 1d (79) + 4h (79) + 1w (79) = 237 features
        all_features = pd.concat([features_1d, features_4h, features_1w])

        # Add altcoin-specific features for ETH/SOL (12 features: 4 per timeframe)
        if is_altcoin:
            print(f"[LiveFeatures] Fetching BTC data for {crypto_id} altcoin features...")

            # Fetch Bitcoin klines for all timeframes
            btc_1d = self.fetch_klines('bitcoin', '1d', limit=500)
            btc_4h = self.fetch_klines('bitcoin', '4h', limit=500)
            btc_1w = self.fetch_klines('bitcoin', '1w', limit=500)

            # Calculate altcoin-specific features (4 per timeframe × 3 = 12 features)
            altcoin_features_1d = self.calculate_altcoin_specific_features(df_1d, btc_1d, prefix='1d_')
            altcoin_features_4h = self.calculate_altcoin_specific_features(df_4h, btc_4h, prefix='4h_')
            altcoin_features_1w = self.calculate_altcoin_specific_features(df_1w, btc_1w, prefix='1w_')

            # Calculate BTC correlation features (33 per timeframe × 3 = 99 features)
            btc_features_1d = self.calculate_btc_correlation_features(df_1d, btc_1d, prefix='1d_')
            btc_features_4h = self.calculate_btc_correlation_features(df_4h, btc_4h, prefix='4h_')
            btc_features_1w = self.calculate_btc_correlation_features(df_1w, btc_1w, prefix='1w_')

            # Combine all altcoin features (12 + 99 = 111 features)
            altcoin_all = pd.concat([
                altcoin_features_1d, altcoin_features_4h, altcoin_features_1w,
                btc_features_1d, btc_features_4h, btc_features_1w
            ])

            # Append to base features: 237 + 111 = 348 features
            all_features = pd.concat([all_features, altcoin_all])

            print(f"[LiveFeatures] Added {len(altcoin_all)} altcoin-specific features (12 + 99 BTC correlation)")

        # Convert to numpy array et remplacer NaN par 0
        features_array = all_features.values.astype(float)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Current price from 4h timeframe (le plus récent et fiable)
        current_price = float(latest_4h['close'])

        print(f"[LiveFeatures] Generated {len(features_array)} features, price=${current_price:.2f}")

        return features_array, current_price
