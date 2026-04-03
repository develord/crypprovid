"""
CNN PREDICTION SERVICE
======================
Prediction service using CNN models (LONG + SHORT) from crypto_v10_multi_tf.

Features:
- 5 coins: BTC, ETH, SOL, DOGE, AVAX
- LONG + SHORT independent models per coin
- Real-time features from Binance API (4h, 1d, 1w)
- Dynamic TP/SL based on ATR
- Intelligent signal filters (bear market, momentum, volatility)
"""

import torch
import numpy as np
import pandas as pd
import json
import joblib
import ccxt
import ta
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from direction_prediction_model import CNNDirectionModel

logger = logging.getLogger(__name__)

COIN_CONFIG = {
    'bitcoin':    {'symbol': 'BTC/USDT', 'short_name': 'btc',  'long_conf': 0.60, 'short_conf': 0.58, 'start': '2017-01-01'},
    'ethereum':   {'symbol': 'ETH/USDT', 'short_name': 'eth',  'long_conf': 0.60, 'short_conf': 0.60, 'start': '2018-01-01'},
    'solana':     {'symbol': 'SOL/USDT', 'short_name': 'sol',  'long_conf': 0.65, 'short_conf': 0.55, 'start': '2020-08-01'},
    'dogecoin':   {'symbol': 'DOGE/USDT','short_name': 'doge', 'long_conf': 0.59, 'short_conf': 0.55, 'start': '2019-07-01'},
    'avalanche':  {'symbol': 'AVAX/USDT','short_name': 'avax', 'long_conf': 0.55, 'short_conf': 0.55, 'start': '2020-09-01'},
}

SEQ_LEN = 30


class CNNPredictionService:
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent / 'models' / 'cnn'
        self.long_models = {}
        self.short_models = {}
        self.long_scalers = {}
        self.short_scalers = {}
        self.long_features = {}
        self.short_features = {}
        self.exchange = ccxt.binance({'enableRateLimit': True})
        logger.info("[CNN] Prediction Service initialized")

    def _load_model(self, path):
        if not path.exists():
            return None
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = CNNDirectionModel(
            feature_dim=ckpt.get('feature_dim', 99),
            sequence_length=ckpt.get('sequence_length', 30),
            dropout=0.4
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model

    async def load_models(self):
        logger.info("[CNN] Loading models...")
        for crypto_id, cfg in COIN_CONFIG.items():
            sn = cfg['short_name']

            # LONG
            m = self._load_model(self.models_dir / f'{sn}_cnn_model.pt')
            if m:
                self.long_models[crypto_id] = m
                s = self.models_dir / f'{sn}_feature_scaler.joblib'
                if s.exists(): self.long_scalers[crypto_id] = joblib.load(s)
                f = self.models_dir / f'{sn}_features.json'
                if f.exists():
                    with open(f) as fh: self.long_features[crypto_id] = json.load(fh)
                logger.info(f"  [OK] {crypto_id} LONG loaded")

            # SHORT
            m = self._load_model(self.models_dir / f'{sn}_short_cnn_model.pt')
            if m:
                self.short_models[crypto_id] = m
                s = self.models_dir / f'{sn}_short_feature_scaler.joblib'
                if s.exists(): self.short_scalers[crypto_id] = joblib.load(s)
                f = self.models_dir / f'{sn}_short_features.json'
                if f.exists():
                    with open(f) as fh: self.short_features[crypto_id] = json.load(fh)
                logger.info(f"  [OK] {crypto_id} SHORT loaded")

        total = len(self.long_models) + len(self.short_models)
        logger.info(f"[CNN] Loaded {total} models ({len(self.long_models)} LONG + {len(self.short_models)} SHORT)")

    def get_live_price(self, crypto_id: str) -> Optional[float]:
        try:
            symbol = COIN_CONFIG[crypto_id]['symbol']
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Price error {crypto_id}: {e}")
            return None

    def _download_ohlcv(self, crypto_id: str, timeframe: str, limit: int = 300):
        symbol = COIN_CONFIG[crypto_id]['symbol']
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def _create_indicators(self, df, prefix=''):
        df[f'{prefix}rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df[f'{prefix}rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        macd = ta.trend.MACD(df['close'])
        df[f'{prefix}macd_line'] = macd.macd()
        df[f'{prefix}macd_signal'] = macd.macd_signal()
        df[f'{prefix}macd_histogram'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(df['close'])
        df[f'{prefix}bb_upper'] = bb.bollinger_hband()
        df[f'{prefix}bb_middle'] = bb.bollinger_mavg()
        df[f'{prefix}bb_lower'] = bb.bollinger_lband()
        df[f'{prefix}bb_width'] = bb.bollinger_wband()
        df[f'{prefix}ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df[f'{prefix}ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df[f'{prefix}ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df[f'{prefix}atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df[f'{prefix}stoch_k'] = stoch.stoch()
        df[f'{prefix}stoch_d'] = stoch.stoch_signal()
        df[f'{prefix}adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        df[f'{prefix}obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df[f'{prefix}cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
        df[f'{prefix}momentum_5'] = df['close'].pct_change(5)
        df[f'{prefix}momentum_10'] = df['close'].pct_change(10)
        df[f'{prefix}hist_vol_20'] = df['close'].pct_change().rolling(20).std()
        df[f'{prefix}volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        return df

    def _add_cross_tf_and_non_tech(self, df):
        # Cross-TF
        for tf1, tf2 in [('1d', '4h'), ('1d', '1w'), ('4h', '1w')]:
            c1, c2 = f'{tf1}_rsi_14', f'{tf2}_rsi_14'
            if c1 in df.columns and c2 in df.columns:
                df[f'rsi_diff_{tf1}_{tf2}'] = df[c1] - df[c2]
        rsi_cols = [c for c in df.columns if c.endswith('_rsi_14')]
        if len(rsi_cols) >= 2:
            df['rsi_bullish_count'] = sum((df[c] > 50).astype(int) for c in rsi_cols)
            df['rsi_oversold_count'] = sum((df[c] < 30).astype(int) for c in rsi_cols)
            df['rsi_overbought_count'] = sum((df[c] > 70).astype(int) for c in rsi_cols)
        macd_cols = [c for c in df.columns if c.endswith('_macd_histogram')]
        if len(macd_cols) >= 2:
            df['macd_bullish_count'] = sum((df[c] > 0).astype(int) for c in macd_cols)
        mom_cols = [c for c in df.columns if c.endswith('_momentum_5')]
        if len(mom_cols) >= 2:
            df['momentum_bullish_count'] = sum((df[c] > 0).astype(int) for c in mom_cols)
        adx_cols = [c for c in df.columns if c.endswith('_adx_14')]
        if len(adx_cols) >= 2:
            df['adx_strong_count'] = sum((df[c] > 25).astype(int) for c in adx_cols)
            df['adx_mean'] = sum(df[c] for c in adx_cols) / len(adx_cols)
        vol_cols = [c for c in df.columns if c.endswith('_hist_vol_20')]
        if len(vol_cols) >= 2:
            df['vol_mean_all_tf'] = sum(df[c] for c in vol_cols) / len(vol_cols)

        # Non-technical
        df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
        df['daily_range_ma5'] = df['daily_range_pct'].rolling(5).mean()
        df['daily_range_ma20'] = df['daily_range_pct'].rolling(20).mean()
        df['volatility_regime'] = (df['daily_range_ma5'] / df['daily_range_ma20']).fillna(1)
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_trend'] = (df['volume_ma5'] / df['volume_ma20']).fillna(1)
        df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
        df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-10)
        body = abs(df['close'] - df['open'])
        wick = df['high'] - df['low']
        df['body_ratio'] = body / (wick + 1e-10)
        df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (wick + 1e-10)
        df['lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (wick + 1e-10)
        df['returns'] = df['close'].pct_change()
        df['distance_from_sma20'] = (df['close'] / df['close'].rolling(20).mean() - 1)
        df['distance_from_sma50'] = (df['close'] / df['close'].rolling(50).mean() - 1)
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
        df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()
        df['trend_score'] = df['higher_highs'] - df['lower_lows']
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        return df

    def compute_live_features(self, crypto_id: str, feature_cols: list, scaler) -> Tuple[Optional[np.ndarray], Optional[pd.Series]]:
        try:
            df_1d = self._download_ohlcv(crypto_id, '1d', 300)
            df_1d = self._create_indicators(df_1d, '1d_')

            for tf in ['4h', '1w']:
                df_tf = self._download_ohlcv(crypto_id, tf, 300 if tf == '4h' else 100)
                df_tf = self._create_indicators(df_tf, f'{tf}_')
                tf_cols = ['date'] + [c for c in df_tf.columns if c.startswith(f'{tf}_')]
                df_1d = pd.merge_asof(df_1d.sort_values('date'), df_tf[tf_cols].sort_values('date'), on='date', direction='backward')

            df_1d = self._add_cross_tf_and_non_tech(df_1d)

            for c in feature_cols:
                if c not in df_1d.columns: df_1d[c] = 0
                df_1d[c] = pd.to_numeric(df_1d[c], errors='coerce')
            df_1d[feature_cols] = df_1d[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            raw_row = df_1d.iloc[-1]
            feat = df_1d[feature_cols].values.astype(np.float32)
            scaled = np.clip(np.nan_to_num(scaler.transform(feat), nan=0, posinf=0, neginf=0), -5, 5)

            if len(scaled) < SEQ_LEN:
                return None, None

            return scaled[-SEQ_LEN:], raw_row

        except Exception as e:
            logger.error(f"Feature error {crypto_id}: {e}")
            return None, None

    def _check_filters(self, raw_row, direction: str) -> Tuple[bool, str]:
        if direction == 'LONG':
            if 'distance_from_sma50' in raw_row.index and pd.notna(raw_row['distance_from_sma50']):
                if raw_row['distance_from_sma50'] < -0.05:
                    return False, "bear_market_sma50"
            if 'distance_from_sma20' in raw_row.index and pd.notna(raw_row['distance_from_sma20']):
                if raw_row['distance_from_sma20'] < -0.03:
                    return False, "bear_market_sma20"
            if 'trend_score' in raw_row.index and pd.notna(raw_row['trend_score']):
                if raw_row['trend_score'] < -3:
                    return False, "downtrend"
        else:
            if 'distance_from_sma50' in raw_row.index and pd.notna(raw_row['distance_from_sma50']):
                if raw_row['distance_from_sma50'] > 0.05:
                    return False, "bull_market"
            if 'trend_score' in raw_row.index and pd.notna(raw_row['trend_score']):
                if raw_row['trend_score'] > 3:
                    return False, "uptrend"
        return True, "pass"

    def _get_dynamic_tp_sl(self, raw_row, price: float, direction: str) -> Dict:
        atr = None
        if '1d_atr_14' in raw_row.index and pd.notna(raw_row['1d_atr_14']):
            atr = raw_row['1d_atr_14']

        if direction == 'LONG':
            if atr and atr > 0:
                tp_m = min(max(atr / price, 0.008), 0.03)
                sl_m = tp_m * 0.5
            else:
                tp_m, sl_m = 0.015, 0.0075
            return {
                'target_price': round(price * (1 + tp_m), 2),
                'stop_loss': round(price * (1 - sl_m), 2),
                'take_profit_pct': round(tp_m * 100, 2),
                'stop_loss_pct': round(sl_m * 100, 2),
                'risk_reward_ratio': round(tp_m / sl_m, 2),
            }
        else:
            if atr and atr > 0:
                tp_m = min(max(atr / price, 0.01), 0.04)
                sl_m = tp_m * 0.5
            else:
                tp_m, sl_m = 0.02, 0.01
            return {
                'target_price': round(price * (1 - tp_m), 2),
                'stop_loss': round(price * (1 + sl_m), 2),
                'take_profit_pct': round(tp_m * 100, 2),
                'stop_loss_pct': round(sl_m * 100, 2),
                'risk_reward_ratio': round(tp_m / sl_m, 2),
            }

    async def predict_one(self, crypto_id: str) -> Dict:
        if crypto_id not in COIN_CONFIG:
            raise ValueError(f"Unknown crypto: {crypto_id}")

        cfg = COIN_CONFIG[crypto_id]
        price = self.get_live_price(crypto_id)

        # LONG prediction
        long_signal, long_conf = None, 0
        long_filter_reason = None
        if crypto_id in self.long_models and crypto_id in self.long_features:
            scaled, raw_row = self.compute_live_features(crypto_id, self.long_features[crypto_id], self.long_scalers[crypto_id])
            if scaled is not None:
                X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    d, c = self.long_models[crypto_id].predict_direction(X)
                if d.item() == 1 and c.item() >= cfg['long_conf']:
                    passes, reason = self._check_filters(raw_row, 'LONG')
                    if passes:
                        long_signal = 'BUY'
                        long_conf = c.item()
                    else:
                        long_filter_reason = reason

        # SHORT prediction
        short_signal, short_conf = None, 0
        short_filter_reason = None
        if crypto_id in self.short_models and crypto_id in self.short_features:
            scaled_s, raw_row_s = self.compute_live_features(crypto_id, self.short_features[crypto_id], self.short_scalers[crypto_id])
            if scaled_s is not None:
                X_s = torch.tensor(scaled_s, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    d_s, c_s = self.short_models[crypto_id].predict_direction(X_s)
                if d_s.item() == 1 and c_s.item() >= cfg['short_conf']:
                    row = raw_row_s if raw_row is None else raw_row
                    passes, reason = self._check_filters(row, 'SHORT')
                    if passes:
                        short_signal = 'SELL'
                        short_conf = c_s.item()
                    else:
                        short_filter_reason = reason

        # Determine final signal (LONG priority)
        if long_signal:
            signal = 'BUY'
            confidence = long_conf
            direction = 'LONG'
        elif short_signal:
            signal = 'SELL'
            confidence = short_conf
            direction = 'SHORT'
        else:
            signal = 'HOLD'
            confidence = max(long_conf, short_conf) if long_conf or short_conf else 0
            direction = None

        # Risk management
        risk_management = None
        if direction and price and raw_row is not None:
            risk_management = self._get_dynamic_tp_sl(raw_row, price, direction)

        # Build response
        symbol_map = {'bitcoin': 'BTCUSDT', 'ethereum': 'ETHUSDT', 'solana': 'SOLUSDT',
                      'dogecoin': 'DOGEUSDT', 'avalanche': 'AVAXUSDT'}

        return {
            "crypto": crypto_id,
            "symbol": symbol_map.get(crypto_id, f'{crypto_id.upper()}USDT'),
            "name": crypto_id.capitalize(),
            "signal": signal,
            "direction": direction,
            "confidence": round(confidence, 4),
            "long_confidence": round(long_conf, 4) if long_conf else None,
            "short_confidence": round(short_conf, 4) if short_conf else None,
            "long_filter": long_filter_reason,
            "short_filter": short_filter_reason,
            "current_price": round(price, 4) if price else None,
            "risk_management": risk_management,
            "model": "CNN_1D_MultiScale",
            "features": "multi_tf_4h_1d_1w",
            "timestamp": datetime.now().isoformat(),
            "data_source": "binance_live"
        }

    # For compatibility with main.py
    @property
    def models(self):
        return {**{k: v for k, v in self.long_models.items()}, **{k: v for k, v in self.short_models.items()}}
