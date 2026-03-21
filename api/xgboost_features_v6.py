"""
XGBoost Features V6 - ENRICHED WITH ANTI-OVERFITTING FEATURES
============================================================

NOUVELLES FEATURES V6 (vs V5):
1. Price Lags (3 features): lag_1, lag_5, lag_7
2. Returns Explicites (4 features): return_1d, return_5d, return_7d, return_14d
3. Rolling Volatility (3 features): std_5d, std_10d, std_20d
4. Market Regime (3 features): is_bear_market, trend_strength_20d, sma_ratio
5. Downside Features (3 features): max_drawdown_20d, negative_momentum, downside_volatility

TOTAL V6: 69 (base) + 16 (nouvelles) = 85 features
Feature selection réduira à ~50 features optimales
"""

import numpy as np
from typing import List, Dict, Optional

# No imports needed - V6 features are independent


# ============================================================================
# NOUVELLES FEATURES V6 - PRICE LAGS
# ============================================================================

def calculate_price_lags(klines: List, current_price: float) -> Dict[str, float]:
    """
    Price lags directs - Donne contexte prix bruts au modèle

    Args:
        klines: Liste de klines
        current_price: Prix actuel

    Returns:
        Dict avec 3 features de price lags
    """
    closes = [float(k[4]) for k in klines[-15:]]  # Last 15 days

    if len(closes) < 15:
        return {
            'price_lag_1': 0,
            'price_lag_5': 0,
            'price_lag_7': 0
        }

    # Normalize par prix actuel (ratios relatifs)
    lag_1 = (current_price - closes[-2]) / current_price
    lag_5 = (current_price - closes[-6]) / current_price if len(closes) >= 6 else 0
    lag_7 = (current_price - closes[-8]) / current_price if len(closes) >= 8 else 0

    return {
        'price_lag_1': np.tanh(lag_1 * 10),  # Tanh pour limiter [-1, 1]
        'price_lag_5': np.tanh(lag_5 * 5),
        'price_lag_7': np.tanh(lag_7 * 3)
    }


# ============================================================================
# NOUVELLES FEATURES V6 - RETURNS EXPLICITES
# ============================================================================

def calculate_explicit_returns(klines: List) -> Dict[str, float]:
    """
    Returns explicites sur différentes fenêtres
    Capture momentum multi-timeframes

    Returns:
        Dict avec 4 features de returns
    """
    closes = [float(k[4]) for k in klines[-20:]]

    if len(closes) < 20:
        return {
            'return_1d': 0,
            'return_5d': 0,
            'return_7d': 0,
            'return_14d': 0
        }

    current = closes[-1]

    # Returns en pourcentage normalisés
    return_1d = (current - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
    return_5d = (current - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
    return_7d = (current - closes[-8]) / closes[-8] if len(closes) >= 8 else 0
    return_14d = (current - closes[-15]) / closes[-15] if len(closes) >= 15 else 0

    # Tanh pour limiter outliers
    return {
        'return_1d': np.tanh(return_1d * 20),  # 1d = plus sensible
        'return_5d': np.tanh(return_5d * 10),
        'return_7d': np.tanh(return_7d * 7),
        'return_14d': np.tanh(return_14d * 5)
    }


# ============================================================================
# NOUVELLES FEATURES V6 - ROLLING VOLATILITY
# ============================================================================

def calculate_rolling_volatility(klines: List) -> Dict[str, float]:
    """
    Rolling volatility sur différentes fenêtres
    Capture changements de régime de volatilité

    Returns:
        Dict avec 3 features de volatility
    """
    closes = [float(k[4]) for k in klines[-25:]]

    if len(closes) < 25:
        return {
            'std_5d': 0,
            'std_10d': 0,
            'std_20d': 0
        }

    # Calcul std normalisé par moyenne
    std_5d = np.std(closes[-5:]) / (np.mean(closes[-5:]) + 1e-10) if len(closes) >= 5 else 0
    std_10d = np.std(closes[-10:]) / (np.mean(closes[-10:]) + 1e-10) if len(closes) >= 10 else 0
    std_20d = np.std(closes[-20:]) / (np.mean(closes[-20:]) + 1e-10) if len(closes) >= 20 else 0

    return {
        'std_5d': np.tanh(std_5d * 20),
        'std_10d': np.tanh(std_10d * 15),
        'std_20d': np.tanh(std_20d * 10)
    }


# ============================================================================
# NOUVELLES FEATURES V6 - MARKET REGIME DETECTION
# ============================================================================

def calculate_market_regime(klines: List) -> Dict[str, float]:
    """
    Market regime detection - CRITIQUE pour bear markets
    Détecte si on est en bull/bear/sideways

    Returns:
        Dict avec 3 features de regime
    """
    closes = [float(k[4]) for k in klines[-50:]]

    if len(closes) < 50:
        return {
            'is_bear_market': 0.5,
            'trend_strength_20d': 0,
            'sma_ratio': 1.0
        }

    current = closes[-1]

    # 1. Is Bear Market (SMA50 < SMA20)
    sma_20 = np.mean(closes[-20:])
    sma_50 = np.mean(closes[-50:])

    is_bear = 1.0 if current < sma_20 < sma_50 else 0.0

    # 2. Trend Strength (slope normalized by volatility)
    slope_20d = (closes[-1] - closes[-20]) / 20
    vol_20d = np.std(closes[-20:])
    trend_strength = slope_20d / (vol_20d + 1e-10)

    # 3. SMA Ratio (current price vs SMA50)
    sma_ratio = current / (sma_50 + 1e-10)

    return {
        'is_bear_market': is_bear,
        'trend_strength_20d': np.tanh(trend_strength),
        'sma_ratio': np.tanh((sma_ratio - 1.0) * 10)  # Center around 1.0
    }


# ============================================================================
# NOUVELLES FEATURES V6 - DOWNSIDE FEATURES (BEAR MARKET PROTECTION)
# ============================================================================

def calculate_downside_features(klines: List) -> Dict[str, float]:
    """
    Downside features - Protection pour bear markets
    Focus sur drawdowns et momentum négatif

    Returns:
        Dict avec 3 features downside
    """
    closes = [float(k[4]) for k in klines[-30:]]

    if len(closes) < 30:
        return {
            'max_drawdown_20d': 0,
            'negative_momentum': 0,
            'downside_volatility': 0
        }

    # 1. Max Drawdown 20d
    peak_20d = np.max(closes[-20:])
    current = closes[-1]
    max_dd = (peak_20d - current) / peak_20d if peak_20d > 0 else 0

    # 2. Negative Momentum (moyenne des returns négatifs seulement)
    returns = np.diff(closes[-20:]) / closes[-20:-1]
    negative_returns = returns[returns < 0]
    neg_momentum = np.mean(negative_returns) if len(negative_returns) > 0 else 0

    # 3. Downside Volatility (std des returns négatifs)
    downside_vol = np.std(negative_returns) if len(negative_returns) > 1 else 0

    return {
        'max_drawdown_20d': np.tanh(max_dd * 10),
        'negative_momentum': np.tanh(neg_momentum * 20),
        'downside_volatility': np.tanh(downside_vol * 15)
    }


# ============================================================================
# MAIN FUNCTION - CALCULATE ALL V6 FEATURES
# ============================================================================

def calculate_all_xgboost_features_v6(
    klines: List,
    indicators: Dict,
    volumes: List,
    crypto_symbol: str = 'BTCUSDT',
    klines_btc: Optional[List] = None,
    indicators_history: List[Dict] = None
) -> List[float]:
    """
    Calculate ONLY the 16 new V6 features
    These will be combined with V5 features in the training script

    Returns:
        Liste de 16 features V6 (nouvelles features anti-overfitting)
    """

    current_price = float(klines[-1][4])

    # ========== NOUVELLES V6 FEATURES (16 features) ==========

    # Price lags (3 features)
    price_lags = calculate_price_lags(klines, current_price)
    features_v6 = [
        price_lags['price_lag_1'],
        price_lags['price_lag_5'],
        price_lags['price_lag_7']
    ]

    # Returns explicites (4 features)
    returns = calculate_explicit_returns(klines)
    features_v6.extend([
        returns['return_1d'],
        returns['return_5d'],
        returns['return_7d'],
        returns['return_14d']
    ])

    # Rolling volatility (3 features)
    roll_vol = calculate_rolling_volatility(klines)
    features_v6.extend([
        roll_vol['std_5d'],
        roll_vol['std_10d'],
        roll_vol['std_20d']
    ])

    # Market regime (3 features)
    regime_v6 = calculate_market_regime(klines)
    features_v6.extend([
        regime_v6['is_bear_market'],
        regime_v6['trend_strength_20d'],
        regime_v6['sma_ratio']
    ])

    # Downside features (3 features)
    downside = calculate_downside_features(klines)
    features_v6.extend([
        downside['max_drawdown_20d'],
        downside['negative_momentum'],
        downside['downside_volatility']
    ])

    return features_v6


# ============================================================================
# NOMS DES FEATURES V6
# ============================================================================

FEATURE_NAMES_V6_NEW = [
    # Price lags (3)
    'price_lag_1', 'price_lag_5', 'price_lag_7',
    # Returns (4)
    'return_1d', 'return_5d', 'return_7d', 'return_14d',
    # Rolling volatility (3)
    'std_5d', 'std_10d', 'std_20d',
    # Market regime (3)
    'is_bear_market', 'trend_strength_20d', 'sma_ratio',
    # Downside features (3)
    'max_drawdown_20d', 'negative_momentum', 'downside_volatility'
]

# Total: 16 nouvelles features V6
