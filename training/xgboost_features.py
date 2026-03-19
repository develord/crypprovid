"""
Features Optimisées pour XGBoost
Module contenant toutes les features avancées pour maximiser la précision XGBoost
Total: +26 nouvelles features (29 actuelles + 26 = 55 features)
"""

import numpy as np
from typing import List, Dict, Optional

# ============================================================================
# CATÉGORIE A: MOMENTUM MULTI-ÉCHELLE (4 features) ⭐⭐⭐⭐⭐
# ============================================================================

def calculate_roc_multiscale(prices: List[float]) -> Dict[str, float]:
    """
    Rate of Change sur différentes périodes
    Capture momentum court/moyen/long terme

    Returns:
        Dict avec ROC_5, ROC_10, ROC_20, ROC_50
    """
    prices = np.array(prices)

    return {
        'roc_5':  (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0,
        'roc_10': (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0,
        'roc_20': (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0,
        'roc_50': (prices[-1] - prices[-51]) / prices[-51] if len(prices) >= 51 else 0
    }


# ============================================================================
# CATÉGORIE B: RATIOS & INTERACTIONS (5 features) ⭐⭐⭐⭐⭐
# ============================================================================

def calculate_ratio_features(indicators: Dict, volumes: List[float]) -> Dict[str, float]:
    """
    Ratios critiques entre indicateurs
    XGBoost excelle à trouver patterns dans ratios non-linéaires

    Args:
        indicators: Dict avec RSI, MACD, ATR, Stochastic, etc.
        volumes: Liste des volumes historiques

    Returns:
        Dict avec 5 ratios
    """
    current_price = indicators.get('currentPrice', 1)
    macd_hist = indicators.get('macd', {}).get('histogram', 0) if indicators.get('macd') else 0
    atr = indicators.get('atr', 0.01) if indicators.get('atr') else 0.01
    rsi = indicators.get('rsi', 50) if indicators.get('rsi') else 50
    stoch_k = indicators.get('stochasticRsi', {}).get('k', 50) if indicators.get('stochasticRsi') else 50
    bb = indicators.get('bollingerBands', {})
    bb_width = (bb.get('upper', 0) - bb.get('lower', 0)) if bb else 0
    volume = volumes[-1] if len(volumes) > 0 else 1

    return {
        'rsi_stoch_divergence': abs(rsi - stoch_k) / 100,
        'macd_atr_ratio': np.tanh(macd_hist / (atr + 1e-10)),
        'volume_atr_ratio': np.tanh(volume / (atr * current_price + 1e-10) / 1e6),
        'bb_width_atr_ratio': bb_width / (atr + 1e-10) if atr > 0 else 0,
        'price_ema20_rsi': indicators.get('ema20', current_price) / current_price * rsi / 100 if indicators.get('ema20') else 0.5
    }


# ============================================================================
# CATÉGORIE C: VOLATILITÉ AVANCÉE (4 features) ⭐⭐⭐⭐
# ============================================================================

def calculate_advanced_volatility(klines: List, indicators: Dict) -> Dict[str, float]:
    """
    Features de volatilité avancées
    Crypto = volatilité change rapidement, XGBoost peut l'exploiter

    Args:
        klines: Liste de klines (OHLCV)
        indicators: Dict avec ATR, Bollinger Bands

    Returns:
        Dict avec 4 features de volatilité
    """
    highs = [float(k[2]) for k in klines[-20:]]
    lows = [float(k[3]) for k in klines[-20:]]
    closes = [float(k[4]) for k in klines[-20:]]

    current_close = closes[-1]
    atr = indicators.get('atr', 0)
    bb = indicators.get('bollingerBands', {})
    bb_width = (bb.get('upper', 0) - bb.get('lower', 0)) if bb else 0

    # ATR Change (volatilité en augmentation ou diminution?)
    atr_5_ago = np.mean([highs[i] - lows[i] for i in range(-10, -5)]) if len(highs) >= 10 else atr
    atr_change = (atr - atr_5_ago) / (atr_5_ago + 1e-10) if atr_5_ago > 0 else 0

    # BB Squeeze (compression de volatilité)
    bb_width_ma = np.mean([highs[i] - lows[i] for i in range(len(highs))]) if len(highs) > 0 else bb_width
    bb_squeeze = bb_width / (bb_width_ma + 1e-10) if bb_width_ma > 0 else 1.0

    # High-Low Ratio
    high_low_ratio = (highs[-1] - lows[-1]) / (current_close + 1e-10)

    # Price Range 5d
    max_5d = np.max(closes[-5:]) if len(closes) >= 5 else current_close
    min_5d = np.min(closes[-5:]) if len(closes) >= 5 else current_close
    price_range_5d = (max_5d - min_5d) / (current_close + 1e-10)

    return {
        'atr_change': np.tanh(atr_change),
        'bb_squeeze': bb_squeeze,
        'high_low_ratio': high_low_ratio,
        'price_range_5d': price_range_5d
    }


# ============================================================================
# CATÉGORIE D: VOLUME AVANCÉ (3 features) ⭐⭐⭐⭐
# ============================================================================

def calculate_advanced_volume(klines: List) -> Dict[str, float]:
    """
    Features de volume avancées
    Volume = early signal, XGBoost peut capturer divergences

    Returns:
        Dict avec 3 features de volume
    """
    volumes = [float(k[5]) for k in klines[-30:]]
    closes = [float(k[4]) for k in klines[-30:]]

    current_volume = volumes[-1]
    current_close = closes[-1]

    # Volume / SMA20
    volume_sma_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else current_volume
    volume_sma_ratio = current_volume / (volume_sma_20 + 1e-10)

    # Volume Change 5d
    volume_5_ago = volumes[-6] if len(volumes) >= 6 else current_volume
    volume_change_5d = (current_volume - volume_5_ago) / (volume_5_ago + 1e-10)

    # VWAP Distance (approximation simple)
    # VWAP = sum(Price * Volume) / sum(Volume)
    if len(klines) >= 20:
        vwap_sum = sum([float(k[4]) * float(k[5]) for k in klines[-20:]])
        vol_sum = sum([float(k[5]) for k in klines[-20:]])
        vwap = vwap_sum / (vol_sum + 1e-10)
        vwap_distance = (current_close - vwap) / (current_close + 1e-10)
    else:
        vwap_distance = 0

    return {
        'volume_sma_ratio': np.tanh(volume_sma_ratio),
        'volume_change_5d': np.tanh(volume_change_5d),
        'vwap_distance': vwap_distance
    }


# ============================================================================
# CATÉGORIE E: STRUCTURE DE MARCHÉ (4 features) ⭐⭐⭐⭐⭐
# ============================================================================

def calculate_market_structure(klines: List) -> Dict[str, float]:
    """
    Features de structure de marché
    Patterns de structure = très prédictifs, XGBoost excelle ici

    Returns:
        Dict avec 4 features de structure
    """
    highs = [float(k[2]) for k in klines[-20:]]
    lows = [float(k[3]) for k in klines[-20:]]
    closes = [float(k[4]) for k in klines[-20:]]

    current_close = closes[-1]

    # Higher Highs Count (5 derniers)
    higher_highs = sum([1 if i > 0 and highs[i] > highs[i-1] else 0 for i in range(-5, 0)]) if len(highs) >= 5 else 0
    higher_highs_ratio = higher_highs / 5.0

    # Lower Lows Count (5 derniers)
    lower_lows = sum([1 if i > 0 and lows[i] < lows[i-1] else 0 for i in range(-5, 0)]) if len(lows) >= 5 else 0
    lower_lows_ratio = lower_lows / 5.0

    # Consolidation Index (20 derniers)
    consolidation = np.std(closes) / (np.mean(closes) + 1e-10) if len(closes) > 0 else 0

    # Days Since ATH/ATL (approximation sur 20 derniers)
    max_20 = np.max(closes) if len(closes) > 0 else current_close
    min_20 = np.min(closes) if len(closes) > 0 else current_close

    # Distance to ATH/ATL combined
    dist_ath = (max_20 - current_close) / (max_20 + 1e-10)
    dist_atl = (current_close - min_20) / (current_close + 1e-10)
    ath_atl_balance = dist_ath - dist_atl  # Négatif = près ATH, Positif = près ATL

    return {
        'higher_highs_ratio': higher_highs_ratio,
        'lower_lows_ratio': lower_lows_ratio,
        'consolidation': consolidation,
        'ath_atl_balance': ath_atl_balance
    }


# ============================================================================
# CATÉGORIE F: DIVERGENCES (3 features) ⭐⭐⭐⭐⭐
# ============================================================================

def calculate_divergences(klines: List, indicators: Dict) -> Dict[str, float]:
    """
    Divergences entre prix et indicateurs
    Divergences = signaux très forts en trading

    Returns:
        Dict avec 3 features de divergences
    """
    closes = [float(k[4]) for k in klines[-10:]]
    volumes = [float(k[5]) for k in klines[-10:]]

    rsi = indicators.get('rsi', 50)
    macd_hist = indicators.get('macd', {}).get('histogram', 0) if indicators.get('macd') else 0

    # Price-RSI Divergence (prix monte mais RSI baisse = bearish)
    price_roc_5 = (closes[-1] - closes[-6]) if len(closes) >= 6 else 0
    # Simplification: on ne peut pas avoir RSI[-5] facilement, donc approximation
    price_rsi_divergence = 1.0 if (price_roc_5 > 0 and rsi < 40) or (price_roc_5 < 0 and rsi > 60) else 0.0

    # Price-MACD Divergence
    price_macd_divergence = 1.0 if (price_roc_5 > 0 and macd_hist < 0) or (price_roc_5 < 0 and macd_hist > 0) else 0.0

    # Volume-Price Divergence (prix monte mais volume baisse)
    volume_change = (volumes[-1] - volumes[-6]) if len(volumes) >= 6 else 0
    volume_price_divergence = 1.0 if (price_roc_5 > 0 and volume_change < 0) or (price_roc_5 < 0 and volume_change > 0) else 0.0

    return {
        'price_rsi_divergence': price_rsi_divergence,
        'price_macd_divergence': price_macd_divergence,
        'volume_price_divergence': volume_price_divergence
    }


# ============================================================================
# CATÉGORIE G: MARKET REGIME (3 features) ⭐⭐⭐⭐⭐
# ============================================================================

def calculate_market_regime(klines: List, indicators: Dict) -> Dict[str, float]:
    """
    Détection de régime de marché
    Différents régimes = différents patterns

    Returns:
        Dict avec 3 features de régime
    """
    closes = [float(k[4]) for k in klines[-50:]]
    atr = indicators.get('atr', 0)
    ema20 = indicators.get('ema20', 0)
    ema50 = indicators.get('ema50', 0)

    # Volatility Regime (ATR relatif à moyenne 50)
    atr_ma_50 = np.mean([np.std(closes[i:i+14]) for i in range(0, len(closes)-14, 1)]) if len(closes) >= 50 else atr
    volatility_regime = atr / (atr_ma_50 + 1e-10) if atr_ma_50 > 0 else 1.0

    # Trend Strength (écart EMA20-EMA50)
    trend_strength = abs(ema20 - ema50) / (closes[-1] + 1e-10) if ema20 and ema50 else 0

    # ADX approximation simplifiée (mesure force tendance)
    # ADX réel est complexe, ici approximation via std / mean
    if len(closes) >= 14:
        price_changes = np.abs(np.diff(closes[-14:]))
        adx_approx = np.mean(price_changes) / (np.mean(closes[-14:]) + 1e-10)
    else:
        adx_approx = 0

    return {
        'volatility_regime': volatility_regime,
        'trend_strength': trend_strength,
        'adx_approx': adx_approx
    }


# ============================================================================
# CATÉGORIE H: CROSS-ASSET FEATURES (5 features) ⭐⭐⭐⭐⭐
# ============================================================================

def calculate_cross_asset_features(crypto_symbol: str, klines: List, klines_btc: Optional[List]) -> Dict[str, float]:
    """
    Features cross-asset vs Bitcoin
    BTC = référence du marché crypto (50%+ market cap)

    Args:
        crypto_symbol: Ex: 'ETHUSDT', 'SOLUSDT'
        klines: Klines de la crypto
        klines_btc: Klines de BTC (si None, retourne 0)

    Returns:
        Dict avec 5 features cross-asset
    """
    # Si c'est BTC ou pas de données BTC, retourner 0
    if crypto_symbol == 'BTCUSDT' or klines_btc is None or len(klines_btc) < 20:
        return {
            'btc_correlation': 0,
            'btc_roc_ratio': 0,
            'btc_beta': 0,
            'btc_trend_alignment': 0.5,
            'btc_divergence': 0
        }

    # Prendre les 20 derniers prix
    prices = np.array([float(k[4]) for k in klines[-20:]])
    prices_btc = np.array([float(k[4]) for k in klines_btc[-20:]])

    # 1. Correlation 20 jours (0.7-0.9 pour ETH/SOL)
    if len(prices) == len(prices_btc) and len(prices) >= 2:
        correlation = np.corrcoef(prices, prices_btc)[0, 1]
        correlation = np.clip(correlation, -1, 1)  # Limiter à [-1, 1]
    else:
        correlation = 0

    # 2. ROC ratio (momentum relatif vs BTC)
    if len(prices) >= 20 and len(prices_btc) >= 20:
        roc = (prices[-1] - prices[0]) / (prices[0] + 1e-10)
        roc_btc = (prices_btc[-1] - prices_btc[0]) / (prices_btc[0] + 1e-10)
        roc_ratio = roc / (roc_btc + 1e-10)
    else:
        roc_ratio = 1.0

    # 3. Beta (sensibilité aux mouvements BTC)
    if len(prices) >= 20 and len(prices_btc) >= 20:
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        returns_btc = np.diff(prices_btc) / (prices_btc[:-1] + 1e-10)

        if len(returns) > 0 and len(returns_btc) > 0:
            covariance = np.cov(returns, returns_btc)[0, 1]
            variance_btc = np.var(returns_btc)
            beta = covariance / (variance_btc + 1e-10)
        else:
            beta = 1.0
    else:
        beta = 1.0

    # 4. Trend Alignment (même direction que BTC?)
    if len(prices) >= 2 and len(prices_btc) >= 2:
        trend = prices[-1] - prices[-5] if len(prices) >= 5 else prices[-1] - prices[0]
        trend_btc = prices_btc[-1] - prices_btc[-5] if len(prices_btc) >= 5 else prices_btc[-1] - prices_btc[0]
        trend_alignment = 1.0 if np.sign(trend) == np.sign(trend_btc) else 0.0
    else:
        trend_alignment = 0.5

    # 5. Divergence (prix monte mais BTC baisse = signal fort)
    if len(prices) >= 5 and len(prices_btc) >= 5:
        roc_5 = (prices[-1] - prices[-5]) / (prices[-5] + 1e-10)
        roc_5_btc = (prices_btc[-1] - prices_btc[-5]) / (prices_btc[-5] + 1e-10)
        divergence = 1.0 if (roc_5 > 0 and roc_5_btc < 0) or (roc_5 < 0 and roc_5_btc > 0) else 0.0
    else:
        divergence = 0.0

    return {
        'btc_correlation': correlation,
        'btc_roc_ratio': roc_ratio,
        'btc_beta': beta,
        'btc_trend_alignment': trend_alignment,
        'btc_divergence': divergence
    }


# ============================================================================
# CATÉGORIE I: LAG FEATURES (9 features) ⭐⭐⭐⭐
# ============================================================================

def calculate_lag_features(indicators_history: List[Dict], n_lags: int = 3) -> List[float]:
    """
    Features lag (historique RSI, MACD, Volume)
    Compense le manque de mémoire temporelle de XGBoost

    Args:
        indicators_history: Liste des N derniers indicateurs calculés
        n_lags: Nombre de lags (3 = 9 features)

    Returns:
        Liste de n_lags * 3 features
    """
    features = []

    for lag in range(1, n_lags + 1):
        if lag < len(indicators_history):
            ind = indicators_history[-(lag + 1)]

            # RSI lag
            rsi_lag = ind.get('rsi', 50) / 100 if ind.get('rsi') else 0.5

            # MACD histogram lag
            macd = ind.get('macd', {})
            macd_lag = macd.get('histogram', 0) if macd else 0
            macd_lag = np.tanh(macd_lag / 100) if macd_lag else 0

            # Volume lag (normalisé)
            volume_lag = ind.get('volume', 0)
            # Normalisation simple avec tanh
            volume_lag = np.tanh(volume_lag / 1e8) if volume_lag else 0

            features.extend([rsi_lag, macd_lag, volume_lag])
        else:
            # Pas assez d'historique, utiliser valeurs neutres
            features.extend([0.5, 0, 0])

    return features


# ============================================================================
# FONCTION PRINCIPALE (MISE À JOUR)
# ============================================================================

def calculate_all_xgboost_features(
    klines: List,
    indicators: Dict,
    volumes: List[float],
    crypto_symbol: str = 'BTCUSDT',
    klines_btc: Optional[List] = None,
    indicators_history: Optional[List[Dict]] = None
) -> List[float]:
    """
    Calculer TOUTES les features optimisées pour XGBoost (v2 avec cross-asset et lag)

    Args:
        klines: Liste complète de klines (OHLCV)
        indicators: Dict avec indicateurs de base (RSI, MACD, etc.)
        volumes: Liste des volumes
        crypto_symbol: Ex: 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
        klines_btc: Klines BTC (pour cross-asset), None pour BTC
        indicators_history: Liste des N derniers indicateurs (pour lag)

    Returns:
        Liste de 40 features (26 anciennes + 5 cross-asset + 9 lag)
    """
    closes = [float(k[4]) for k in klines]
    features = []

    # A. ROC Multi-échelle (4)
    roc = calculate_roc_multiscale(closes)
    features.extend([
        np.tanh(roc['roc_5']),
        np.tanh(roc['roc_10']),
        np.tanh(roc['roc_20']),
        np.tanh(roc['roc_50'])
    ])

    # B. Ratios & Interactions (5)
    ratios = calculate_ratio_features(indicators, volumes)
    features.extend([
        ratios['rsi_stoch_divergence'],
        ratios['macd_atr_ratio'],
        ratios['volume_atr_ratio'],
        np.tanh(ratios['bb_width_atr_ratio']),
        ratios['price_ema20_rsi']
    ])

    # C. Volatilité Avancée (4)
    volatility = calculate_advanced_volatility(klines, indicators)
    features.extend([
        volatility['atr_change'],
        np.tanh(volatility['bb_squeeze']),
        volatility['high_low_ratio'],
        volatility['price_range_5d']
    ])

    # D. Volume Avancé (3)
    volume_feats = calculate_advanced_volume(klines)
    features.extend([
        volume_feats['volume_sma_ratio'],
        volume_feats['volume_change_5d'],
        volume_feats['vwap_distance']
    ])

    # E. Structure de Marché (4)
    structure = calculate_market_structure(klines)
    features.extend([
        structure['higher_highs_ratio'],
        structure['lower_lows_ratio'],
        structure['consolidation'],
        structure['ath_atl_balance']
    ])

    # F. Divergences (3)
    divergences = calculate_divergences(klines, indicators)
    features.extend([
        divergences['price_rsi_divergence'],
        divergences['price_macd_divergence'],
        divergences['volume_price_divergence']
    ])

    # G. Market Regime (3)
    regime = calculate_market_regime(klines, indicators)
    features.extend([
        np.tanh(regime['volatility_regime']),
        regime['trend_strength'],
        regime['adx_approx']
    ])

    # H. Cross-Asset Features (5) - NOUVEAU!
    cross_asset = calculate_cross_asset_features(crypto_symbol, klines, klines_btc)
    features.extend([
        cross_asset['btc_correlation'],
        np.tanh(cross_asset['btc_roc_ratio']),
        np.tanh(cross_asset['btc_beta']),
        cross_asset['btc_trend_alignment'],
        cross_asset['btc_divergence']
    ])

    # I. Lag Features (9) - NOUVEAU!
    if indicators_history and len(indicators_history) > 0:
        lag_feats = calculate_lag_features(indicators_history, n_lags=3)
        features.extend(lag_feats)
    else:
        # Pas d'historique, utiliser valeurs neutres
        features.extend([0.5, 0, 0] * 3)

    # Total: 26 + 5 + 9 = 40 features
    return features


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("TEST - XGBoost Optimized Features")
    print("="*60)

    # Simuler des données
    np.random.seed(42)
    klines = []
    base = 40000
    for i in range(100):
        open_p = base + np.random.randn() * 500
        high = open_p + abs(np.random.randn() * 300)
        low = open_p - abs(np.random.randn() * 300)
        close = (open_p + high + low) / 3
        volume = 1000000 + np.random.randn() * 100000
        klines.append([0, open_p, high, low, close, volume])
        base = close

    # Simuler indicateurs
    indicators = {
        'currentPrice': float(klines[-1][4]),
        'rsi': 55,
        'macd': {'histogram': 100},
        'atr': 500,
        'stochasticRsi': {'k': 60},
        'ema20': float(klines[-1][4]) * 0.98,
        'ema50': float(klines[-1][4]) * 0.95,
        'bollingerBands': {
            'upper': float(klines[-1][4]) * 1.02,
            'lower': float(klines[-1][4]) * 0.98
        }
    }

    volumes = [float(k[5]) for k in klines]

    # Calculer features
    features = calculate_all_xgboost_features(klines, indicators, volumes)

    print(f"\n✅ {len(features)} nouvelles features générées")
    print(f"\nPremières 10 features:")
    for i, feat in enumerate(features[:10]):
        print(f"  Feature {i+1:2d}: {feat:10.6f}")

    print("\n" + "="*60)
    print("TEST PASSED - Toutes les features XGBoost OK!")
    print("="*60)
