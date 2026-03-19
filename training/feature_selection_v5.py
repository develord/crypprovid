"""
Feature Selection pour XGBoost v5
Sélectionne les TOP 40 features les plus importantes (au lieu de 69)
Basé sur l'analyse d'importance des modèles v4
"""

# TOP 40 FEATURES sélectionnées (ordre d'importance globale)
SELECTED_FEATURES_V5 = [
    # TOP 20 Global (importance moyenne la plus élevée)
    'macd_lag2',            # #1 - 3.23%
    'macd_lag3',            # #2 - 3.11%
    'macd',                 # #3 - 2.50%
    'macd_histogram',       # #4 - 2.30%
    'macd_signal',          # #5 - 2.28%
    'bb_squeeze',           # #6 - 2.07%
    'btc_beta',             # #7 - 2.06% (cross-asset)
    'price_range_ratio',    # #8 - 2.01%
    'volume_sma_ratio',     # #9 - 1.99%
    'price_macd_div',       # #10 - 1.96%
    'ema50_dist',           # #11 - 1.96%
    'btc_roc_ratio',        # #12 - 1.91% (cross-asset)
    'volatility_long',      # #13 - 1.89%
    'atr_change',           # #14 - 1.87%
    'ema200_dist',          # #15 - 1.84%
    'mean_reversion_20',    # #16 - 1.81%
    'bb_width',             # #17 - 1.80%
    'volatility_short',     # #18 - 1.79%
    'high_low_ratio',       # #19 - 1.78%
    'volume_atr_ratio',     # #20 - 1.69%

    # TOP 21-40 (features importantes suivantes)
    'consolidation',        # Structure de marché
    'rsi_lag1',             # Lag RSI
    'volume_lag1',          # Lag Volume
    'btc_correlation',      # Cross-asset
    'price_ema20_rsi',      # Ratio
    'rsi',                  # Indicateur de base
    'bb_position',          # Bollinger
    'bb_middle_dist',       # Bollinger
    'atr',                  # ATR de base
    'momentum_20',          # Phase 1
    'macd_atr_ratio',       # Ratio
    'obv',                  # Volume
    'price_range_5d',       # Volatility
    'lower_lows_ratio',     # Structure
    'btc_trend_alignment',  # Cross-asset
    'volume_price_div',     # Divergence
    'vwap_distance',        # Volume
    'rsi_stoch_div',        # Ratio
    'trend_strength',       # Regime
    'volatility_regime'     # Regime
]

# Mapping des indices pour extraction
# Base features (29): 0-28
# XGBoost features (40): 29-68
FEATURE_NAMES_FULL_69 = [
    # Base features (29): indices 0-28
    'rsi', 'rsi_distance_50',
    'ema20_dist', 'ema50_dist', 'ema200_dist',
    'macd', 'macd_signal', 'macd_histogram',
    'bb_middle_dist', 'bb_width', 'bb_position',
    'stoch_k', 'stoch_d',
    'atr',
    'obv',
    # Phase 1 (11): indices 15-25
    'volatility_short', 'volatility_long',
    'momentum_5', 'momentum_10', 'momentum_20',
    'mean_reversion_5', 'mean_reversion_20',
    'price_acceleration',
    'higher_highs_ratio', 'lower_lows_ratio',
    'price_range_ratio',
    # XGBoost features (40): indices 26-68
    # ROC (4)
    'roc_5', 'roc_10', 'roc_20', 'roc_50',
    # Ratios (5)
    'rsi_stoch_div', 'macd_atr_ratio', 'volume_atr_ratio',
    'bb_width_atr_ratio', 'price_ema20_rsi',
    # Volatility (4)
    'atr_change', 'bb_squeeze', 'high_low_ratio', 'price_range_5d',
    # Volume (3)
    'volume_sma_ratio', 'volume_change_5d', 'vwap_distance',
    # Structure (4)
    'higher_highs', 'lower_lows', 'consolidation', 'ath_atl_balance',
    # Divergences (3)
    'price_rsi_div', 'price_macd_div', 'volume_price_div',
    # Regime (3)
    'volatility_regime', 'trend_strength', 'adx_approx',
    # Cross-asset BTC (5)
    'btc_correlation', 'btc_roc_ratio', 'btc_beta',
    'btc_trend_alignment', 'btc_divergence',
    # Lag features (9)
    'rsi_lag1', 'macd_lag1', 'volume_lag1',
    'rsi_lag2', 'macd_lag2', 'volume_lag2',
    'rsi_lag3', 'macd_lag3', 'volume_lag3'
]


def get_feature_indices(selected_features):
    """Retourne les indices des features sélectionnées dans le vecteur 69"""
    indices = []
    for feat in selected_features:
        if feat in FEATURE_NAMES_FULL_69:
            indices.append(FEATURE_NAMES_FULL_69.index(feat))
        else:
            print(f"[WARNING] Feature not found: {feat}")
    return indices


def select_features_from_vector(features_69, selected_features):
    """
    Extrait les features sélectionnées d'un vecteur de 69 features

    Args:
        features_69: Liste de 69 features
        selected_features: Liste des noms de features à garder

    Returns:
        Liste des features sélectionnées (40 ou moins)
    """
    indices = get_feature_indices(selected_features)
    return [features_69[i] for i in indices]


def print_selection_summary():
    """Afficher résumé de la sélection"""
    print("\n" + "="*70)
    print("FEATURE SELECTION v5 - Summary")
    print("="*70)
    print(f"Original features (v4): 69")
    print(f"Selected features (v5): {len(SELECTED_FEATURES_V5)}")
    print(f"Reduction: {69 - len(SELECTED_FEATURES_V5)} features removed")
    print(f"Percentage kept: {len(SELECTED_FEATURES_V5)/69*100:.1f}%")

    print(f"\n📊 Selected Features by Category:")
    categories = {
        'MACD & Lags': ['macd', 'macd_histogram', 'macd_signal', 'macd_lag1', 'macd_lag2', 'macd_lag3', 'macd_atr_ratio'],
        'Cross-Asset BTC': ['btc_correlation', 'btc_beta', 'btc_roc_ratio', 'btc_trend_alignment'],
        'Lag Features': ['rsi_lag1', 'macd_lag1', 'volume_lag1', 'macd_lag2', 'macd_lag3'],
        'Volatility': ['volatility_short', 'volatility_long', 'atr_change', 'bb_squeeze', 'high_low_ratio', 'price_range_5d', 'volatility_regime'],
        'Bollinger Bands': ['bb_width', 'bb_position', 'bb_middle_dist', 'bb_squeeze'],
        'EMAs': ['ema50_dist', 'ema200_dist'],
        'Volume': ['volume_sma_ratio', 'volume_atr_ratio', 'obv', 'vwap_distance', 'volume_lag1', 'volume_price_div'],
        'Structure': ['consolidation', 'lower_lows_ratio', 'price_range_ratio'],
        'Momentum': ['momentum_20', 'mean_reversion_20'],
        'Regime': ['trend_strength', 'volatility_regime'],
        'Others': ['rsi', 'atr', 'price_ema20_rsi', 'rsi_stoch_div', 'price_macd_div']
    }

    for cat, feats in categories.items():
        selected_in_cat = [f for f in feats if f in SELECTED_FEATURES_V5]
        print(f"  {cat:<20}: {len(selected_in_cat)}/{len(feats)}")

    print("\n✅ Top 10 Features:")
    for i, feat in enumerate(SELECTED_FEATURES_V5[:10], 1):
        print(f"  {i}. {feat}")

    print("="*70)


if __name__ == '__main__':
    print_selection_summary()

    # Test extraction
    print("\n[TEST] Feature extraction:")
    dummy_features_69 = list(range(69))
    selected_40 = select_features_from_vector(dummy_features_69, SELECTED_FEATURES_V5)
    print(f"Input: {len(dummy_features_69)} features")
    print(f"Output: {len(selected_40)} features")
    print(f"Indices: {selected_40[:10]}...")
    print("\n✅ Feature selection ready for v5 training!")
