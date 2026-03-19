"""
Analyse de Feature Importance pour XGBoost
Affiche les features les plus importantes pour chaque modèle
"""

import os
import sys
import pickle
import numpy as np

# Matplotlib optionnel (pour graphiques)
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend non-interactif
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib not available, graphs will be skipped")

# Liste des features (69 total)
FEATURE_NAMES = [
    # Base features (29)
    'rsi', 'rsi_distance_50',
    'ema20_dist', 'ema50_dist', 'ema200_dist',
    'macd', 'macd_signal', 'macd_histogram',
    'bb_middle_dist', 'bb_width', 'bb_position',
    'stoch_k', 'stoch_d',
    'atr',
    'obv',
    # Phase 1 (11)
    'volatility_short', 'volatility_long',
    'momentum_5', 'momentum_10', 'momentum_20',
    'mean_reversion_5', 'mean_reversion_20',
    'price_acceleration',
    'higher_highs_ratio', 'lower_lows_ratio',
    'price_range_ratio',
    # XGBoost features (40)
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

MODELS = [
    ('Bitcoin', '1d', './models/xgboost/bitcoin_1d_xgboost.pkl'),
    ('Bitcoin', '1w', './models/xgboost/bitcoin_1w_xgboost.pkl'),
    ('Ethereum', '1d', './models/xgboost/ethereum_1d_xgboost.pkl'),
    ('Ethereum', '1w', './models/xgboost/ethereum_1w_xgboost.pkl'),
    ('Solana', '1d', './models/xgboost/solana_1d_xgboost.pkl'),
    ('Solana', '1w', './models/xgboost/solana_1w_xgboost.pkl')
]


def analyze_model_importance(crypto, timeframe, model_path):
    """Analyser l'importance des features pour un modèle"""

    print(f"\n{'='*70}")
    print(f"{crypto} {timeframe.upper()} - Feature Importance Analysis")
    print('='*70)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    # Charger le modèle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Obtenir feature importance
    importance = model.feature_importances_

    # Vérifier nombre de features
    if len(importance) != len(FEATURE_NAMES):
        print(f"[WARNING] Expected {len(FEATURE_NAMES)} features, got {len(importance)}")
        # Ajuster si nécessaire
        feature_names = FEATURE_NAMES[:len(importance)]
    else:
        feature_names = FEATURE_NAMES

    # Créer liste triée
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Afficher top 20
    print(f"\n📊 TOP 20 Features:")
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'%'}")
    print('-' * 70)

    total_importance = sum(importance)
    cumulative = 0

    for i, (name, imp) in enumerate(feature_importance[:20], 1):
        pct = (imp / total_importance) * 100
        cumulative += pct
        print(f"{i:<6} {name:<30} {imp:<12.6f} {pct:>5.2f}%  (cum: {cumulative:.1f}%)")

    # Analyser catégories
    print(f"\n📈 Feature Importance by Category:")
    print('-' * 70)

    categories = {
        'Cross-Asset BTC': feature_importance[feature_names.index('btc_correlation'):feature_names.index('btc_correlation')+5],
        'Lag Features': feature_importance[feature_names.index('rsi_lag1'):feature_names.index('rsi_lag1')+9],
        'ROC Multi-scale': feature_importance[feature_names.index('roc_5'):feature_names.index('roc_5')+4],
        'Ratios': feature_importance[feature_names.index('rsi_stoch_div'):feature_names.index('rsi_stoch_div')+5],
        'Structure': feature_importance[feature_names.index('higher_highs'):feature_names.index('higher_highs')+4],
        'Volatility': feature_importance[feature_names.index('atr_change'):feature_names.index('atr_change')+4],
        'Volume': feature_importance[feature_names.index('volume_sma_ratio'):feature_names.index('volume_sma_ratio')+3],
    }

    category_totals = []
    for cat_name, features in categories.items():
        cat_importance = sum([imp for _, imp in features])
        cat_pct = (cat_importance / total_importance) * 100
        category_totals.append((cat_name, cat_pct))

    category_totals.sort(key=lambda x: x[1], reverse=True)

    for cat, pct in category_totals:
        print(f"{cat:<25} {pct:>6.2f}%")

    # Sauvegarder graphique (si matplotlib disponible)
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(12, 8))
            names = [name for name, _ in feature_importance[:20]]
            values = [imp for _, imp in feature_importance[:20]]

            plt.barh(range(len(names)), values)
            plt.yticks(range(len(names)), names)
            plt.xlabel('Importance')
            plt.title(f'{crypto} {timeframe.upper()} - Top 20 Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            output_file = f'./feature_importance_{crypto.lower()}_{timeframe}.png'
            plt.savefig(output_file)
            print(f"\n💾 Graph saved: {output_file}")
            plt.close()
        except Exception as e:
            print(f"[WARNING] Could not save graph: {e}")

    return feature_importance


def main():
    print("\n" + "="*70)
    print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
    print("Analyzing 69 features across 6 models")
    print("="*70)

    all_results = {}

    for crypto, timeframe, model_path in MODELS:
        result = analyze_model_importance(crypto, timeframe, model_path)
        if result:
            all_results[f"{crypto}_{timeframe}"] = result

    # Résumé global
    print(f"\n{'='*70}")
    print("GLOBAL SUMMARY - Most Important Features Overall")
    print('='*70)

    # Agréger importance sur tous les modèles
    global_importance = {}
    for model_results in all_results.values():
        for feature, importance in model_results:
            if feature not in global_importance:
                global_importance[feature] = []
            global_importance[feature].append(importance)

    # Moyenne d'importance
    global_avg = [(feat, np.mean(imps)) for feat, imps in global_importance.items()]
    global_avg.sort(key=lambda x: x[1], reverse=True)

    print(f"\n📊 TOP 20 Features (averaged across all models):")
    print(f"{'Rank':<6} {'Feature':<30} {'Avg Importance':<15}")
    print('-' * 70)

    for i, (name, avg_imp) in enumerate(global_avg[:20], 1):
        print(f"{i:<6} {name:<30} {avg_imp:<15.6f}")

    # Analyser si cross-asset et lag features sont utilisés
    print(f"\n🔍 Cross-Asset & Lag Features Usage:")
    print('-' * 70)

    btc_features = [f for f, _ in global_avg if f.startswith('btc_')]
    lag_features = [f for f, _ in global_avg if 'lag' in f]

    print(f"\nCross-Asset BTC features in top 20:")
    btc_in_top20 = [f for f, _ in global_avg[:20] if f.startswith('btc_')]
    if btc_in_top20:
        for feat in btc_in_top20:
            rank = [i for i, (f, _) in enumerate(global_avg, 1) if f == feat][0]
            print(f"  ✅ {feat} (rank {rank})")
    else:
        print("  ❌ None in top 20")

    print(f"\nLag features in top 20:")
    lag_in_top20 = [f for f, _ in global_avg[:20] if 'lag' in f]
    if lag_in_top20:
        for feat in lag_in_top20:
            rank = [i for i, (f, _) in enumerate(global_avg, 1) if f == feat][0]
            print(f"  ✅ {feat} (rank {rank})")
    else:
        print("  ❌ None in top 20")

    print("\n" + "="*70)
    print("✅ Feature importance analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
