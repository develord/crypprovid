"""
Script de diagnostic pour analyser les prédictions des modèles
Affiche la distribution des prédictions pour chaque modèle
"""

import os
import sys
import pickle
import numpy as np
from collections import Counter

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))

from data_manager import get_historical_data
from train_models import calculate_indicators, prepare_features
from xgboost_features import calculate_all_xgboost_features
from feature_selection_v5 import SELECTED_FEATURES_V5, select_features_from_vector

CRYPTOS = [
    {'cryptoId': 'bitcoin', 'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
    {'cryptoId': 'ethereum', 'symbol': 'ETHUSDT', 'name': 'Ethereum'},
    {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana'}
]

TIMEFRAMES = {
    '1d': {'interval': '1d', 'limit': 3000},
    '1w': {'interval': '1w', 'limit': 800}
}

def diagnose_model(crypto, timeframe_key):
    """Diagnostiquer les prédictions d'un modèle"""

    model_path = f'./training/models/xgboost/{crypto["cryptoId"]}_{timeframe_key}_xgboost.pkl'

    if not os.path.exists(model_path):
        print(f"  [ERROR] Model not found: {model_path}")
        return None

    # Charger modèle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Télécharger données
    tf_config = TIMEFRAMES[timeframe_key]
    klines = get_historical_data(
        crypto['symbol'],
        interval=tf_config['interval'],
        limit=tf_config['limit']
    )

    if not klines or len(klines) < 500:
        print(f"  [ERROR] Insufficient data")
        return None

    # BTC pour cross-asset
    klines_btc = None
    if crypto['symbol'] != 'BTCUSDT':
        klines_btc = get_historical_data(
            'BTCUSDT',
            interval=tf_config['interval'],
            limit=tf_config['limit']
        )

    # Dernières 100 périodes
    test_start = max(200, len(klines) - 104)  # ~2 ans pour weekly
    predictions = []
    indicators_history = []

    for i in range(test_start, len(klines) - 4):
        window_data = klines[max(0, i-200):i+1]
        indicators = calculate_indicators(window_data)
        current_price = float(klines[i][4])

        indicators_history.append(indicators)
        if len(indicators_history) > 10:
            indicators_history.pop(0)

        prices_history = [float(k[4]) for k in window_data]
        features_base = prepare_features(indicators, current_price, prices_history)

        window_data_btc = None
        if klines_btc and len(klines_btc) >= i+1:
            window_data_btc = klines_btc[max(0, i-200):i+1]

        volumes = [float(k[5]) for k in window_data]
        features_xgb = calculate_all_xgboost_features(
            window_data,
            indicators,
            volumes,
            crypto_symbol=crypto['symbol'],
            klines_btc=window_data_btc,
            indicators_history=indicators_history
        )

        features_69 = features_base + features_xgb
        features = select_features_from_vector(features_69, SELECTED_FEATURES_V5)

        X = np.array([features])
        prediction = model.predict(X)[0]
        predictions.append(prediction)

    # Analyser distribution
    counter = Counter(predictions)
    total = len(predictions)

    return {
        'total_predictions': total,
        'buy_count': counter.get(0, 0),
        'sell_count': counter.get(1, 0),
        'hold_count': counter.get(2, 0),
        'buy_pct': (counter.get(0, 0) / total * 100) if total > 0 else 0,
        'sell_pct': (counter.get(1, 0) / total * 100) if total > 0 else 0,
        'hold_pct': (counter.get(2, 0) / total * 100) if total > 0 else 0
    }


def main():
    print("\n" + "="*80)
    print("DIAGNOSTIC DES PRÉDICTIONS - XGBOOST v5")
    print("="*80)

    results = []

    for crypto in CRYPTOS:
        for timeframe_key in ['1d', '1w']:
            print(f"\n[>>] {crypto['name']} {timeframe_key.upper()}...")

            stats = diagnose_model(crypto, timeframe_key)

            if stats:
                results.append({
                    'crypto': crypto['name'],
                    'timeframe': timeframe_key,
                    **stats
                })

    # Tableau résumé
    print("\n" + "="*100)
    print(f"{'Modèle':<20} {'Total':<10} {'BUY':<15} {'SELL':<15} {'HOLD':<15}")
    print("-"*100)

    for r in results:
        print(f"{r['crypto']} {r['timeframe'].upper():<15} "
              f"{r['total_predictions']:>6}    "
              f"{r['buy_count']:>4} ({r['buy_pct']:>5.1f}%)   "
              f"{r['sell_count']:>4} ({r['sell_pct']:>5.1f}%)   "
              f"{r['hold_count']:>4} ({r['hold_pct']:>5.1f}%)")

    print("="*100)

    print("\nNote: BUY=0, SELL=1, HOLD=2")
    print("Un modèle qui prédit toujours HOLD ne génère aucun trade dans le backtest.")
    print("="*80)


if __name__ == '__main__':
    main()
