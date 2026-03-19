"""
Script d'entraînement XGBoost (v5 - 40 features optimisées)
Utilise les TOP 40 features sélectionnées par analyse d'importance
Version de production optimale
"""

import os
import sys
import json
import pickle
import numpy as np
import xgboost as xgb
from datetime import datetime

# Add data manager to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from data_manager import get_historical_data
from advanced_features import calculate_phase1_features, normalize_phase1_features

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration - Bitcoin + Ethereum + Solana
CRYPTOS = [
    {'cryptoId': 'bitcoin', 'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
    {'cryptoId': 'ethereum', 'symbol': 'ETHUSDT', 'name': 'Ethereum'},
    {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana'}
]

TIMEFRAME_CONFIGS = {
    '1d': {
        'interval': '1d',
        'limit': 3000,           # ↑ Augmenté de 2000 → 3000 (+50%)
        'lookahead': 7,
        'buy_threshold': 2.0,    # ±2% pour daily
        'sell_threshold': -2.0
    },
    '1w': {
        'interval': '1w',
        'limit': 800,            # ↑ Augmenté de 400 → 800 (+100%)
        'lookahead': 4,
        'buy_threshold': 3.5,    # ±3.5% pour weekly (plus large)
        'sell_threshold': -3.5
    }
}

# Importer les fonctions de train_models.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_models import (
    download_historical_data,
    calculate_indicators,
    prepare_features
)
from xgboost_features import calculate_all_xgboost_features
from feature_selection_v5 import SELECTED_FEATURES_V5, select_features_from_vector

def train_xgboost_model(crypto, timeframe_key='1d'):
    """Entraîner un modèle XGBoost avec cross-asset et lag features"""
    tf_config = TIMEFRAME_CONFIGS[timeframe_key]

    print(f"\n{'='*60}")
    print(f"[*] Training XGBoost: {crypto['name']} - {timeframe_key.upper()}")
    print('='*60)

    try:
        # 1. Télécharger données crypto
        print(f"  [>>] Downloading {tf_config['interval']} candles...")
        klines = download_historical_data(
            crypto['symbol'],
            interval=tf_config['interval'],
            limit=tf_config['limit']
        )

        if not klines or len(klines) < 100:
            print(f"  [ERROR] Insufficient data: {len(klines)} candles")
            return None

        print(f"  [OK] Downloaded {len(klines)} candles")

        # 1b. Télécharger données BTC en parallèle (pour cross-asset features)
        klines_btc = None
        if crypto['symbol'] != 'BTCUSDT':
            print(f"  [>>] Downloading BTC data for cross-asset features...")
            klines_btc = download_historical_data(
                'BTCUSDT',
                interval=tf_config['interval'],
                limit=tf_config['limit']
            )
            if klines_btc:
                print(f"  [OK] Downloaded {len(klines_btc)} BTC candles")
            else:
                print(f"  [WARNING] BTC data not available, cross-asset features = 0")

        # 2. Préparer features et labels avec lag et cross-asset
        print(f"  [>>] Preparing features and labels (with BTC cross-asset + lag)...")
        features_list = []
        labels_list = []
        indicators_history = []  # Stocker historique pour lag features

        for i in range(200, len(klines) - tf_config['lookahead']):
            # Calculer indicateurs
            window_data = klines[max(0, i-200):i+1]
            indicators = calculate_indicators(window_data)
            current_price = float(klines[i][4])

            # Stocker dans historique pour lag features
            indicators_history.append(indicators)
            # Garder seulement les 10 derniers pour économiser mémoire
            if len(indicators_history) > 10:
                indicators_history.pop(0)

            # Préparer features de base (29 features: 18 base + 11 Phase 1)
            prices_history = [float(k[4]) for k in window_data]
            features_base = prepare_features(indicators, current_price, prices_history)

            # Préparer données BTC pour cross-asset (alignées temporellement)
            window_data_btc = None
            if klines_btc and len(klines_btc) >= i+1:
                window_data_btc = klines_btc[max(0, i-200):i+1]

            # Ajouter features XGBoost optimisées (40 features: 26 + 5 cross-asset + 9 lag)
            volumes = [float(k[5]) for k in window_data]
            features_xgb = calculate_all_xgboost_features(
                window_data,
                indicators,
                volumes,
                crypto_symbol=crypto['symbol'],
                klines_btc=window_data_btc,
                indicators_history=indicators_history
            )

            # Combiner: 29 + 40 = 69 features total
            features_69 = features_base + features_xgb

            # v5: Sélectionner TOP 40 features optimisées
            features = select_features_from_vector(features_69, SELECTED_FEATURES_V5)

            # Calculer label (BUY/SELL/HOLD)
            # Seuils adaptatifs par timeframe (±2% pour 1d, ±3.5% pour 1w)
            future_price = float(klines[i + tf_config['lookahead']][4])
            change_pct = ((future_price - current_price) / current_price) * 100

            # Seuils depuis configuration
            buy_threshold = tf_config['buy_threshold']
            sell_threshold = tf_config['sell_threshold']

            if change_pct > buy_threshold:
                label = 0  # BUY
            elif change_pct < sell_threshold:
                label = 1  # SELL
            else:
                label = 2  # HOLD

            features_list.append(features)
            labels_list.append(label)

        X = np.array(features_list)
        y = np.array(labels_list)

        print(f"  [OK] Prepared {len(X)} samples with {X.shape[1]} features")

        # 3. Split train/validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"  [>>] Training XGBoost model...")
        print(f"      Train: {len(X_train)} | Validation: {len(X_val)}")

        # 4. Entraîner XGBoost - OPTIMIZED v5 (69 features + équilibre overfit/underfit)
        # Hyperparamètres ÉQUILIBRÉS: ni trop conservateurs ni trop agressifs
        model = xgb.XGBClassifier(
            n_estimators=250,          # ↑ de 200 (plus d'arbres pour mieux apprendre)
            max_depth=5,               # ↑ de 4 (plus de profondeur)
            learning_rate=0.04,        # ↑ de 0.03 (apprentissage plus rapide)
            subsample=0.75,            # ↑ de 0.7 (plus de données)
            colsample_bytree=0.65,     # ↑ de 0.6 (plus de features)
            colsample_bylevel=0.75,    # ↑ de 0.7 (plus de diversité)
            min_child_weight=3,        # ↓ de 5 (moins conservateur)
            gamma=0.2,                 # ↓ de 0.3 (moins de pruning)
            reg_alpha=0.05,            # ↓ de 0.1 (L1 modérée)
            reg_lambda=2,              # ↓ de 3 (L2 modérée)
            scale_pos_weight=1,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss',
            early_stopping_rounds=25,  # ↑ de 20 (plus de patience)
            verbose=False
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # 5. Évaluer
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        print(f"  [OK] Training accuracy: {train_acc*100:.2f}%")
        print(f"  [OK] Validation accuracy: {val_acc*100:.2f}%")

        # 6. Sauvegarder
        model_dir = './models/xgboost'
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{crypto['cryptoId']}_{timeframe_key}_xgboost.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"  [OK] Model saved: {model_path}")

        return {
            'crypto': crypto['name'],
            'timeframe': timeframe_key,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'samples': len(X),
            'model_path': model_path
        }

    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "="*60)
    print("XGBOOST v5 - PRODUCTION TRAINING")
    print("Training with TOP 40 optimized features")
    print(f"Selected: {len(SELECTED_FEATURES_V5)} features (42% reduction)")
    print("="*60)

    results = []
    total = len(CRYPTOS) * len(TIMEFRAME_CONFIGS)
    current = 0

    for crypto in CRYPTOS:
        for timeframe_key in TIMEFRAME_CONFIGS.keys():
            current += 1
            print(f"\n[{current}/{total}] {crypto['name']} - {timeframe_key.upper()}")

            result = train_xgboost_model(crypto, timeframe_key)
            if result:
                results.append(result)

    # Résumé
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for result in results:
        print(f"\n{result['crypto']} - {result['timeframe'].upper()}:")
        print(f"   Train Acc: {result['train_accuracy']*100:.2f}%")
        print(f"   Val Acc:   {result['val_accuracy']*100:.2f}%")
        print(f"   Samples:   {result['samples']}")

    # Sauvegarder résultats
    with open('training_xgboost_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] {len(results)} XGBoost models trained successfully!")
    print("="*60)

if __name__ == '__main__':
    main()
