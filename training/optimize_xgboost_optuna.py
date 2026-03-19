"""
Script d'optimisation XGBoost avec Optuna
Trouve les meilleurs hyperparamètres pour chaque crypto/timeframe
Puis ré-entraîne les modèles avec les paramètres optimaux
"""

import os
import sys
import json
import pickle
import numpy as np
import xgboost as xgb
import optuna
from datetime import datetime
from optuna.pruners import MedianPruner

# Add data manager to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from data_manager import get_historical_data
from advanced_features import calculate_phase1_features, normalize_phase1_features

# Fix Windows encoding and handle closed stdout
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass  # stdout already configured or closed

# Create safe print function that handles closed stdout
_original_print = print
def safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except (ValueError, OSError):
        pass  # stdout closed, continue silently

# Replace built-in print
import builtins
builtins.print = safe_print

# Configuration
CRYPTOS = [
    {'cryptoId': 'bitcoin', 'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
    {'cryptoId': 'ethereum', 'symbol': 'ETHUSDT', 'name': 'Ethereum'},
    {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana'}
]

TIMEFRAME_CONFIGS = {
    '1d': {
        'interval': '1d',
        'limit': 3000,
        'lookahead': 7,
        'buy_threshold': 2.0,
        'sell_threshold': -2.0
    },
    '1w': {
        'interval': '1w',
        'limit': 800,
        'lookahead': 4,
        'buy_threshold': 3.5,
        'sell_threshold': -3.5
    }
}

# Import functions directly from train_models_xgboost
sys.path.insert(0, os.path.dirname(__file__))

# Import train_models_xgboost module to reuse prepare_data logic
import train_models_xgboost

download_historical_data = train_models_xgboost.download_historical_data
calculate_indicators = train_models_xgboost.calculate_indicators
prepare_features = train_models_xgboost.prepare_features

from xgboost_features import calculate_all_xgboost_features
from feature_selection_v5 import SELECTED_FEATURES_V5, select_features_from_vector

# Global cache pour éviter de recharger les données à chaque essai
DATA_CACHE = {}

def prepare_data_for_optimization(crypto, timeframe_key):
    """Préparer les données une seule fois pour l'optimisation"""
    cache_key = f"{crypto['symbol']}_{timeframe_key}"

    if cache_key in DATA_CACHE:
        print(f"  [CACHE] Using cached data for {cache_key}")
        return DATA_CACHE[cache_key]

    tf_config = TIMEFRAME_CONFIGS[timeframe_key]

    print(f"  [>>] Preparing data for {crypto['name']} {timeframe_key}...")

    # Télécharger données
    klines = download_historical_data(
        crypto['symbol'],
        interval=tf_config['interval'],
        limit=tf_config['limit']
    )

    if not klines or len(klines) < 100:
        print(f"  [ERROR] Insufficient data")
        return None

    # Télécharger BTC si nécessaire
    klines_btc = None
    if crypto['symbol'] != 'BTCUSDT':
        klines_btc = download_historical_data(
            'BTCUSDT',
            interval=tf_config['interval'],
            limit=tf_config['limit']
        )

    # Préparer features et labels
    features_list = []
    labels_list = []
    indicators_history = []

    for i in range(200, len(klines) - tf_config['lookahead']):
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

        # Label
        future_price = float(klines[i + tf_config['lookahead']][4])
        change_pct = ((future_price - current_price) / current_price) * 100

        if change_pct > tf_config['buy_threshold']:
            label = 0  # BUY
        elif change_pct < tf_config['sell_threshold']:
            label = 1  # SELL
        else:
            label = 2  # HOLD

        features_list.append(features)
        labels_list.append(label)

    X = np.array(features_list)
    y = np.array(labels_list)

    # Extraire les prix pour la simulation de trading
    prices = [float(klines[i][4]) for i in range(200, len(klines) - tf_config['lookahead'])]

    # Split train/validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    prices_val = prices[split_idx:]

    data = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'prices_val': prices_val  # Prix pour simuler le trading
    }

    DATA_CACHE[cache_key] = data
    print(f"  [OK] Prepared {len(X)} samples with {X.shape[1]} features")

    return data

def objective(trial, crypto, timeframe_key, data):
    """Fonction objective pour Optuna - Optimise le RENDEMENT au lieu de l'accuracy"""

    # Hyperparamètres à optimiser (plages élargies)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 600),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 15.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 15.0),
        'scale_pos_weight': 1,
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 20,
        'verbose': 0
    }

    # Entraîner modèle
    model = xgb.XGBClassifier(**params)

    model.fit(
        data['X_train'],
        data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False
    )

    # Prédire sur validation
    y_pred = model.predict(data['X_val'])

    # Simuler le trading sur validation set pour calculer le rendement
    capital = 10000.0
    position = None  # None = flat, 'long' = long

    for i, pred in enumerate(y_pred):
        if pred == 0:  # BUY signal
            if position is None:
                position = 'long'
                entry_price = data['prices_val'][i]
        elif pred == 1:  # SELL signal
            if position == 'long':
                exit_price = data['prices_val'][i]
                pnl_pct = ((exit_price - entry_price) / entry_price)
                capital *= (1 + pnl_pct)
                position = None

    # Close position si toujours ouvert
    if position == 'long':
        exit_price = data['prices_val'][-1]
        pnl_pct = ((exit_price - entry_price) / entry_price)
        capital *= (1 + pnl_pct)

    # Retourner le rendement (en pourcentage)
    return_pct = ((capital - 10000) / 10000) * 100

    return return_pct

def optimize_hyperparameters(crypto, timeframe_key, n_trials=300):
    """Optimiser les hyperparamètres avec Optuna"""

    print(f"\n{'='*80}")
    print(f"OPTIMIZING: {crypto['name']} - {timeframe_key.upper()}")
    print(f"Running {n_trials} trials with Optuna...")
    print(f"Objective: Maximize TRADING RETURN (not accuracy)")
    print('='*80)

    # Préparer données
    data = prepare_data_for_optimization(crypto, timeframe_key)
    if data is None:
        return None

    # Créer étude Optuna
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name=f"{crypto['cryptoId']}_{timeframe_key}"
    )

    # Optimiser
    print(f"\n  [>>] Starting optimization ({n_trials} trials)...")
    study.optimize(
        lambda trial: objective(trial, crypto, timeframe_key, data),
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1
    )

    # Meilleurs paramètres
    best_params = study.best_params
    best_score = study.best_value

    print(f"\n  [OK] Optimization completed!")
    print(f"      Best validation return: {best_score:.2f}%")
    print(f"      Best parameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"         {param}: {value:.4f}")
        else:
            print(f"         {param}: {value}")

    return {
        'crypto': crypto['name'],
        'cryptoId': crypto['cryptoId'],
        'symbol': crypto['symbol'],
        'timeframe': timeframe_key,
        'best_params': best_params,
        'best_score': best_score,
        'n_trials': n_trials
    }

def train_with_best_params(crypto, timeframe_key, best_params):
    """Ré-entraîner le modèle avec les meilleurs paramètres"""

    print(f"\n  [>>] Retraining {crypto['name']} {timeframe_key} with best params...")

    # Récupérer données du cache
    cache_key = f"{crypto['symbol']}_{timeframe_key}"
    data = DATA_CACHE.get(cache_key)

    if data is None:
        print(f"  [ERROR] No cached data found")
        return None

    # Ajouter paramètres fixes
    params = best_params.copy()
    params.update({
        'scale_pos_weight': 1,
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 20,
        'verbose': 0
    })

    # Entraîner modèle final
    model = xgb.XGBClassifier(**params)

    model.fit(
        data['X_train'],
        data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False
    )

    # Évaluer
    train_acc = model.score(data['X_train'], data['y_train'])
    val_acc = model.score(data['X_val'], data['y_val'])

    print(f"  [OK] Train accuracy: {train_acc*100:.2f}%")
    print(f"  [OK] Validation accuracy: {val_acc*100:.2f}%")

    # Sauvegarder modèle optimisé
    model_dir = './models/xgboost'
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/{crypto['cryptoId']}_{timeframe_key}_xgboost.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"  [OK] Optimized model saved: {model_path}")

    return {
        'crypto': crypto['name'],
        'timeframe': timeframe_key,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'samples': len(data['X_train']) + len(data['X_val']),
        'model_path': model_path,
        'best_params': best_params
    }

def main():
    print("\n" + "="*80)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION WITH OPTUNA - ADVANCED")
    print("="*80)
    print("\nObjective: Maximize TRADING RETURN on validation set")
    print("Method: Bayesian optimization with median pruning")
    print("Trials per model: 300 (3x more than before)")
    print("Search space: Expanded hyperparameter ranges")
    print("="*80)

    # Sélectionner les modèles à optimiser
    # BTC 1d, BTC 1w, ETH 1d, SOL 1d
    models_to_optimize = [
        ('bitcoin', '1d'),
        ('bitcoin', '1w'),
        ('ethereum', '1d'),
        ('solana', '1d')
    ]

    optimization_results = []
    training_results = []

    # 1. Optimiser les hyperparamètres
    print("\n" + "="*80)
    print("PHASE 1: HYPERPARAMETER OPTIMIZATION")
    print("="*80)

    for crypto_id, timeframe in models_to_optimize:
        crypto = next(c for c in CRYPTOS if c['cryptoId'] == crypto_id)

        result = optimize_hyperparameters(crypto, timeframe, n_trials=300)
        if result:
            optimization_results.append(result)

    # Sauvegarder résultats d'optimisation
    with open('optimization_results.json', 'w') as f:
        json_results = []
        for r in optimization_results:
            json_r = r.copy()
            json_r['best_score'] = float(r['best_score'])
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)

    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)

    for result in optimization_results:
        print(f"\n{result['crypto']} - {result['timeframe'].upper()}:")
        print(f"   Best Return: {result['best_score']:.2f}%")
        print(f"   Trials: {result['n_trials']}")

    # 2. Ré-entraîner avec les meilleurs paramètres
    print("\n" + "="*80)
    print("PHASE 2: RETRAINING WITH OPTIMAL PARAMETERS")
    print("="*80)

    for opt_result in optimization_results:
        crypto = next(c for c in CRYPTOS if c['cryptoId'] == opt_result['cryptoId'])

        train_result = train_with_best_params(
            crypto,
            opt_result['timeframe'],
            opt_result['best_params']
        )

        if train_result:
            training_results.append(train_result)

    # Sauvegarder résultats d'entraînement
    with open('training_xgboost_results.json', 'w') as f:
        json_results = []
        for r in training_results:
            json_r = r.copy()
            json_r['train_accuracy'] = float(r['train_accuracy'])
            json_r['val_accuracy'] = float(r['val_accuracy'])
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)

    # Résumé final
    print("\n" + "="*80)
    print("FINAL TRAINING SUMMARY")
    print("="*80)

    for result in training_results:
        print(f"\n{result['crypto']} - {result['timeframe'].upper()}:")
        print(f"   Train Acc: {result['train_accuracy']*100:.2f}%")
        print(f"   Val Acc:   {result['val_accuracy']*100:.2f}%")
        print(f"   Samples:   {result['samples']}")

    print(f"\n[OK] {len(training_results)} optimized models trained successfully!")
    print("[OK] Optimization results saved to: optimization_results.json")
    print("[OK] Training results saved to: training_xgboost_results.json")
    print("="*80)

if __name__ == '__main__':
    main()
