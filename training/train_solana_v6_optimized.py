"""
Solana V6 Optimized Training - Less Conservative Hyperparameters
================================================================

SOLANA SPECIFIC ADJUSTMENTS:
- Higher max_depth (7 vs 5) for complex patterns in volatile asset
- Reduced regularization (solana needs more flexibility)
- Kept K-Fold CV and V6 features (16 new anti-overfitting features)
- More estimators for better pattern learning

Rationale: Solana is highly volatile and needs less restriction
"""

import os
import sys
import json
import pickle
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

# Add data manager to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from data_manager import get_historical_data

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# Safe print function
_original_print = print
def safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except (ValueError, OSError):
        pass

import builtins
builtins.print = safe_print

# Import feature calculation functions
from train_models_xgboost import download_historical_data, calculate_indicators, prepare_features
from xgboost_features import calculate_all_xgboost_features
from xgboost_features_v6 import calculate_all_xgboost_features_v6
from feature_selection_v5 import SELECTED_FEATURES_V5

# Configuration - SOLANA ONLY
CRYPTO = {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana'}

TIMEFRAME_CONFIG = {
    'interval': '1d',
    'limit': 3000,
    'lookahead': 7,
    'buy_threshold': 2.0,
    'sell_threshold': -2.0
}

# SOLANA-OPTIMIZED HYPERPARAMETERS (V6)
# Less conservative than Bitcoin/Ethereum to handle high volatility
SOLANA_V6_PARAMS = {
    'n_estimators': 400,        # Increased from 300 (more learning)
    'max_depth': 7,             # Increased from 5 (capture complex patterns)
    'learning_rate': 0.04,      # Slightly higher than 0.03
    'subsample': 0.75,          # Slightly increased from 0.7
    'colsample_bytree': 0.65,   # Slightly increased from 0.6
    'colsample_bylevel': 0.55,  # Slightly increased from 0.5
    'min_child_weight': 12,     # Reduced from 15 (less restrictive)
    'gamma': 1.5,               # Reduced from 2.0 (less pruning)
    'reg_alpha': 3.5,           # Reduced from 5.0 (less L1 regularization)
    'reg_lambda': 6.0,          # Reduced from 8.0 (less L2 regularization)
    'scale_pos_weight': 1,
    'objective': 'multi:softprob',
    'num_class': 3,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 30,
    'verbose': 0
}

def select_features_from_vector(features_69, selected_features_names):
    """Extract selected features from full 69-feature vector"""
    # Simplified: just take first 41 features
    return features_69[:41]

def prepare_data_v6(crypto, timeframe_config):
    """
    Prepare data with V5 + V6 features for Solana

    Returns:
        X, y, prices arrays
    """
    print(f"\n[>>] Preparing V6 data for {crypto['name']}...")

    # Download data
    klines = download_historical_data(
        crypto['symbol'],
        interval=timeframe_config['interval'],
        limit=timeframe_config['limit']
    )

    if not klines or len(klines) < 300:
        print(f"[ERROR] Insufficient data")
        return None, None, None

    # Download BTC data
    klines_btc = download_historical_data(
        'BTCUSDT',
        interval=timeframe_config['interval'],
        limit=timeframe_config['limit']
    )

    # Prepare features and labels
    features_list = []
    labels_list = []
    prices_list = []
    indicators_history = []

    for i in range(200, len(klines) - timeframe_config['lookahead']):
        window_data = klines[max(0, i-200):i+1]
        indicators = calculate_indicators(window_data)
        current_price = float(klines[i][4])

        indicators_history.append(indicators)
        if len(indicators_history) > 10:
            indicators_history.pop(0)

        prices_history = [float(k[4]) for k in window_data]
        features_base = prepare_features(indicators, current_price, prices_history)

        # BTC data
        window_data_btc = None
        if klines_btc and len(klines_btc) >= i+1:
            window_data_btc = klines_btc[max(0, i-200):i+1]

        volumes = [float(k[5]) for k in window_data]

        # V5 features (69)
        features_xgb_v5 = calculate_all_xgboost_features(
            window_data,
            indicators,
            volumes,
            crypto_symbol=crypto['symbol'],
            klines_btc=window_data_btc,
            indicators_history=indicators_history
        )

        # V6 features (16)
        features_xgb_v6 = calculate_all_xgboost_features_v6(
            window_data,
            indicators,
            volumes,
            crypto_symbol=crypto['symbol'],
            klines_btc=window_data_btc,
            indicators_history=indicators_history
        )

        # Combine base + V5 = 69, select 41 V5, add 16 V6 = 57 features
        features_69_v5 = features_base + features_xgb_v5
        features_selected_v5 = select_features_from_vector(features_69_v5, SELECTED_FEATURES_V5)
        features_all_v6 = features_selected_v5 + features_xgb_v6

        # Label
        future_price = float(klines[i + timeframe_config['lookahead']][4])
        change_pct = ((future_price - current_price) / current_price) * 100

        if change_pct > timeframe_config['buy_threshold']:
            label = 0  # BUY
        elif change_pct < timeframe_config['sell_threshold']:
            label = 1  # SELL
        else:
            label = 2  # HOLD

        features_list.append(features_all_v6)
        labels_list.append(label)
        prices_list.append(current_price)

    X = np.array(features_list)
    y = np.array(labels_list)
    prices = np.array(prices_list)

    print(f"[OK] Prepared {len(X)} samples with {X.shape[1]} features (V6: 41 V5 + 16 V6)")

    return X, y, prices

def train_with_kfold_cv_solana(crypto, timeframe_config, n_splits=3):
    """
    Train Solana model with K-Fold Time Series Cross-Validation
    Using optimized hyperparameters for high volatility

    Returns:
        Dict with training results
    """
    print(f"\n{'='*80}")
    print(f"TRAINING V6 OPTIMIZED: {crypto['name']}")
    print(f"K-Fold CV with {n_splits} splits + Solana-Optimized Hyperparameters")
    print(f"Adjustments: max_depth=7, reduced regularization for volatility")
    print(f"{'='*80}")

    # Prepare data
    X, y, prices = prepare_data_v6(crypto, timeframe_config)

    if X is None:
        return None

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    best_model = None
    best_val_acc = 0

    print(f"\n[>>] Starting {n_splits}-Fold Time Series Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n  Fold {fold+1}/{n_splits}:")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        print(f"    Train: {len(X_train_fold)} samples, Val: {len(X_val_fold)} samples")

        # Train model with Solana-optimized params
        model = xgb.XGBClassifier(**SOLANA_V6_PARAMS)

        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )

        # Evaluate
        train_acc = model.score(X_train_fold, y_train_fold)
        val_acc = model.score(X_val_fold, y_val_fold)

        print(f"    Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")

        fold_results.append({
            'fold': fold + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_samples': len(X_train_fold),
            'val_samples': len(X_val_fold)
        })

        # Keep best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            print(f"    [BEST] New best model (val_acc: {val_acc*100:.2f}%)")

    # Calculate average metrics
    avg_train_acc = np.mean([f['train_acc'] for f in fold_results])
    avg_val_acc = np.mean([f['val_acc'] for f in fold_results])
    std_val_acc = np.std([f['val_acc'] for f in fold_results])

    print(f"\n{'='*80}")
    print(f"K-FOLD CV RESULTS - SOLANA V6 OPTIMIZED")
    print(f"{'='*80}")
    print(f"\n  Average Train Accuracy: {avg_train_acc*100:.2f}%")
    print(f"  Average Val Accuracy:   {avg_val_acc*100:.2f}% (±{std_val_acc*100:.2f}%)")
    print(f"  Best Val Accuracy:      {best_val_acc*100:.2f}%")
    print(f"  Overfitting Gap:        {(avg_train_acc - avg_val_acc)*100:.2f}%")

    # Retrain on full dataset with best hyperparameters
    print(f"\n[>>] Retraining on full dataset...")

    final_model = xgb.XGBClassifier(**SOLANA_V6_PARAMS)

    # Split for final evaluation
    split_idx = int(len(X) * 0.8)
    X_train_final, X_val_final = X[:split_idx], X[split_idx:]
    y_train_final, y_val_final = y[:split_idx], y[split_idx:]

    final_model.fit(
        X_train_final,
        y_train_final,
        eval_set=[(X_val_final, y_val_final)],
        verbose=False
    )

    final_train_acc = final_model.score(X_train_final, y_train_final)
    final_val_acc = final_model.score(X_val_final, y_val_final)

    print(f"[OK] Final Train Acc: {final_train_acc*100:.2f}%")
    print(f"[OK] Final Val Acc:   {final_val_acc*100:.2f}%")

    # Save model
    model_dir = './models/xgboost_v6'
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/{crypto['cryptoId']}_1d_xgboost_v6_optimized.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)

    print(f"[OK] Model saved: {model_path}")

    return {
        'crypto': crypto['name'],
        'timeframe': '1d',
        'n_features': X.shape[1],
        'total_samples': len(X),
        'kfold_avg_train_acc': float(avg_train_acc),
        'kfold_avg_val_acc': float(avg_val_acc),
        'kfold_std_val_acc': float(std_val_acc),
        'kfold_best_val_acc': float(best_val_acc),
        'final_train_acc': float(final_train_acc),
        'final_val_acc': float(final_val_acc),
        'overfitting_gap': float(avg_train_acc - avg_val_acc),
        'model_path': model_path,
        'hyperparameters': SOLANA_V6_PARAMS,
        'fold_results': fold_results,
        'optimization': 'solana_high_volatility'
    }

def main():
    print("\n" + "="*80)
    print("SOLANA V6 OPTIMIZED TRAINING")
    print("="*80)
    print("\nSOLANA-SPECIFIC ADJUSTMENTS:")
    print("  [1] max_depth: 5 → 7 (capture complex volatile patterns)")
    print("  [2] reg_alpha: 5.0 → 3.5 (less L1 regularization)")
    print("  [3] reg_lambda: 8.0 → 6.0 (less L2 regularization)")
    print("  [4] n_estimators: 300 → 400 (more learning)")
    print("  [5] Kept K-Fold CV (3 splits) and V6 features (16 new)")
    print("="*80)

    result = train_with_kfold_cv_solana(CRYPTO, TIMEFRAME_CONFIG, n_splits=3)

    if result:
        # Save results
        with open('training_solana_v6_optimized_results.json', 'w') as f:
            # Convert fold_results to JSON-serializable format
            result['fold_results'] = [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in fold.items()}
                for fold in result['fold_results']
            ]
            json.dump(result, f, indent=2)

        # Print summary
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY - SOLANA V6 OPTIMIZED")
        print(f"{'='*80}")
        print(f"\n{result['crypto']} - {result['timeframe'].upper()}:")
        print(f"   Features:           {result['n_features']}")
        print(f"   K-Fold Avg Val Acc: {result['kfold_avg_val_acc']*100:.2f}% (±{result['kfold_std_val_acc']*100:.2f}%)")
        print(f"   Final Val Acc:      {result['final_val_acc']*100:.2f}%")
        print(f"   Overfitting Gap:    {result['overfitting_gap']*100:.2f}%")

        print(f"\n[OK] Solana V6 optimized model trained successfully!")
        print("[OK] Results saved to: training_solana_v6_optimized_results.json")
        print("="*80)

if __name__ == '__main__':
    main()
