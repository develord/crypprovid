"""
Backtest Real-World Test 2025-2026
Train on all data until Dec 31, 2024
Test on Jan 1, 2025 - Today (Mar 18, 2026)

This provides a true out-of-sample test on completely unseen data
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
import time

# Add training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

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

from data_manager import get_historical_data
from train_models_xgboost import download_historical_data, calculate_indicators, prepare_features
from xgboost_features import calculate_all_xgboost_features
from feature_selection_v5 import SELECTED_FEATURES_V5, select_features_from_vector
import xgboost as xgb

# Configuration
CRYPTOS = [
    {'cryptoId': 'bitcoin', 'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
    {'cryptoId': 'ethereum', 'symbol': 'ETHUSDT', 'name': 'Ethereum'},
    {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana'}
]

TIMEFRAME_CONFIGS = {
    '1d': {
        'interval': '1d',
        'limit': 3000,  # Get all available data
        'lookahead': 7,
        'buy_threshold': 2.0,
        'sell_threshold': -2.0
    },
    '1w': {
        'interval': '1w',
        'limit': 1000,
        'lookahead': 4,
        'buy_threshold': 3.5,
        'sell_threshold': -3.5
    }
}

# Cut-off date for training/testing split
TRAIN_END_DATE = datetime(2024, 12, 31, 23, 59, 59)  # Train until end of 2024
TEST_START_DATE = datetime(2025, 1, 1, 0, 0, 0)      # Test from Jan 1, 2025

def timestamp_to_datetime(timestamp_ms):
    """Convert millisecond timestamp to datetime"""
    return datetime.fromtimestamp(timestamp_ms / 1000)

def split_data_by_date(klines, cutoff_date):
    """Split klines into training and testing based on date"""
    train_klines = []
    test_klines = []

    for kline in klines:
        kline_date = timestamp_to_datetime(kline[0])
        if kline_date <= cutoff_date:
            train_klines.append(kline)
        else:
            test_klines.append(kline)

    return train_klines, test_klines

def prepare_data_with_date_split(crypto, timeframe_key):
    """
    Prepare training and testing data with date-based split
    Train: all data until Dec 31, 2024
    Test: Jan 1, 2025 - Today
    """
    tf_config = TIMEFRAME_CONFIGS[timeframe_key]

    print(f"\n[>>] Downloading data for {crypto['name']} {timeframe_key}...")

    # Download all available historical data
    klines = download_historical_data(
        crypto['symbol'],
        interval=tf_config['interval'],
        limit=tf_config['limit']
    )

    if not klines or len(klines) < 300:
        print(f"[ERROR] Insufficient data: {len(klines)} candles")
        return None

    print(f"[OK] Downloaded {len(klines)} candles")

    # Split data by date
    train_klines, test_klines = split_data_by_date(klines, TRAIN_END_DATE)

    print(f"[OK] Training data: {len(train_klines)} candles (until Dec 31, 2024)")
    print(f"[OK] Testing data: {len(test_klines)} candles (Jan 1, 2025 - Today)")

    if len(train_klines) < 200:
        print(f"[ERROR] Insufficient training data")
        return None

    if len(test_klines) < 10:
        print(f"[ERROR] Insufficient testing data")
        return None

    # Download BTC data if needed
    klines_btc = None
    klines_btc_train = None
    klines_btc_test = None

    if crypto['symbol'] != 'BTCUSDT':
        klines_btc = download_historical_data(
            'BTCUSDT',
            interval=tf_config['interval'],
            limit=tf_config['limit']
        )
        if klines_btc:
            klines_btc_train, klines_btc_test = split_data_by_date(klines_btc, TRAIN_END_DATE)

    # Prepare TRAINING features
    print(f"[>>] Preparing TRAINING features...")
    X_train, y_train = prepare_features_and_labels(
        train_klines,
        klines_btc_train,
        crypto,
        tf_config
    )

    # Prepare TESTING features
    print(f"[>>] Preparing TESTING features...")
    # Combine train + test data for feature calculation (need historical context)
    all_klines = train_klines + test_klines
    all_klines_btc = None
    if klines_btc_train and klines_btc_test:
        all_klines_btc = klines_btc_train + klines_btc_test

    # Calculate features for all data, but only extract test period labels
    X_all, y_all, dates_all = prepare_features_and_labels_with_dates(
        all_klines,
        all_klines_btc,
        crypto,
        tf_config
    )

    # Extract only test period
    # Features start at index 200 in klines, so we need to offset correctly
    train_features_count = len(X_train)
    X_test = X_all[train_features_count:]
    y_test = y_all[train_features_count:]
    test_dates = dates_all[train_features_count:]

    # Extract prices for test period - these correspond to the current prices when predictions are made
    test_prices = []
    for i in range(200 + train_features_count, len(all_klines) - tf_config['lookahead']):
        test_prices.append(float(all_klines[i][4]))

    print(f"[OK] Train: {len(X_train)} samples")
    print(f"[OK] Test: {len(X_test)} samples")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'test_dates': test_dates,
        'test_prices': test_prices,
        'crypto': crypto,
        'timeframe': timeframe_key
    }

def prepare_features_and_labels(klines, klines_btc, crypto, tf_config):
    """Prepare features and labels from klines"""
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

        # Calculate label
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

    return np.array(features_list), np.array(labels_list)

def prepare_features_and_labels_with_dates(klines, klines_btc, crypto, tf_config):
    """Prepare features, labels and dates"""
    features_list = []
    labels_list = []
    dates_list = []
    indicators_history = []

    for i in range(200, len(klines) - tf_config['lookahead']):
        window_data = klines[max(0, i-200):i+1]
        indicators = calculate_indicators(window_data)
        current_price = float(klines[i][4])
        current_date = timestamp_to_datetime(klines[i][0])

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

        # Calculate label
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
        dates_list.append(current_date)

    return np.array(features_list), np.array(labels_list), dates_list

def simulate_trading(predictions, actual_labels, prices, dates, initial_capital=10000):
    """Simulate trading based on predictions"""
    capital = initial_capital
    position = None  # None, 'long', or 'short'
    entry_price = 0
    trades = []
    portfolio_values = [initial_capital]

    for i in range(len(predictions)):
        pred = predictions[i]
        current_price = prices[i]

        # Close position logic
        if position == 'long' and pred == 1:  # Sell signal while long
            pnl = (current_price - entry_price) / entry_price
            capital *= (1 + pnl)
            trades.append({
                'date': dates[i],
                'type': 'close_long',
                'price': current_price,
                'pnl_pct': pnl * 100
            })
            position = None

        # Open position logic
        if position is None:
            if pred == 0:  # Buy signal
                position = 'long'
                entry_price = current_price
                trades.append({
                    'date': dates[i],
                    'type': 'open_long',
                    'price': current_price,
                    'pnl_pct': 0
                })

        # Calculate current portfolio value
        if position == 'long':
            current_value = capital * (1 + (current_price - entry_price) / entry_price)
        else:
            current_value = capital

        portfolio_values.append(current_value)

    # Close any open position at the end
    if position == 'long':
        final_price = prices[-1]
        pnl = (final_price - entry_price) / entry_price
        capital *= (1 + pnl)
        trades.append({
            'date': dates[-1],
            'type': 'close_long',
            'price': final_price,
            'pnl_pct': pnl * 100
        })

    return {
        'final_capital': capital,
        'total_return_pct': ((capital - initial_capital) / initial_capital) * 100,
        'num_trades': len([t for t in trades if t['type'].startswith('open')]),
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def run_backtest(crypto, timeframe_key, best_params):
    """Run backtest for a specific crypto and timeframe"""

    print(f"\n{'='*80}")
    print(f"BACKTEST: {crypto['name']} - {timeframe_key.upper()}")
    print(f"Train Period: Beginning - Dec 31, 2024")
    print(f"Test Period: Jan 1, 2025 - Mar 18, 2026")
    print('='*80)

    print(f"[OK] Using Optuna-optimized hyperparameters")

    # Prepare data
    data = prepare_data_with_date_split(crypto, timeframe_key)
    if data is None:
        return None

    # Train model with optimized hyperparameters on PRE-2025 data ONLY
    print(f"\n[>>] Training model on pre-2025 data ({len(data['X_train'])} samples)...")

    # Add fixed parameters to best_params
    params = best_params.copy()
    params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42
    })

    model = xgb.XGBClassifier(**params)
    model.fit(
        data['X_train'],
        data['y_train']
    )

    train_acc = model.score(data['X_train'], data['y_train'])
    print(f"[OK] Model trained - Training accuracy: {train_acc*100:.2f}%")

    # Make predictions on test set
    print(f"\n[>>] Running predictions on test period...")
    y_pred = model.predict(data['X_test'])
    y_pred_proba = model.predict_proba(data['X_test'])

    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    test_accuracy = accuracy_score(data['y_test'], y_pred)

    print(f"\n[METRICS]")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(data['y_test'], y_pred,
                                target_names=['BUY', 'SELL', 'HOLD']))

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(data['y_test'], y_pred))

    # Simulate trading
    print(f"\n[>>] Simulating trading strategy...")
    trading_results = simulate_trading(
        y_pred,
        data['y_test'],
        data['test_prices'],
        data['test_dates']
    )

    print(f"\n[TRADING RESULTS]")
    print(f"Initial Capital: $10,000")
    print(f"Final Capital: ${trading_results['final_capital']:.2f}")
    print(f"Total Return: {trading_results['total_return_pct']:.2f}%")
    print(f"Number of Trades: {trading_results['num_trades']}")

    # Calculate buy & hold
    first_price = data['test_prices'][0]
    last_price = data['test_prices'][-1]
    buy_hold_return = ((last_price - first_price) / first_price) * 100
    print(f"\nBuy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Strategy vs Buy&Hold: {trading_results['total_return_pct'] - buy_hold_return:+.2f}%")

    return {
        'crypto': crypto['name'],
        'timeframe': timeframe_key,
        'test_accuracy': test_accuracy,
        'final_capital': trading_results['final_capital'],
        'total_return_pct': trading_results['total_return_pct'],
        'buy_hold_return_pct': buy_hold_return,
        'outperformance_pct': trading_results['total_return_pct'] - buy_hold_return,
        'num_trades': trading_results['num_trades'],
        'test_samples': len(data['X_test'])
    }

def main():
    print("\n" + "="*80)
    print("REAL-WORLD BACKTEST 2025-2026")
    print("="*80)
    print(f"Training Period: Beginning - Dec 31, 2024")
    print(f"Testing Period: Jan 1, 2025 - Mar 18, 2026 ({(datetime.now() - TEST_START_DATE).days} days)")
    print(f"Models: Optuna-optimized XGBoost (trained on PRE-2025 data ONLY)")
    print("="*80)

    # Load Optuna optimization results
    optuna_results_path = './training/optimization_results.json'
    if not os.path.exists(optuna_results_path):
        print(f"[ERROR] Optuna results not found: {optuna_results_path}")
        return

    with open(optuna_results_path, 'r') as f:
        optuna_results = json.load(f)

    print(f"[OK] Loaded {len(optuna_results)} optimized hyperparameter sets")

    results = []

    # Run backtest for each optimized model
    for opt_result in optuna_results:
        crypto_id = opt_result['cryptoId']
        timeframe = opt_result['timeframe']
        best_params = opt_result['best_params']

        # Skip Bitcoin 1w
        if crypto_id == 'bitcoin' and timeframe == '1w':
            print(f"\n[SKIP] Skipping Bitcoin 1w")
            continue

        crypto = next(c for c in CRYPTOS if c['cryptoId'] == crypto_id)

        result = run_backtest(crypto, timeframe, best_params)
        if result:
            results.append(result)

        time.sleep(2)  # Rate limiting

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - REAL WORLD TEST")
    print("="*80)

    for result in results:
        print(f"\n{result['crypto']} - {result['timeframe'].upper()}:")
        print(f"   Test Accuracy: {result['test_accuracy']*100:.2f}%")
        print(f"   Strategy Return: {result['total_return_pct']:+.2f}%")
        print(f"   Buy&Hold Return: {result['buy_hold_return_pct']:+.2f}%")
        print(f"   Outperformance: {result['outperformance_pct']:+.2f}%")
        print(f"   Trades: {result['num_trades']}")

    # Save results
    with open('realworld_backtest_2025_results.json', 'w') as f:
        json_results = []
        for r in results:
            json_r = r.copy()
            json_r['test_accuracy'] = float(r['test_accuracy'])
            json_r['final_capital'] = float(r['final_capital'])
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)

    print(f"\n[OK] Results saved to: realworld_backtest_2025_results.json")
    print("="*80)

if __name__ == '__main__':
    main()
