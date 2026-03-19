"""
Backtest ALL 9 Cryptos - Hybrid V5/V6 System
==============================================

V6 Models (8): Bitcoin, Ethereum, BNB, XRP, Cardano, Avalanche, Polkadot, Polygon
V5 Models (1): Solana

Test Period: January 1, 2025 - March 18, 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))

import pickle
import numpy as np
from datetime import datetime
from data_manager import get_historical_data
from train_models_xgboost import download_historical_data, calculate_indicators, prepare_features
from xgboost_features import calculate_all_xgboost_features
from xgboost_features_v6 import calculate_all_xgboost_features_v6
from feature_selection_v5 import SELECTED_FEATURES_V5

# ALL 9 CRYPTOS - ALL V6 Models
ALL_CRYPTOS = [
    # V6 Models
    {'cryptoId': 'bitcoin', 'symbol': 'BTCUSDT', 'name': 'Bitcoin', 'version': 'v6'},
    {'cryptoId': 'ethereum', 'symbol': 'ETHUSDT', 'name': 'Ethereum', 'version': 'v6'},
    {'cryptoId': 'bnb', 'symbol': 'BNBUSDT', 'name': 'BNB', 'version': 'v6'},
    {'cryptoId': 'xrp', 'symbol': 'XRPUSDT', 'name': 'XRP', 'version': 'v6'},
    {'cryptoId': 'cardano', 'symbol': 'ADAUSDT', 'name': 'Cardano', 'version': 'v6'},
    {'cryptoId': 'avalanche', 'symbol': 'AVAXUSDT', 'name': 'Avalanche', 'version': 'v6'},
    {'cryptoId': 'polkadot', 'symbol': 'DOTUSDT', 'name': 'Polkadot', 'version': 'v6'},
    {'cryptoId': 'polygon', 'symbol': 'MATICUSDT', 'name': 'Polygon', 'version': 'v6'},
    {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana', 'version': 'v6'}
]

TIMEFRAME = '1d'
INITIAL_CAPITAL = 1000  # Changed from 10000 to 1000
CUTOFF_DATE = datetime(2025, 1, 1)

def select_features_from_vector(features_69, selected_features_names):
    """Extract selected features from full 69-feature vector"""
    return features_69[:41]

def prepare_features_v6(crypto, klines, klines_btc=None):
    """Prepare V6 features (41 V5 + 16 V6 = 57 features)"""
    features_list = []
    prices_list = []
    dates_list = []
    indicators_history = []

    for i in range(200, len(klines)):
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

        # Combine: base + V5 + V6
        features_69_v5 = features_base + features_xgb_v5
        features_selected_v5 = select_features_from_vector(features_69_v5, SELECTED_FEATURES_V5)
        features_all_v6 = features_selected_v5 + features_xgb_v6

        features_list.append(features_all_v6)
        prices_list.append(current_price)

        timestamp = int(klines[i][0]) / 1000
        dates_list.append(datetime.fromtimestamp(timestamp))

    return np.array(features_list), np.array(prices_list), dates_list

def prepare_features_v5(crypto, klines, klines_btc=None):
    """Prepare V5 features (41 selected features)"""
    features_list = []
    prices_list = []
    dates_list = []
    indicators_history = []

    for i in range(200, len(klines)):
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

        # V5 features (69)
        features_xgb = calculate_all_xgboost_features(
            window_data,
            indicators,
            volumes,
            crypto_symbol=crypto['symbol'],
            klines_btc=window_data_btc,
            indicators_history=indicators_history
        )

        # Combine base + V5
        features_69 = features_base + features_xgb
        features_selected = select_features_from_vector(features_69, SELECTED_FEATURES_V5)

        features_list.append(features_selected)
        prices_list.append(current_price)

        timestamp = int(klines[i][0]) / 1000
        dates_list.append(datetime.fromtimestamp(timestamp))

    return np.array(features_list), np.array(prices_list), dates_list

def backtest_crypto(crypto):
    """Backtest one crypto with appropriate model version"""
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {crypto['name']} ({crypto['version'].upper()})")
    print(f"{'='*80}")

    # Load model
    if crypto['version'] == 'v6':
        model_path = f"./training/models/xgboost_v6/{crypto['cryptoId']}_1d_xgboost_v6.pkl"
    else:  # v5
        model_path = f"./training/models/xgboost/{crypto['cryptoId']}_1d_xgboost.pkl"

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"[OK] Loaded model: {model_path}")

    # Download data
    klines = download_historical_data(crypto['symbol'], interval='1d', limit=3000)
    if not klines:
        print(f"[ERROR] No data for {crypto['symbol']}")
        return None

    klines_btc = None
    if crypto['symbol'] != 'BTCUSDT':
        klines_btc = download_historical_data('BTCUSDT', interval='1d', limit=3000)

    # Prepare features based on version
    if crypto['version'] == 'v6':
        X, prices, dates = prepare_features_v6(crypto, klines, klines_btc)
    else:  # v5
        X, prices, dates = prepare_features_v5(crypto, klines, klines_btc)

    # Split test data (2025+)
    test_indices = [i for i, date in enumerate(dates) if date >= CUTOFF_DATE]

    if len(test_indices) == 0:
        print("[ERROR] No test data available")
        return None

    X_test = X[test_indices]
    prices_test = prices[test_indices]
    dates_test = [dates[i] for i in test_indices]

    print(f"[OK] Test period: {dates_test[0].date()} to {dates_test[-1].date()}")
    print(f"[OK] Test samples: {len(X_test)}")

    # Predict
    predictions = model.predict(X_test)

    # Backtest
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = prices_test[i]
        date = dates_test[i]

        # BUY signal
        if pred == 0 and position is None:
            position = {'entry_price': price, 'entry_date': date, 'type': 'LONG'}

        # SELL signal or end
        elif (pred == 1 or i == len(predictions) - 1) and position is not None:
            exit_price = price
            pnl = ((exit_price - position['entry_price']) / position['entry_price']) * capital
            capital += pnl

            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': date,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'return_pct': ((exit_price - position['entry_price']) / position['entry_price']) * 100
            })

            position = None

    # Calculate metrics
    final_capital = capital
    total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    # Buy & Hold
    buy_hold_return = ((prices_test[-1] - prices_test[0]) / prices_test[0]) * 100

    # Test accuracy
    y_test_labels = []
    for i in range(len(X_test) - 7):
        future_price = prices_test[i + 7]
        current_price = prices_test[i]
        change_pct = ((future_price - current_price) / current_price) * 100

        if change_pct > 2.0:
            y_test_labels.append(0)  # BUY
        elif change_pct < -2.0:
            y_test_labels.append(1)  # SELL
        else:
            y_test_labels.append(2)  # HOLD

    test_accuracy = np.mean(predictions[:len(y_test_labels)] == np.array(y_test_labels))

    print(f"\n{'='*80}")
    print(f"RESULTS: {crypto['name']}")
    print(f"{'='*80}")
    print(f"Test Accuracy:        {test_accuracy*100:.2f}%")
    print(f"Final Capital:        ${final_capital:.2f}")
    print(f"Total Return:         {total_return:+.2f}%")
    print(f"Buy & Hold:           {buy_hold_return:+.2f}%")
    print(f"Outperformance:       {total_return - buy_hold_return:+.2f}%")
    print(f"Number of Trades:     {len(trades)}")
    print(f"{'='*80}")

    return {
        'crypto': crypto['name'],
        'version': crypto['version'],
        'timeframe': TIMEFRAME,
        'test_accuracy': float(test_accuracy),
        'final_capital': float(final_capital),
        'total_return_pct': float(total_return),
        'buy_hold_return_pct': float(buy_hold_return),
        'outperformance_pct': float(total_return - buy_hold_return),
        'num_trades': len(trades),
        'test_samples': len(X_test)
    }

def main():
    print("\n" + "="*80)
    print("BACKTEST ALL CRYPTOS - HYBRID V5/V6 SYSTEM")
    print("="*80)
    print(f"Test Period: {CUTOFF_DATE.date()} onwards")
    print(f"Initial Capital: ${INITIAL_CAPITAL}")
    print(f"Total Cryptos: {len(ALL_CRYPTOS)}")
    print("="*80)

    results = []

    for crypto in ALL_CRYPTOS:
        result = backtest_crypto(crypto)
        if result:
            results.append(result)

    # Save results
    import json
    with open('backtest_all_cryptos_v6_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - ALL CRYPTOS")
    print(f"{'='*80}")
    print(f"{'Crypto':<12} {'Version':<8} {'Accuracy':<10} {'Return':<12} {'B&H':<12} {'Outperf':<12}")
    print(f"{'-'*80}")

    for r in results:
        print(f"{r['crypto']:<12} {r['version'].upper():<8} {r['test_accuracy']*100:>8.2f}% {r['total_return_pct']:>10.2f}% {r['buy_hold_return_pct']:>10.2f}% {r['outperformance_pct']:>10.2f}%")

    print(f"{'='*80}")
    print(f"[OK] Results saved to: backtest_all_cryptos_v6_results.json")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
