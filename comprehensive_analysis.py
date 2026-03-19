"""
Comprehensive XGBoost Analysis - 2025-2026 Test Period
Detailed analysis including:
- Performance metrics (Sharpe, Sortino, Calmar, Win Rate)
- Trade-by-trade analysis
- Confusion matrices
- Market condition analysis (bull/bear/sideways)
- Feature importance
- Error analysis
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

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

TIMEFRAME = '1d'
TIMEFRAME_CONFIG = {
    'interval': '1d',
    'limit': 3000,
    'lookahead': 7,
    'buy_threshold': 2.0,
    'sell_threshold': -2.0
}

TRAIN_END_DATE = datetime(2024, 12, 31, 23, 59, 59)
TEST_START_DATE = datetime(2025, 1, 1, 0, 0, 0)

def timestamp_to_datetime(timestamp_ms):
    """Convert millisecond timestamp to datetime"""
    return datetime.fromtimestamp(timestamp_ms / 1000)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Calculate annualized Sortino ratio"""
    if len(returns) == 0:
        return 0
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0
    downside_std = np.std(downside_returns)
    return np.sqrt(252) * (np.mean(excess_returns) / downside_std)

def calculate_max_drawdown(capital_history):
    """Calculate maximum drawdown"""
    if len(capital_history) == 0:
        return 0
    peak = capital_history[0]
    max_dd = 0
    for capital in capital_history:
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100

def calculate_calmar_ratio(returns, max_drawdown):
    """Calculate Calmar ratio (annual return / max drawdown)"""
    if max_drawdown == 0:
        return 0
    annual_return = np.mean(returns) * 252  # Annualized
    return annual_return / (max_drawdown / 100)

def detect_market_regime(prices, window=20):
    """Detect market regime: bull/bear/sideways"""
    if len(prices) < window:
        return 'sideways'

    sma = np.mean(prices[-window:])
    current = prices[-1]

    # Calculate trend strength
    slope = (prices[-1] - prices[-window]) / window
    volatility = np.std(prices[-window:])

    # Normalize slope by volatility
    normalized_slope = slope / (volatility + 1e-10)

    if normalized_slope > 0.5:
        return 'bull'
    elif normalized_slope < -0.5:
        return 'bear'
    else:
        return 'sideways'

def backtest_with_analysis(crypto, model):
    """Run backtest and collect detailed analysis data"""

    print(f"\n{'='*80}")
    print(f"ANALYZING: {crypto['name']} - {TIMEFRAME}")
    print(f"{'='*80}")

    # Download data
    print(f"\n[>>] Downloading historical data...")
    klines = download_historical_data(
        crypto['symbol'],
        interval=TIMEFRAME_CONFIG['interval'],
        limit=TIMEFRAME_CONFIG['limit']
    )

    if not klines or len(klines) < 300:
        print(f"[ERROR] Insufficient data")
        return None

    # Download BTC data if needed
    klines_btc = None
    if crypto['symbol'] != 'BTCUSDT':
        klines_btc = download_historical_data(
            'BTCUSDT',
            interval=TIMEFRAME_CONFIG['interval'],
            limit=TIMEFRAME_CONFIG['limit']
        )

    # Split data by date
    train_klines = []
    test_klines = []

    for kline in klines:
        kline_date = timestamp_to_datetime(kline[0])
        if kline_date <= TRAIN_END_DATE:
            train_klines.append(kline)
        elif kline_date >= TEST_START_DATE:
            test_klines.append(kline)

    print(f"[OK] Train samples: {len(train_klines)}, Test samples: {len(test_klines)}")

    # Prepare test features
    print(f"\n[>>] Preparing test features...")
    test_features = []
    test_labels = []
    test_prices = []
    test_dates = []
    test_market_regimes = []
    indicators_history = []

    # Need context from training data
    context_start = max(0, len(train_klines) - 200)
    context_klines = train_klines[context_start:] + test_klines

    for i in range(len(train_klines) - context_start, len(context_klines) - TIMEFRAME_CONFIG['lookahead']):
        window_data = context_klines[max(0, i-200):i+1]
        indicators = calculate_indicators(window_data)
        current_price = float(context_klines[i][4])

        indicators_history.append(indicators)
        if len(indicators_history) > 10:
            indicators_history.pop(0)

        prices_history = [float(k[4]) for k in window_data]
        features_base = prepare_features(indicators, current_price, prices_history)

        # BTC data if needed
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
        future_price = float(context_klines[i + TIMEFRAME_CONFIG['lookahead']][4])
        change_pct = ((future_price - current_price) / current_price) * 100

        if change_pct > TIMEFRAME_CONFIG['buy_threshold']:
            label = 0  # BUY
        elif change_pct < TIMEFRAME_CONFIG['sell_threshold']:
            label = 1  # SELL
        else:
            label = 2  # HOLD

        # Market regime
        regime = detect_market_regime(prices_history)

        test_features.append(features)
        test_labels.append(label)
        test_prices.append(current_price)
        test_dates.append(timestamp_to_datetime(context_klines[i][0]))
        test_market_regimes.append(regime)

    X_test = np.array(test_features)
    y_test = np.array(test_labels)

    print(f"[OK] Prepared {len(X_test)} test samples with {X_test.shape[1]} features")

    # Make predictions
    print(f"\n[>>] Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"[OK] Test accuracy: {accuracy*100:.2f}%")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"\n{'='*80}")
    print("CONFUSION MATRIX")
    print(f"{'='*80}")
    print(f"\n          Predicted")
    print(f"            BUY  SELL  HOLD")
    print(f"Actual BUY  {cm[0,0]:4d} {cm[0,1]:5d} {cm[0,2]:5d}")
    print(f"      SELL  {cm[1,0]:4d} {cm[1,1]:5d} {cm[1,2]:5d}")
    print(f"      HOLD  {cm[2,0]:4d} {cm[2,1]:5d} {cm[2,2]:5d}")

    # Classification report
    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*80}")
    target_names = ['BUY', 'SELL', 'HOLD']
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Trading simulation with detailed tracking
    print(f"\n{'='*80}")
    print("TRADING SIMULATION")
    print(f"{'='*80}")

    capital = 10000.0
    capital_history = [capital]
    position = None
    entry_price = None
    entry_date = None

    trades = []
    daily_returns = []

    wins = 0
    losses = 0
    total_win_pct = 0
    total_loss_pct = 0

    for i, pred in enumerate(y_pred):
        current_price = test_prices[i]
        current_date = test_dates[i]

        if pred == 0:  # BUY signal
            if position is None:
                position = 'long'
                entry_price = current_price
                entry_date = current_date

        elif pred == 1:  # SELL signal
            if position == 'long':
                exit_price = current_price
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl_amount = capital * (pnl_pct / 100)
                capital += pnl_amount

                # Record trade
                trade = {
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_amount': pnl_amount,
                    'duration_days': (current_date - entry_date).days,
                    'regime': test_market_regimes[i]
                }
                trades.append(trade)

                # Track wins/losses
                if pnl_pct > 0:
                    wins += 1
                    total_win_pct += pnl_pct
                else:
                    losses += 1
                    total_loss_pct += abs(pnl_pct)

                daily_returns.append(pnl_pct / 100)
                position = None

        capital_history.append(capital)

    # Close position if still open
    if position == 'long':
        exit_price = test_prices[-1]
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        pnl_amount = capital * (pnl_pct / 100)
        capital += pnl_amount

        trade = {
            'entry_date': entry_date,
            'exit_date': test_dates[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'duration_days': (test_dates[-1] - entry_date).days,
            'regime': test_market_regimes[-1]
        }
        trades.append(trade)

        if pnl_pct > 0:
            wins += 1
            total_win_pct += pnl_pct
        else:
            losses += 1
            total_loss_pct += abs(pnl_pct)

        daily_returns.append(pnl_pct / 100)

    # Calculate metrics
    total_return = ((capital - 10000) / 10000) * 100
    num_trades = len(trades)
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
    avg_win = (total_win_pct / wins) if wins > 0 else 0
    avg_loss = (total_loss_pct / losses) if losses > 0 else 0
    profit_factor = (total_win_pct / total_loss_pct) if total_loss_pct > 0 else 0

    # Advanced metrics
    returns_array = np.array(daily_returns) if daily_returns else np.array([0])
    sharpe = calculate_sharpe_ratio(returns_array)
    sortino = calculate_sortino_ratio(returns_array)
    max_dd = calculate_max_drawdown(capital_history)
    calmar = calculate_calmar_ratio(returns_array, max_dd) if max_dd > 0 else 0

    # Buy & Hold
    buy_hold_return = ((test_prices[-1] - test_prices[0]) / test_prices[0]) * 100
    outperformance = total_return - buy_hold_return

    # Print results
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Final Capital:      ${capital:,.2f}")
    print(f"  Total Return:       {total_return:+.2f}%")
    print(f"  Buy & Hold Return:  {buy_hold_return:+.2f}%")
    print(f"  Outperformance:     {outperformance:+.2f}%")
    print(f"\n  Number of Trades:   {num_trades}")
    print(f"  Win Rate:           {win_rate:.2f}%")
    print(f"  Wins / Losses:      {wins} / {losses}")
    print(f"  Avg Win:            {avg_win:.2f}%")
    print(f"  Avg Loss:           {avg_loss:.2f}%")
    print(f"  Profit Factor:      {profit_factor:.2f}")
    print(f"\n  Sharpe Ratio:       {sharpe:.2f}")
    print(f"  Sortino Ratio:      {sortino:.2f}")
    print(f"  Max Drawdown:       {max_dd:.2f}%")
    print(f"  Calmar Ratio:       {calmar:.2f}")

    # Performance by market regime
    print(f"\n{'='*80}")
    print("PERFORMANCE BY MARKET REGIME")
    print(f"{'='*80}")

    regime_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
    for trade in trades:
        regime = trade['regime']
        if trade['pnl_pct'] > 0:
            regime_stats[regime]['wins'] += 1
        else:
            regime_stats[regime]['losses'] += 1
        regime_stats[regime]['total_pnl'] += trade['pnl_pct']

    for regime in ['bull', 'bear', 'sideways']:
        stats = regime_stats[regime]
        total_trades = stats['wins'] + stats['losses']
        if total_trades > 0:
            win_rate_regime = (stats['wins'] / total_trades) * 100
            avg_pnl = stats['total_pnl'] / total_trades
            print(f"\n  {regime.upper():8s}:  {total_trades:3d} trades, Win Rate: {win_rate_regime:.1f}%, Avg PnL: {avg_pnl:+.2f}%")

    # Top 5 best and worst trades
    print(f"\n{'='*80}")
    print("TOP 5 BEST TRADES")
    print(f"{'='*80}")

    sorted_trades = sorted(trades, key=lambda x: x['pnl_pct'], reverse=True)
    for i, trade in enumerate(sorted_trades[:5]):
        print(f"\n  #{i+1}: {trade['entry_date'].strftime('%Y-%m-%d')} → {trade['exit_date'].strftime('%Y-%m-%d')}")
        print(f"       Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}")
        print(f"       PnL: {trade['pnl_pct']:+.2f}% (${trade['pnl_amount']:+.2f})")
        print(f"       Duration: {trade['duration_days']} days, Regime: {trade['regime']}")

    print(f"\n{'='*80}")
    print("TOP 5 WORST TRADES")
    print(f"{'='*80}")

    for i, trade in enumerate(sorted_trades[-5:][::-1]):
        print(f"\n  #{i+1}: {trade['entry_date'].strftime('%Y-%m-%d')} → {trade['exit_date'].strftime('%Y-%m-%d')}")
        print(f"       Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}")
        print(f"       PnL: {trade['pnl_pct']:+.2f}% (${trade['pnl_amount']:+.2f})")
        print(f"       Duration: {trade['duration_days']} days, Regime: {trade['regime']}")

    # Feature importance
    print(f"\n{'='*80}")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print(f"{'='*80}")

    feature_importance = model.feature_importances_
    feature_names = SELECTED_FEATURES_V5

    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Feature                    Importance")
    print(f"  {'-'*40}")
    for i, (name, importance) in enumerate(importance_pairs[:10]):
        print(f"  {i+1:2d}. {name:22s}  {importance:.4f}")

    # Return comprehensive results
    return {
        'crypto': crypto['name'],
        'timeframe': TIMEFRAME,
        'test_samples': len(X_test),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'final_capital': float(capital),
        'total_return_pct': float(total_return),
        'buy_hold_return_pct': float(buy_hold_return),
        'outperformance_pct': float(outperformance),
        'num_trades': int(num_trades),
        'win_rate': float(win_rate),
        'wins': int(wins),
        'losses': int(losses),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_dd),
        'calmar_ratio': float(calmar),
        'regime_stats': dict(regime_stats),
        'best_trades': [
            {k: (v.strftime('%Y-%m-%d') if isinstance(v, datetime) else v)
             for k, v in t.items()}
            for t in sorted_trades[:5]
        ],
        'worst_trades': [
            {k: (v.strftime('%Y-%m-%d') if isinstance(v, datetime) else v)
             for k, v in t.items()}
            for t in sorted_trades[-5:][::-1]
        ],
        'feature_importance': [
            {'feature': name, 'importance': float(imp)}
            for name, imp in importance_pairs
        ]
    }

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE XGBOOST ANALYSIS - 2025-2026 TEST PERIOD")
    print("="*80)
    print("\nTrain: All data until Dec 31, 2024")
    print("Test:  Jan 1, 2025 - Mar 18, 2026")
    print("="*80)

    results_all = []

    for crypto in CRYPTOS:
        # Load optimized model
        model_path = f"./training/models/xgboost/{crypto['cryptoId']}_{TIMEFRAME}_xgboost.pkl"

        if not os.path.exists(model_path):
            print(f"\n[ERROR] Model not found: {model_path}")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Run comprehensive analysis
        result = backtest_with_analysis(crypto, model)
        if result:
            results_all.append(result)

    # Save comprehensive results
    with open('comprehensive_analysis_results.json', 'w') as f:
        json.dump(results_all, f, indent=2)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n[OK] Comprehensive results saved to: comprehensive_analysis_results.json")
    print(f"[OK] Analyzed {len(results_all)} cryptos")
    print("="*80)

if __name__ == '__main__':
    main()
