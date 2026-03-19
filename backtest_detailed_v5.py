"""
Backtest Détaillé XGBoost v5 - 2 ans
Simule trading avec $1000 initial pour BTC, ETH, SOL (1d + 1w)
Génère tableau d'analyse complet
"""

import os
import sys
import pickle
import numpy as np
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))

from data_manager import get_historical_data
from train_models import calculate_indicators, prepare_features
from xgboost_features import calculate_all_xgboost_features
from feature_selection_v5 import SELECTED_FEATURES_V5, select_features_from_vector

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
CRYPTOS = [
    {'cryptoId': 'bitcoin', 'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
    {'cryptoId': 'ethereum', 'symbol': 'ETHUSDT', 'name': 'Ethereum'},
    {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana'}
]

TIMEFRAMES = {
    '1d': {
        'interval': '1d',
        'limit': 3000,
        'backtest_days': 730,  # 2 ans
        'lookahead': 7
    },
    '1w': {
        'interval': '1w',
        'limit': 800,
        'backtest_weeks': 104,  # 2 ans
        'lookahead': 4
    }
}

INITIAL_CAPITAL = 1000  # $1000


class DetailedBacktest:
    """Backtest avec statistiques détaillées"""

    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0
        self.entry_time = None
        self.trades = []
        self.equity_curve = []
        self.signals_log = []

    def execute_signal(self, signal, current_price, timestamp):
        """Exécuter signal avec tracking détaillé"""

        # Log signal
        self.signals_log.append({
            'timestamp': timestamp,
            'signal': signal,
            'price': current_price,
            'position_before': self.position
        })

        # Fermer position si signal opposé
        if self.position == 'long' and signal == 1:  # SELL
            pnl = (current_price - self.entry_price) / self.entry_price
            self.capital *= (1 + pnl)

            trade_duration = (timestamp - self.entry_time) / (1000 * 60 * 60 * 24) if self.entry_time else 0

            self.trades.append({
                'type': 'long',
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'entry_time': self.entry_time,
                'exit_time': timestamp,
                'duration_days': trade_duration,
                'pnl_pct': pnl * 100,
                'pnl_usd': self.capital - (self.capital / (1 + pnl))
            })
            self.position = None

        elif self.position == 'short' and signal == 0:  # BUY
            pnl = (self.entry_price - current_price) / self.entry_price
            self.capital *= (1 + pnl)

            trade_duration = (timestamp - self.entry_time) / (1000 * 60 * 60 * 24) if self.entry_time else 0

            self.trades.append({
                'type': 'short',
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'entry_time': self.entry_time,
                'exit_time': timestamp,
                'duration_days': trade_duration,
                'pnl_pct': pnl * 100,
                'pnl_usd': self.capital - (self.capital / (1 + pnl))
            })
            self.position = None

        # Ouvrir nouvelle position
        if self.position is None:
            if signal == 0:  # BUY
                self.position = 'long'
                self.entry_price = current_price
                self.entry_time = timestamp
            elif signal == 1:  # SELL
                self.position = 'short'
                self.entry_price = current_price
                self.entry_time = timestamp

        # Equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'capital': self.capital,
            'position': self.position
        })

    def close_final_position(self, final_price, timestamp):
        """Fermer position finale"""
        if self.position == 'long':
            pnl = (final_price - self.entry_price) / self.entry_price
            self.capital *= (1 + pnl)

            trade_duration = (timestamp - self.entry_time) / (1000 * 60 * 60 * 24) if self.entry_time else 0

            self.trades.append({
                'type': 'long',
                'entry_price': self.entry_price,
                'exit_price': final_price,
                'entry_time': self.entry_time,
                'exit_time': timestamp,
                'duration_days': trade_duration,
                'pnl_pct': pnl * 100,
                'pnl_usd': self.capital - (self.capital / (1 + pnl))
            })

        elif self.position == 'short':
            pnl = (self.entry_price - final_price) / self.entry_price
            self.capital *= (1 + pnl)

            trade_duration = (timestamp - self.entry_time) / (1000 * 60 * 60 * 24) if self.entry_time else 0

            self.trades.append({
                'type': 'short',
                'entry_price': self.entry_price,
                'exit_price': final_price,
                'entry_time': self.entry_time,
                'exit_time': timestamp,
                'duration_days': trade_duration,
                'pnl_pct': pnl * 100,
                'pnl_usd': self.capital - (self.capital / (1 + pnl))
            })

        self.position = None

    def get_detailed_stats(self):
        """Statistiques détaillées"""
        if not self.trades:
            return None

        # Séparer wins/losses
        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        losses = [t for t in self.trades if t['pnl_pct'] <= 0]

        # ROI total
        total_return_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100

        # Calculs
        total_trades = len(self.trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        avg_win_pct = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss_pct = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

        avg_win_usd = np.mean([t['pnl_usd'] for t in wins]) if wins else 0
        avg_loss_usd = np.mean([t['pnl_usd'] for t in losses]) if losses else 0

        max_win_pct = max([t['pnl_pct'] for t in self.trades])
        max_loss_pct = min([t['pnl_pct'] for t in self.trades])

        max_win_usd = max([t['pnl_usd'] for t in self.trades])
        max_loss_usd = min([t['pnl_usd'] for t in self.trades])

        avg_trade_duration = np.mean([t['duration_days'] for t in self.trades])

        # Profit factor
        total_wins = sum([t['pnl_usd'] for t in wins]) if wins else 0
        total_losses = abs(sum([t['pnl_usd'] for t in losses])) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Max drawdown
        peak = self.initial_capital
        max_dd_pct = 0
        for point in self.equity_curve:
            if point['capital'] > peak:
                peak = point['capital']
            dd = ((peak - point['capital']) / peak) * 100
            if dd > max_dd_pct:
                max_dd_pct = dd

        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i]['capital'] - self.equity_curve[i-1]['capital']) / self.equity_curve[i-1]['capital']
                returns.append(ret)

            if returns and np.std(returns) > 0:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        return {
            'total_return_pct': total_return_pct,
            'final_capital': self.capital,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_win_usd': avg_win_usd,
            'avg_loss_usd': avg_loss_usd,
            'max_win_pct': max_win_pct,
            'max_loss_pct': max_loss_pct,
            'max_win_usd': max_win_usd,
            'max_loss_usd': max_loss_usd,
            'avg_trade_duration': avg_trade_duration,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd_pct
        }


def run_backtest(crypto, timeframe_key, model_path):
    """Exécuter backtest détaillé"""

    if not os.path.exists(model_path):
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
        return None

    # BTC pour cross-asset
    klines_btc = None
    if crypto['symbol'] != 'BTCUSDT':
        klines_btc = get_historical_data(
            'BTCUSDT',
            interval=tf_config['interval'],
            limit=tf_config['limit']
        )

    # Split
    if timeframe_key == '1d':
        test_size = min(tf_config['backtest_days'], len(klines) // 3)
    else:
        test_size = min(tf_config['backtest_weeks'], len(klines) // 3)

    test_start = len(klines) - test_size - tf_config['lookahead']

    # Backtest
    bt = DetailedBacktest(initial_capital=INITIAL_CAPITAL)
    indicators_history = []

    for i in range(test_start, len(klines) - tf_config['lookahead']):
        window_data = klines[max(0, i-200):i+1]
        indicators = calculate_indicators(window_data)
        current_price = float(klines[i][4])
        timestamp = klines[i][0]

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

        bt.execute_signal(prediction, current_price, timestamp)

    # Fermer position finale
    final_price = float(klines[-1][4])
    final_timestamp = klines[-1][0]
    bt.close_final_position(final_price, final_timestamp)

    stats = bt.get_detailed_stats()

    if stats:
        # Buy & Hold
        start_price = float(klines[test_start][4])
        buy_hold_return = ((final_price - start_price) / start_price) * 100

        stats['buy_hold_return'] = buy_hold_return
        stats['outperformance'] = stats['total_return_pct'] - buy_hold_return

    return stats


def main():
    print("\n" + "="*80)
    print("BACKTEST DÉTAILLÉ XGBoost v5 - 2 ANS (Capital Initial: $1000)")
    print("="*80)

    results = []

    for crypto in CRYPTOS:
        for timeframe_key in ['1d', '1w']:
            model_path = f'./training/models/xgboost/{crypto["cryptoId"]}_{timeframe_key}_xgboost.pkl'

            print(f"\n[>>] {crypto['name']} {timeframe_key.upper()}...")

            stats = run_backtest(crypto, timeframe_key, model_path)

            if stats:
                results.append({
                    'crypto': crypto['name'],
                    'timeframe': timeframe_key,
                    **stats
                })

    # Tableau final
    print("\n" + "="*80)
    print("RÉSULTATS DÉTAILLÉS - TABLEAU D'ANALYSE")
    print("="*80)

    print(f"\n{'='*140}")
    print(f"{'Modèle':<15} {'Capital':<12} {'ROI':<10} {'Trades':<8} {'Win%':<8} "
          f"{'Avg Win':<12} {'Avg Loss':<12} {'Max Win':<12} {'Max Loss':<12}")
    print('-'*140)

    for r in results:
        print(f"{r['crypto']} {r['timeframe'].upper():<10} "
              f"${r['final_capital']:>8.2f}   "
              f"{r['total_return_pct']:>+7.1f}%  "
              f"{r['total_trades']:>5}   "
              f"{r['win_rate']:>5.1f}%  "
              f"{r['avg_win_pct']:>+5.1f}% (${r['avg_win_usd']:>5.0f})  "
              f"{r['avg_loss_pct']:>+5.1f}% (${r['avg_loss_usd']:>5.0f})  "
              f"{r['max_win_pct']:>+5.1f}% (${r['max_win_usd']:>5.0f})  "
              f"{r['max_loss_pct']:>+5.1f}% (${r['max_loss_usd']:>5.0f})")

    print('='*140)

    print(f"\n{'='*120}")
    print(f"{'Modèle':<15} {'Sharpe':<10} {'Max DD':<10} {'Profit Factor':<15} "
          f"{'Avg Duration':<15} {'Buy&Hold':<12} {'Outperf':<12}")
    print('-'*120)

    for r in results:
        print(f"{r['crypto']} {r['timeframe'].upper():<10} "
              f"{r['sharpe_ratio']:>7.3f}   "
              f"{r['max_drawdown_pct']:>6.1f}%   "
              f"{r['profit_factor']:>12.2f}   "
              f"{r['avg_trade_duration']:>10.1f} jours  "
              f"{r['buy_hold_return']:>+8.1f}%   "
              f"{r['outperformance']:>+9.1f}%")

    print('='*120)

    # Sauvegarder
    with open('backtest_detailed_v5_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Résultats sauvegardés: backtest_detailed_v5_results.json")
    print("="*80)


if __name__ == '__main__':
    main()
