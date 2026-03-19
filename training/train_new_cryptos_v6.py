"""
Train V6 models for 6 NEW cryptos
(Bitcoin and Ethereum already trained)
"""
import sys
import os

# Set path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

# Import from existing v6 script
from train_models_xgboost_v6 import train_with_kfold_cv

# NEW CRYPTOS ONLY (skip BTC and ETH - already trained)
NEW_CRYPTOS = [
    {'cryptoId': 'bnb', 'symbol': 'BNBUSDT', 'name': 'BNB'},
    {'cryptoId': 'xrp', 'symbol': 'XRPUSDT', 'name': 'XRP'},
    {'cryptoId': 'cardano', 'symbol': 'ADAUSDT', 'name': 'Cardano'},
    {'cryptoId': 'avalanche', 'symbol': 'AVAXUSDT', 'name': 'Avalanche'},
    {'cryptoId': 'polkadot', 'symbol': 'DOTUSDT', 'name': 'Polkadot'},
    {'cryptoId': 'polygon', 'symbol': 'MATICUSDT', 'name': 'Polygon'}
]

def main():
    print("="*80)
    print("TRAINING V6 MODELS - 6 NEW CRYPTOS")
    print("="*80)
    print("BTC and ETH already trained - skipping")
    print(f"Training {len(NEW_CRYPTOS)} new cryptos with V6 features...")
    print("="*80)
    
    results = []
    for crypto in NEW_CRYPTOS:
        print(f"\n[{NEW_CRYPTOS.index(crypto)+1}/{len(NEW_CRYPTOS)}] Training {crypto['name']}...")
        result = train_with_kfold_cv(crypto, '1d', n_splits=3)
        if result:
            results.append(result)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - 6 NEW CRYPTOS")
    print("="*80)
    
    for r in results:
        print(f"\n{r['crypto']}:")
        print(f"  K-Fold Avg Val Acc: {r['kfold_avg_val_acc']*100:.2f}%")
        print(f"  Final Val Acc: {r['final_val_acc']*100:.2f}%")
        print(f"  Overfitting Gap: {r['overfitting_gap']*100:.2f}%")
    
    import json
    with open('training_new_cryptos_v6_results.json', 'w') as f:
        json_results = []
        for r in results:
            json_r = r.copy()
            json_r['fold_results'] = [
                {k: float(v) if isinstance(v, (float, int)) else v for k, v in fold.items()}
                for fold in r.get('fold_results', [])
            ]
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    print(f"\n[OK] Results saved to: training_new_cryptos_v6_results.json")
    print("="*80)

if __name__ == '__main__':
    main()
