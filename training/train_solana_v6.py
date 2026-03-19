"""
Train Solana V6 Model
"""
import sys
import os

# Set path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

# Import from existing v6 script
from train_models_xgboost_v6 import train_with_kfold_cv

# Solana config
SOLANA = {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana'}

def main():
    print("="*80)
    print("TRAINING SOLANA V6 MODEL")
    print("="*80)
    print("Training Solana with V6 features (57 features)...")
    print("="*80)

    result = train_with_kfold_cv(SOLANA, '1d', n_splits=3)

    if result:
        print("\n" + "="*80)
        print("SOLANA V6 TRAINING COMPLETE")
        print("="*80)
        print(f"K-Fold Avg Val Acc: {result['kfold_avg_val_acc']*100:.2f}%")
        print(f"Final Val Acc: {result['final_val_acc']*100:.2f}%")
        print(f"Overfitting Gap: {result['overfitting_gap']*100:.2f}%")
        print("="*80)

        import json
        with open('training_solana_v6_results.json', 'w') as f:
            json_result = result.copy()
            json_result['fold_results'] = [
                {k: float(v) if isinstance(v, (float, int)) else v for k, v in fold.items()}
                for fold in result.get('fold_results', [])
            ]
            json.dump([json_result], f, indent=2)

        print(f"[OK] Results saved to: training_solana_v6_results.json")
    else:
        print("[ERROR] Training failed")

if __name__ == '__main__':
    main()
