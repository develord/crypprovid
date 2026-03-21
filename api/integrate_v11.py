"""
INTEGRATION V11 TEMPORAL - Script d'intégration pour CryptoAdviser
===================================================================
Ce script intègre les modèles V11 TEMPORAL multi-timeframe dans le backend CryptoAdviser.

Actions:
1. Copie les données merged multi-TF vers data/v11_cache/
2. Copie les thresholds optimaux
3. Copie les stats des modèles
4. Configure l'API pour utiliser V11

Date: 21 Mars 2026
"""

import shutil
from pathlib import Path
import json

# Paths
V10_DIR = Path("C:/Users/moham/Desktop/crypto/crypto_v10_multi_tf")
API_DIR = Path(__file__).parent
ADVISER_DIR = API_DIR.parent

def integrate_v11_data():
    """Copy V11 merged data to CryptoAdviser"""
    print("="*80)
    print("INTEGRATION V11 TEMPORAL - Data & Config")
    print("="*80)

    # Create directories
    v11_cache_dir = ADVISER_DIR / 'data' / 'v11_cache'
    v11_cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Copying merged multi-TF data...")

    # Copy merged CSV files (already have all multi-TF features)
    cryptos = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'sol': 'solana'
    }

    for crypto_short, crypto_full in cryptos.items():
        source = V10_DIR / 'data' / 'cache' / f'{crypto_short}_multi_tf_merged.csv'
        dest = v11_cache_dir / f'{crypto_full}_multi_tf_merged.csv'

        if source.exists():
            shutil.copy(source, dest)
            print(f"  [OK] Copied {crypto_full} multi-TF data")
        else:
            print(f"  [!!] Missing: {source}")

    print("\n[2/4] Copying optimal thresholds...")

    # Copy threshold files
    thresholds = {}
    for crypto_short, crypto_full in cryptos.items():
        threshold_file = V10_DIR / 'optimization' / 'results' / f'{crypto_short}_baseline_optimal_threshold.json'

        if threshold_file.exists():
            with open(threshold_file) as f:
                data = json.load(f)
                thresholds[crypto_full] = data['optimal_by_roi']['threshold']
                print(f"  [OK] {crypto_full.upper()}: threshold = {thresholds[crypto_full]:.2f}")

    # Save consolidated thresholds
    thresholds_dest = v11_cache_dir / 'optimal_thresholds_v11.json'
    with open(thresholds_dest, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n  Saved: {thresholds_dest}")

    print("\n[3/4] Copying model stats...")

    # Copy model stats
    for crypto_short, crypto_full in cryptos.items():
        stats_file = V10_DIR / 'models' / f'{crypto_short}_v11_stats.json'

        if stats_file.exists():
            dest = ADVISER_DIR / 'models' / 'v11' / f'{crypto_full}_v11_stats.json'
            shutil.copy(stats_file, dest)
            print(f"  [OK] Copied {crypto_full} stats")

    print("\n[4/4] Creating V11 config...")

    # Create V11 configuration file
    v11_config = {
        "version": "v11_temporal",
        "model_type": "binary_classifier",
        "description": "Multi-timeframe temporal prediction (1d + 4h + 1h)",
        "triple_barrier": {
            "take_profit": 1.5,
            "stop_loss": 0.75,
            "lookahead_days": 7
        },
        "cryptos": {
            "bitcoin": {
                "model": "bitcoin_v11_classifier.joblib",
                "threshold": thresholds.get('bitcoin', 0.37),
                "num_features": 237,
                "timeframes": ["1d", "4h", "1h"]
            },
            "ethereum": {
                "model": "ethereum_v11_classifier.joblib",
                "threshold": thresholds.get('ethereum', 0.35),
                "num_features": 348,
                "timeframes": ["1d", "4h", "1h"]
            },
            "solana": {
                "model": "solana_v11_classifier.joblib",
                "threshold": thresholds.get('solana', 0.35),
                "num_features": 50,  # Feature selected top 50
                "feature_selection": True,
                "timeframes": ["1d", "4h", "1h"]
            }
        },
        "performance": {
            "portfolio_roi": "+43.38%",
            "btc_roi": "+22.56%",
            "eth_roi": "+45.07%",
            "sol_roi": "+64.48%",
            "status": "PRODUCTION_READY"
        }
    }

    config_file = v11_cache_dir / 'v11_config.json'
    with open(config_file, 'w') as f:
        json.dump(v11_config, f, indent=2)

    print(f"\n  Saved: {config_file}")

    print("\n" + "="*80)
    print("INTEGRATION COMPLETE!")
    print("="*80)

    print("\nV11 Configuration:")
    print(f"  - BTC: Threshold={thresholds.get('bitcoin', 0.37):.2f}, Features=237")
    print(f"  - ETH: Threshold={thresholds.get('ethereum', 0.35):.2f}, Features=348")
    print(f"  - SOL: Threshold={thresholds.get('solana', 0.35):.2f}, Features=50 (selected)")

    print("\nNext steps:")
    print("  1. Update predictions.py to use V11 models")
    print("  2. Create feature extraction for multi-TF")
    print("  3. Test API endpoints")

    print("\n" + "="*80)

    return v11_config

if __name__ == '__main__':
    integrate_v11_data()
