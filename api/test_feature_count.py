"""
Test script to verify feature counts match model expectations
"""
import sys
sys.path.insert(0, 'C:\\Users\\moham\\Desktop\\crypto\\CryptoAdviser\\api')

from live_features import LiveFeatureEngine
import json
from pathlib import Path

# Load config to see expected feature counts
config_file = Path('C:\\Users\\moham\\Desktop\\crypto\\CryptoAdviser\\data\\v11_cache\\v11_config.json')
with open(config_file) as f:
    config = json.load(f)

print("=" * 60)
print("V11 FEATURE COUNT VERIFICATION")
print("=" * 60)

# Initialize engine
engine = LiveFeatureEngine()

for crypto_id in ['bitcoin', 'ethereum', 'solana']:
    print(f"\n{crypto_id.upper()}:")
    print("-" * 40)

    # Expected from config
    expected = config['cryptos'][crypto_id]['num_features']
    print(f"  Expected features: {expected}")

    # Generate live features
    try:
        features, price = engine.get_live_features(crypto_id)
        actual = len(features)
        print(f"  Generated features: {actual}")
        print(f"  Current price: ${price:.2f}")

        # Check match
        if actual == expected:
            print(f"  ✓ MATCH - Features count is correct!")
        else:
            print(f"  ✗ MISMATCH - Expected {expected}, got {actual}")
            print(f"  Difference: {actual - expected}")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

print("\n" + "=" * 60)
