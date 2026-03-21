import pandas as pd
from pathlib import Path

# Load Bitcoin CSV
csv_path = Path('C:\\Users\\moham\\Desktop\\crypto\\CryptoAdviser\\data\\v11_cache\\bitcoin_multi_tf_merged.csv')
df = pd.read_csv(csv_path, index_col=0)

# Exclude non-feature columns
exclude = ['open', 'high', 'low', 'close', 'volume', 'label_class', 'label_numeric', 'price_target_pct', 'future_price', 'triple_barrier_label']
features = [col for col in df.columns if col not in exclude]

print(f"Total features: {len(features)}\n")
print("ALL FEATURES:")
print("=" * 80)
for i, f in enumerate(features, 1):
    print(f"{i:3d}. {f}")
