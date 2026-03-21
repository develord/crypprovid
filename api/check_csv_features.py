import pandas as pd
from pathlib import Path

# Load Bitcoin CSV
csv_path = Path('C:\\Users\\moham\\Desktop\\crypto\\CryptoAdviser\\data\\v11_cache\\bitcoin_multi_tf_merged.csv')
df = pd.read_csv(csv_path, index_col=0)

# Exclude non-feature columns
exclude = ['open', 'high', 'low', 'close', 'volume', 'label_class', 'label_numeric', 'price_target_pct', 'future_price', 'triple_barrier_label']
features = [col for col in df.columns if col not in exclude]

print(f"Total features in Bitcoin CSV: {len(features)}")
print("\nFirst 30 features:")
for i, f in enumerate(features[:30]):
    print(f"{i+1}. {f}")

print("\nLast 10 features:")
for i, f in enumerate(features[-10:], len(features)-9):
    print(f"{i}. {f}")
