"""
Advanced Features (Phase 1) - Features techniques avancées pour ML crypto
Calcule 11 features supplémentaires au-delà des indicateurs de base
"""

import numpy as np

def calculate_phase1_features(prices_history, current_price):
    """
    Calculer les 11 features Phase 1

    Args:
        prices_history: Liste de prix historiques (au moins 50 valeurs)
        current_price: Prix actuel

    Returns:
        Dict avec 11 features avancées
    """
    if len(prices_history) < 50:
        # Pas assez de données, retourner features par défaut
        return {
            'volatility_short': 0,
            'volatility_long': 0,
            'momentum_5': 0,
            'momentum_10': 0,
            'momentum_20': 0,
            'mean_reversion_5': 0,
            'mean_reversion_20': 0,
            'price_acceleration': 0,
            'higher_highs_ratio': 0,
            'lower_lows_ratio': 0,
            'price_range_ratio': 0
        }

    prices = np.array(prices_history)

    # 1. Volatility Short (5 derniers jours)
    volatility_short = np.std(prices[-5:]) / np.mean(prices[-5:]) if len(prices) >= 5 else 0

    # 2. Volatility Long (20 derniers jours)
    volatility_long = np.std(prices[-20:]) / np.mean(prices[-20:]) if len(prices) >= 20 else 0

    # 3. Momentum 5 jours (Rate of Change)
    momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0

    # 4. Momentum 10 jours
    momentum_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0

    # 5. Momentum 20 jours
    momentum_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0

    # 6. Mean Reversion 5j (Distance à la moyenne mobile 5j)
    mean_5 = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
    mean_reversion_5 = (current_price - mean_5) / mean_5

    # 7. Mean Reversion 20j (Distance à la moyenne mobile 20j)
    mean_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
    mean_reversion_20 = (current_price - mean_20) / mean_20

    # 8. Price Acceleration (dérivée seconde du prix)
    if len(prices) >= 3:
        returns = np.diff(prices[-5:]) / prices[-5:-1]  # Returns
        acceleration = np.mean(np.diff(returns)) if len(returns) > 1 else 0
    else:
        acceleration = 0

    # 9. Higher Highs Ratio (% de nouveaux highs dans les 10 derniers jours)
    if len(prices) >= 10:
        highs = []
        for i in range(-10, 0):
            if i == -10:
                highs.append(False)
            else:
                highs.append(prices[i] > prices[i-1])
        higher_highs_ratio = sum(highs) / len(highs)
    else:
        higher_highs_ratio = 0.5

    # 10. Lower Lows Ratio (% de nouveaux lows dans les 10 derniers jours)
    if len(prices) >= 10:
        lows = []
        for i in range(-10, 0):
            if i == -10:
                lows.append(False)
            else:
                lows.append(prices[i] < prices[i-1])
        lower_lows_ratio = sum(lows) / len(lows)
    else:
        lower_lows_ratio = 0.5

    # 11. Price Range Ratio (range sur 20j / prix actuel)
    if len(prices) >= 20:
        max_20 = np.max(prices[-20:])
        min_20 = np.min(prices[-20:])
        price_range_ratio = (max_20 - min_20) / current_price
    else:
        price_range_ratio = 0

    return {
        'volatility_short': volatility_short,
        'volatility_long': volatility_long,
        'momentum_5': momentum_5,
        'momentum_10': momentum_10,
        'momentum_20': momentum_20,
        'mean_reversion_5': mean_reversion_5,
        'mean_reversion_20': mean_reversion_20,
        'price_acceleration': acceleration,
        'higher_highs_ratio': higher_highs_ratio,
        'lower_lows_ratio': lower_lows_ratio,
        'price_range_ratio': price_range_ratio
    }


def normalize_phase1_features(features, current_price):
    """
    Normaliser les features Phase 1 pour être dans des ranges raisonnables

    Args:
        features: Dict avec features Phase 1
        current_price: Prix actuel (pour normalisation)

    Returns:
        Liste de 11 valeurs normalisées
    """
    normalized = []

    # 1. Volatility short (0-1, tanh pour limiter extremes)
    normalized.append(np.tanh(features['volatility_short'] * 10))

    # 2. Volatility long (0-1, tanh)
    normalized.append(np.tanh(features['volatility_long'] * 10))

    # 3. Momentum 5j (tanh pour limiter à -1/+1)
    normalized.append(np.tanh(features['momentum_5']))

    # 4. Momentum 10j
    normalized.append(np.tanh(features['momentum_10']))

    # 5. Momentum 20j
    normalized.append(np.tanh(features['momentum_20']))

    # 6. Mean reversion 5j
    normalized.append(np.tanh(features['mean_reversion_5'] * 5))

    # 7. Mean reversion 20j
    normalized.append(np.tanh(features['mean_reversion_20'] * 5))

    # 8. Price acceleration (déjà petit, juste tanh)
    normalized.append(np.tanh(features['price_acceleration'] * 100))

    # 9. Higher highs ratio (déjà 0-1)
    normalized.append(features['higher_highs_ratio'])

    # 10. Lower lows ratio (déjà 0-1)
    normalized.append(features['lower_lows_ratio'])

    # 11. Price range ratio (tanh)
    normalized.append(np.tanh(features['price_range_ratio']))

    return normalized


def test_features():
    """Test des features Phase 1 avec données simulées"""
    # Simuler historique de prix (trend haussier avec volatilité)
    np.random.seed(42)
    base_price = 40000
    prices = []
    for i in range(100):
        trend = i * 50  # Trend haussier
        noise = np.random.randn() * 500  # Volatilité
        prices.append(base_price + trend + noise)

    current_price = prices[-1]

    # Calculer features
    features = calculate_phase1_features(prices, current_price)
    normalized = normalize_phase1_features(features, current_price)

    print("="*60)
    print("TEST - Advanced Features (Phase 1)")
    print("="*60)
    print(f"\nCurrent Price: ${current_price:,.2f}")
    print(f"\nRaw Features:")
    for key, value in features.items():
        print(f"  {key:25s}: {value:10.6f}")

    print(f"\nNormalized Features ({len(normalized)} values):")
    feature_names = [
        'volatility_short', 'volatility_long',
        'momentum_5', 'momentum_10', 'momentum_20',
        'mean_reversion_5', 'mean_reversion_20',
        'price_acceleration',
        'higher_highs_ratio', 'lower_lows_ratio',
        'price_range_ratio'
    ]
    for name, value in zip(feature_names, normalized):
        print(f"  {name:25s}: {value:10.6f}")

    print("\n" + "="*60)
    print("✅ Test passed - 11 features generated successfully")
    print("="*60)


if __name__ == '__main__':
    test_features()
