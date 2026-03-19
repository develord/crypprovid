"""
Script d'entraînement ML pour toutes les cryptos (Python + TensorFlow)
Usage: python train_models.py
"""

import os
import sys
import json
import time
import requests
import numpy as np
import tensorflow as tf
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from data_manager import get_historical_data
from advanced_features import calculate_phase1_features, normalize_phase1_features

# Fix Windows encoding for emojis
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass  # stdout already configured or closed

# Vérifier si GPU est disponible
try:
    print("\n" + "="*60)
    print("Configuration GPU")
    print("="*60)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[OK] GPU detected: {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"   - {gpu}")
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[WARNING] No GPU detected, using CPU")
except (ValueError, OSError):
    # Stdout closed or GPU config fails - continue silently
    gpus = []
    pass

# Liste des cryptos à entraîner - Bitcoin + Ethereum + Solana
CRYPTOS = [
    {'cryptoId': 'bitcoin', 'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
    {'cryptoId': 'ethereum', 'symbol': 'ETHUSDT', 'name': 'Ethereum'},
    {'cryptoId': 'solana', 'symbol': 'SOLUSDT', 'name': 'Solana'}
]

# Configuration multi-timeframe - FOCUS: 1d + 1w (swing & position trading)
TIMEFRAME_CONFIGS = {
    '1d': {
        'interval': '1d',
        'limit': 2000,  # ~5-6 ans
        'lookahead': 7,  # 7 jours = 1 semaine
        'description': 'Daily candles (swing trading)'
    },
    '1w': {
        'interval': '1w',
        'limit': 400,  # ~7-8 ans
        'lookahead': 4,  # 4 semaines = 1 mois
        'description': 'Weekly candles (position trading)'
    }
}

# Configuration de l'entraînement
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'verbose': 1
}

def download_historical_data(symbol, interval='1d', limit=2000):
    """
    Obtenir données historiques (depuis cache ou Binance)
    Utilise data_manager pour éviter téléchargements répétés
    """
    return get_historical_data(symbol, interval, limit)

def calculate_rsi(prices, period=14):
    """Calculer RSI"""
    if len(prices) < period + 1:
        return None

    gains = 0
    losses = 0

    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(prices, period):
    """Calculer EMA"""
    if len(prices) < period:
        return None

    k = 2 / (period + 1)
    ema = np.mean(prices[:period])

    for price in prices[period:]:
        ema = price * k + ema * (1 - k)

    return ema

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculer MACD"""
    if len(prices) < slow:
        return None

    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    if ema_fast is None or ema_slow is None:
        return None

    macd_line = ema_fast - ema_slow

    # Calculer la ligne de signal (EMA du MACD)
    # Pour simplifier, on utilise une approximation
    signal_line = macd_line * 0.8  # Approximation simplifiée
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculer Bollinger Bands"""
    if len(prices) < period:
        return None

    recent_prices = prices[-period:]
    middle = np.mean(recent_prices)
    std = np.std(recent_prices)

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }

def calculate_stochastic_rsi(prices, period=14, smooth_k=3, smooth_d=3):
    """Calculer Stochastic RSI"""
    if len(prices) < period + smooth_k:
        return None

    # Calculer RSI sur la période
    rsi_values = []
    for i in range(len(prices) - period + 1):
        window = prices[i:i+period]
        rsi = calculate_rsi(window, period)
        if rsi is not None:
            rsi_values.append(rsi)

    if len(rsi_values) < smooth_k:
        return None

    # Calculer %K (stochastic sur RSI)
    recent_rsi = rsi_values[-smooth_k:]
    rsi_min = min(recent_rsi)
    rsi_max = max(recent_rsi)

    if rsi_max == rsi_min:
        k = 50
    else:
        k = ((recent_rsi[-1] - rsi_min) / (rsi_max - rsi_min)) * 100

    # %D est une moyenne mobile de %K (simplifié)
    d = k * 0.9  # Approximation

    return {
        'k': k,
        'd': d
    }

def calculate_atr(klines, period=14):
    """Calculer Average True Range"""
    if len(klines) < period:
        return None

    tr_values = []
    for i in range(1, len(klines)):
        high = float(klines[i][2])
        low = float(klines[i][3])
        prev_close = float(klines[i-1][4])

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_values.append(tr)

    if len(tr_values) < period:
        return None

    return np.mean(tr_values[-period:])

def calculate_obv(klines):
    """Calculer On Balance Volume"""
    if len(klines) < 2:
        return 0

    obv = 0
    for i in range(1, len(klines)):
        close = float(klines[i][4])
        prev_close = float(klines[i-1][4])
        volume = float(klines[i][5])

        if close > prev_close:
            obv += volume
        elif close < prev_close:
            obv -= volume

    return obv

def calculate_indicators(klines):
    """Calculer les indicateurs techniques"""
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    current_price = closes[-1]

    return {
        'rsi': calculate_rsi(closes),
        'ema20': calculate_ema(closes, 20),
        'ema50': calculate_ema(closes, 50),
        'ema200': calculate_ema(closes, 200),
        'macd': calculate_macd(closes),
        'bollingerBands': calculate_bollinger_bands(closes),
        'stochasticRsi': calculate_stochastic_rsi(closes),
        'atr': calculate_atr(klines),
        'obv': calculate_obv(klines),
        'currentPrice': current_price,
        'high': highs[-1],
        'low': lows[-1],
        'volume': volumes[-1]
    }

def prepare_features(indicators, current_price, prices_history=None):
    """Préparer les features (29 features complètes: 18 base + 11 Phase 1)"""
    features = []

    # RSI (2 features)
    if indicators['rsi'] is not None:
        features.append(indicators['rsi'] / 100)
        features.append((indicators['rsi'] - 50) / 50 if indicators['rsi'] > 50 else (50 - indicators['rsi']) / 50)
    else:
        features.extend([0.5, 0])

    # EMA20 (1 feature)
    if indicators['ema20'] is not None:
        features.append((current_price - indicators['ema20']) / current_price)
    else:
        features.append(0)

    # EMA50 (1 feature)
    if indicators['ema50'] is not None:
        features.append((current_price - indicators['ema50']) / current_price)
    else:
        features.append(0)

    # EMA200 (1 feature)
    if indicators['ema200'] is not None:
        features.append((current_price - indicators['ema200']) / current_price)
    else:
        features.append(0)

    # MACD (3 features)
    if indicators['macd'] is not None:
        features.append(indicators['macd']['macd'] / current_price)
        features.append(indicators['macd']['signal'] / current_price)
        features.append(indicators['macd']['histogram'] / current_price)
    else:
        features.extend([0, 0, 0])

    # Bollinger Bands (3 features)
    if indicators['bollingerBands'] is not None:
        bb = indicators['bollingerBands']
        features.append((current_price - bb['middle']) / current_price)
        features.append((bb['upper'] - bb['lower']) / current_price)
        # Position dans les bandes (0 = lower, 1 = upper)
        bb_position = (current_price - bb['lower']) / (bb['upper'] - bb['lower']) if bb['upper'] != bb['lower'] else 0.5
        features.append(bb_position)
    else:
        features.extend([0, 0, 0.5])

    # Stochastic RSI (2 features)
    if indicators['stochasticRsi'] is not None:
        features.append(indicators['stochasticRsi']['k'] / 100)
        features.append(indicators['stochasticRsi']['d'] / 100)
    else:
        features.extend([0.5, 0.5])

    # ATR (1 feature)
    if indicators['atr'] is not None:
        features.append(indicators['atr'] / current_price)
    else:
        features.append(0)

    # OBV (1 feature - normalisé avec tanh)
    if indicators['obv'] is not None:
        obv_normalized = np.tanh(indicators['obv'] / 1e9)
        features.append(obv_normalized)
    else:
        features.append(0)

    # Phase 1 Advanced Features (11 features)
    if prices_history is not None and len(prices_history) >= 50:
        try:
            phase1_features = calculate_phase1_features(prices_history, current_price)
            normalized_phase1 = normalize_phase1_features(phase1_features, current_price)
            features.extend(normalized_phase1)
        except Exception as e:
            # If Phase 1 features fail, use default values
            features.extend([0] * 11)
    else:
        # Not enough data for Phase 1 features, use defaults
        features.extend([0] * 11)

    # Compléter ou tronquer à exactement 29 features
    while len(features) < 29:
        features.append(0)

    return features[:29]

def create_labels(klines, lookahead=7, label_smoothing=0.05):
    """Créer les labels basés sur le prix futur avec label smoothing"""
    labels = []
    closes = [float(k[4]) for k in klines]

    for i in range(len(closes) - lookahead):
        current_price = closes[i]
        future_price = closes[i + lookahead]
        change = ((future_price - current_price) / current_price) * 100

        # Label smoothing: au lieu de [1, 0, 0], utiliser [0.95, 0.025, 0.025]
        # OPTIMISE: Reduit de 0.1 a 0.05 pour rendre le modele plus decisif
        smooth = label_smoothing / 2  # 0.025 pour les autres classes

        # OPTIMISE: Seuils reduits de 3% a 2% pour plus de signaux de trading
        if change > 2:
            labels.append([1 - label_smoothing, smooth, smooth])  # BUY: [0.95, 0.025, 0.025]
        elif change < -2:
            labels.append([smooth, 1 - label_smoothing, smooth])  # SELL: [0.025, 0.95, 0.025]
        else:
            labels.append([smooth, smooth, 1 - label_smoothing])  # HOLD: [0.025, 0.025, 0.95]

    return labels

def create_classification_model():
    """Créer le modèle de classification DNN avec régularisation améliorée (29 features: Phase 1)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(29,),
                             kernel_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),  # Augmenté de 0.3 à 0.4

        tf.keras.layers.Dense(64, activation='relu',
                             kernel_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),  # Augmenté de 0.3 à 0.4

        tf.keras.layers.Dense(32, activation='relu',
                             kernel_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),  # Augmenté de 0.2 à 0.3

        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_crypto(crypto, timeframe_key='1d'):
    """Entraîner le modèle pour une crypto avec un timeframe spécifique"""
    tf_config = TIMEFRAME_CONFIGS[timeframe_key]

    print(f"\n{'='*60}")
    print(f"[*] Training: {crypto['name']} ({crypto['symbol']}) - {timeframe_key.upper()}")
    print(f"    {tf_config['description']}")
    print('='*60)

    try:
        # 1. Télécharger les données selon le timeframe
        print(f"  [>>] Downloading {tf_config['interval']} candles (limit={tf_config['limit']})...")
        klines = download_historical_data(crypto['symbol'], tf_config['interval'], tf_config['limit'])

        if len(klines) < 200:
            print(f"  [WARNING] Insufficient data ({len(klines)} points)")
            return {'success': False, 'error': 'Insufficient data'}

        # 2. Préparer features et labels
        print(f"  [>>] Preparing features...")
        all_features = []
        lookahead = tf_config['lookahead']

        for i in range(len(klines) - lookahead):
            window_klines = klines[max(0, i - 200):i + 1]
            indicators = calculate_indicators(window_klines)
            current_price = float(klines[i][4])

            # Extract prices history for Phase 1 features
            prices_history = [float(k[4]) for k in window_klines]

            features = prepare_features(indicators, current_price, prices_history)
            all_features.append(features)

        labels = create_labels(klines, lookahead)

        print(f"  [>>] {len(all_features)} samples prepared")

        # Compter les labels
        labels_array = np.array(labels)
        buy_count = np.sum(labels_array[:, 0] == 1)
        sell_count = np.sum(labels_array[:, 1] == 1)
        hold_count = np.sum(labels_array[:, 2] == 1)
        print(f"  [>>] Distribution: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")

        # 3. Créer et entraîner le modèle
        print(f"  [>>] Creating model...")
        model = create_classification_model()

        print(f"  [>>] Training in progress ({TRAINING_CONFIG['epochs']} epochs)...")

        X = np.array(all_features)
        y = np.array(labels)

        start_time = time.time()

        # Callbacks pour afficher la progression et early stopping
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 10 == 0 or epoch == TRAINING_CONFIG['epochs'] - 1:
                    print(f"    Epoch {epoch + 1}/{TRAINING_CONFIG['epochs']} - "
                          f"Loss: {logs['loss']:.4f} - "
                          f"Acc: {logs['accuracy']*100:.2f}% - "
                          f"Val Acc: {logs['val_accuracy']*100:.2f}%")

        # Early stopping: arrête l'entraînement si val_loss ne s'améliore plus
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Attend 15 epochs sans amélioration
            restore_best_weights=True,  # Restaure les meilleurs poids
            verbose=0
        )

        # Reduce learning rate si val_loss stagne
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Divise le LR par 2
            patience=7,
            min_lr=0.00001,
            verbose=0
        )

        history = model.fit(
            X, y,
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            validation_split=TRAINING_CONFIG['validation_split'],
            verbose=0,
            callbacks=[ProgressCallback(), early_stopping, reduce_lr]
        )

        training_time = time.time() - start_time

        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        print(f"\n  [OK] Training completed in {training_time:.1f}s")
        print(f"  [>>] Final Accuracy: {final_acc*100:.2f}%")
        print(f"  [>>] Validation Accuracy: {final_val_acc*100:.2f}%")
        print(f"  [>>] Final Loss: {final_loss:.4f}")

        # 4. Sauvegarder le modèle avec timeframe dans le nom
        model_dir = "./models/snn"
        os.makedirs(model_dir, exist_ok=True)
        save_path = f"{model_dir}/{crypto['cryptoId']}_{timeframe_key}.keras"
        print(f"  [>>] Saving model: {save_path}")

        # Sauvegarder au format Keras natif
        model.save(save_path)

        return {
            'success': True,
            'timeframe': timeframe_key,
            'finalLoss': final_loss,
            'finalAccuracy': final_acc,
            'finalValAccuracy': final_val_acc,
            'samples': len(all_features),
            'trainingTime': training_time
        }

    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def main():
    """Fonction principale - Multi-Timeframe Training"""
    print('\n' + '='*60)
    print('[*] ML TRAINING - MULTI-TIMEFRAME (4h, 1d, 1w)')
    print('='*60)
    print(f"\n[>>] Configuration:")
    print(f"   - Cryptos: {len(CRYPTOS)}")
    print(f"   - Timeframes: {list(TIMEFRAME_CONFIGS.keys())}")
    print(f"   - Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   - Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   - Validation split: {TRAINING_CONFIG['validation_split']*100}%")
    print(f"\n[>>] Total models: {len(CRYPTOS) * len(TIMEFRAME_CONFIGS)}")
    print(f"[>>] Estimated time: {len(CRYPTOS) * len(TIMEFRAME_CONFIGS) * 2} minutes\n")

    results = []
    start_time = time.time()

    # Entraîner chaque crypto pour chaque timeframe
    total_models = len(CRYPTOS) * len(TIMEFRAME_CONFIGS)
    current = 0

    for i, crypto in enumerate(CRYPTOS):
        for tf_key in TIMEFRAME_CONFIGS.keys():
            current += 1
            print(f"\n[{current}/{total_models}] {crypto['name']} - {tf_key.upper()}")

            result = train_crypto(crypto, tf_key)
            results.append({
                'crypto': crypto['name'],
                'timeframe': tf_key,
                **result
            })

            # Attendre un peu entre chaque modèle pour ne pas surcharger l'API
            if current < total_models:
                time.sleep(2)

    # Résumé final
    total_time = (time.time() - start_time) / 60

    print('\n' + '='*60)
    print('[*] FINAL SUMMARY')
    print('='*60)
    print(f"\n[>>] Total time: {total_time:.1f} minutes\n")

    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    print(f"[OK] Successful: {len(successful)}/{total_models}")
    print(f"[ERROR] Failed: {len(failed)}/{total_models}\n")

    if successful:
        print('Successfully trained models:')
        # Grouper par crypto
        for crypto in CRYPTOS:
            crypto_results = [r for r in successful if r['crypto'] == crypto['name']]
            if crypto_results:
                print(f"\n  {crypto['name']}:")
                for r in crypto_results:
                    print(f"    - {r['timeframe'].upper()}: Acc={r['finalAccuracy']*100:.2f}%, "
                          f"Val Acc={r['finalValAccuracy']*100:.2f}%, Samples={r['samples']}")

    if failed:
        print('\nFailures:')
        for r in failed:
            print(f"  - {r['crypto']}: {r.get('error', 'Unknown error')}")

    print('\n' + '='*60)
    print('[*] TRAINING COMPLETED!')
    print('='*60 + '\n')

if __name__ == '__main__':
    main()
