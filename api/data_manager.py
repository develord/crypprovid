"""
Gestionnaire de cache pour données historiques (Binance + CoinGecko)
Télécharge et sauvegarde les données en JSON pour éviter les appels API répétés
CoinGecko utilisé comme fallback pour weekly data (plus d'historique)
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta

# Chemin absolu du cache (toujours dans le dossier data/)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_VALIDITY_HOURS = 24  # Recharger si cache > 24h

# Mapping symboles Binance -> CoinGecko IDs
COINGECKO_MAPPING = {
    'BTCUSDT': 'bitcoin',
    'ETHUSDT': 'ethereum',
    'SOLUSDT': 'solana'
}

# Mapping symboles Binance -> Yahoo Finance tickers
YAHOO_MAPPING = {
    'BTCUSDT': 'BTC-USD',
    'ETHUSDT': 'ETH-USD'
    # SOL-USD n'est pas disponible sur Yahoo Finance
}

# Mapping symboles Binance -> CryptoCompare symbols
CRYPTOCOMPARE_MAPPING = {
    'BTCUSDT': 'BTC',
    'ETHUSDT': 'ETH',
    'SOLUSDT': 'SOL'
}

def ensure_cache_dir():
    """Créer le dossier cache s'il n'existe pas"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"[INFO] Cache directory created: {CACHE_DIR}")

def get_cache_path(symbol, interval):
    """Obtenir le chemin du fichier cache"""
    # bitcoin_1d.json, ethereum_1w.json
    crypto_name = symbol.replace('USDT', '').lower()
    return os.path.join(CACHE_DIR, f"{crypto_name}_{interval}.json")

def is_cache_valid(cache_path, max_age_hours=CACHE_VALIDITY_HOURS):
    """Vérifier si le cache est encore valide"""
    if not os.path.exists(cache_path):
        return False

    # Vérifier l'âge du fichier
    file_time = os.path.getmtime(cache_path)
    age_hours = (time.time() - file_time) / 3600

    return age_hours < max_age_hours

def download_binance_data(symbol, interval='1d', limit=2000):
    """Télécharger données depuis Binance API"""
    print(f"  [>>] Downloading from Binance API: {symbol} ({interval}, {limit} candles)...")

    url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    remaining = limit

    while remaining > 0:
        batch_size = min(1000, remaining)
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': batch_size
        }

        if all_data:
            # Utiliser la dernière timestamp pour continuer
            params['endTime'] = all_data[0][0] - 1

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            batch_data = response.json()

            if not batch_data or len(batch_data) == 0:
                break

            # Ajouter au début (chronologique inverse)
            all_data = batch_data + all_data
            remaining -= len(batch_data)

            if len(batch_data) < batch_size:
                break

            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            break

    print(f"  [OK] Downloaded {len(all_data)} candles")
    return all_data

def download_coingecko_data(symbol, interval='1d', limit=2000):
    """
    Télécharger données depuis CoinGecko API (fallback avec plus d'historique)
    CoinGecko a beaucoup plus de données weekly que Binance!
    """
    if symbol not in COINGECKO_MAPPING:
        print(f"  [ERROR] No CoinGecko mapping for {symbol}")
        return None

    coin_id = COINGECKO_MAPPING[symbol]
    print(f"  [>>] Downloading from CoinGecko API: {coin_id} ({interval}, {limit} candles)...")

    # CoinGecko OHLC API
    # Pour avoir maximum de données disponibles, on utilise 'max' ou grand nombre
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc'
    params = {
        'vs_currency': 'usd',
        'days': 'max'  # Demander toutes les données disponibles
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        raw_data = response.json()

        if not raw_data:
            print(f"  [ERROR] No data received from CoinGecko")
            return None

        # Convertir format CoinGecko vers format Binance
        # CoinGecko: [[timestamp_ms, open, high, low, close], ...]
        # Binance: [[timestamp_ms, open, high, low, close, volume, ...], ...]
        binance_format = []
        for candle in raw_data:
            # CoinGecko retourne timestamp, open, high, low, close
            timestamp = candle[0]
            open_price = str(candle[1])
            high = str(candle[2])
            low = str(candle[3])
            close = str(candle[4])
            volume = "0"  # CoinGecko ne fournit pas volume dans OHLC

            # Format Binance compatible
            binance_format.append([
                timestamp,
                open_price,
                high,
                low,
                close,
                volume,
                timestamp + 86400000,  # Close time (fake)
                "0",  # Quote asset volume
                0,    # Number of trades
                "0",  # Taker buy base
                "0",  # Taker buy quote
                "0"   # Ignore
            ])

        # Filtrer selon l'intervalle demandé
        if interval == '1w':
            # Garder seulement 1 candle sur 7 (approximatif)
            filtered = []
            for i in range(0, len(binance_format), 7):
                filtered.append(binance_format[i])
            binance_format = filtered

        # Limiter au nombre demandé
        binance_format = binance_format[-limit:] if len(binance_format) > limit else binance_format

        print(f"  [OK] Downloaded {len(binance_format)} candles from CoinGecko")
        return binance_format

    except Exception as e:
        print(f"  [ERROR] CoinGecko download failed: {e}")
        return None

def download_yahoo_data(symbol, interval='1d', limit=2000):
    """
    Télécharger données depuis Yahoo Finance (yfinance)
    Parfait pour BTC/ETH avec beaucoup d'historique
    """
    if symbol not in YAHOO_MAPPING:
        print(f"  [ERROR] No Yahoo mapping for {symbol}")
        return None

    try:
        import yfinance as yf
    except ImportError:
        print(f"  [ERROR] yfinance not installed. Run: pip install yfinance")
        return None

    ticker_symbol = YAHOO_MAPPING[symbol]
    print(f"  [>>] Downloading from Yahoo Finance: {ticker_symbol} ({interval}, {limit} candles)...")

    try:
        # Télécharger avec period='max' pour historique complet
        yf_interval = '1wk' if interval == '1w' else '1d'
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period='max', interval=yf_interval)

        if df.empty:
            print(f"  [ERROR] No data received from Yahoo Finance")
            return None

        # Convertir format Yahoo -> Binance
        binance_format = []
        for index, row in df.iterrows():
            timestamp = int(index.timestamp() * 1000)
            open_price = str(row['Open'])
            high = str(row['High'])
            low = str(row['Low'])
            close = str(row['Close'])
            volume = str(row['Volume'])

            # Format Binance compatible
            binance_format.append([
                timestamp,
                open_price,
                high,
                low,
                close,
                volume,
                timestamp + 86400000,  # Close time
                "0",  # Quote asset volume
                0,    # Number of trades
                "0",  # Taker buy base
                "0",  # Taker buy quote
                "0"   # Ignore
            ])

        # Limiter au nombre demandé (les plus récents)
        binance_format = binance_format[-limit:] if len(binance_format) > limit else binance_format

        print(f"  [OK] Downloaded {len(binance_format)} candles from Yahoo Finance")
        return binance_format

    except Exception as e:
        print(f"  [ERROR] Yahoo Finance download failed: {e}")
        return None

def download_cryptocompare_data(symbol, interval='1d', limit=2000):
    """
    Télécharger données depuis CryptoCompare API (gratuit, sans API key)
    Excellent pour Solana et fallback général
    """
    if symbol not in CRYPTOCOMPARE_MAPPING:
        print(f"  [ERROR] No CryptoCompare mapping for {symbol}")
        return None

    crypto_symbol = CRYPTOCOMPARE_MAPPING[symbol]
    print(f"  [>>] Downloading from CryptoCompare API: {crypto_symbol} ({interval}, {limit} candles)...")

    # CryptoCompare API endpoint
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'

    # Pour weekly, utiliser aggregate=7
    aggregate = 7 if interval == '1w' else 1

    params = {
        'fsym': crypto_symbol,
        'tsym': 'USD',
        'limit': limit,
        'aggregate': aggregate
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()

        if result['Response'] != 'Success':
            print(f"  [ERROR] CryptoCompare API error: {result.get('Message', 'Unknown error')}")
            return None

        raw_data = result['Data']['Data']

        if not raw_data:
            print(f"  [ERROR] No data received from CryptoCompare")
            return None

        # Convertir format CryptoCompare vers format Binance
        binance_format = []
        for candle in raw_data:
            timestamp = candle['time'] * 1000  # Convert to milliseconds
            open_price = str(candle['open'])
            high = str(candle['high'])
            low = str(candle['low'])
            close = str(candle['close'])
            volume = str(candle['volumefrom'])

            # Format Binance compatible
            binance_format.append([
                timestamp,
                open_price,
                high,
                low,
                close,
                volume,
                timestamp + 86400000,  # Close time
                "0",  # Quote asset volume
                0,    # Number of trades
                "0",  # Taker buy base
                "0",  # Taker buy quote
                "0"   # Ignore
            ])

        print(f"  [OK] Downloaded {len(binance_format)} candles from CryptoCompare")
        return binance_format

    except Exception as e:
        print(f"  [ERROR] CryptoCompare download failed: {e}")
        return None

def save_to_cache(data, symbol, interval):
    """Sauvegarder données dans le cache avec métadonnées"""
    ensure_cache_dir()
    cache_path = get_cache_path(symbol, interval)

    cache_data = {
        'symbol': symbol,
        'interval': interval,
        'downloaded_at': datetime.now().isoformat(),
        'candles_count': len(data),
        'data': data
    }

    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f"  [CACHE] Saved to {cache_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to save cache: {e}")
        return False

def load_from_cache(symbol, interval):
    """Charger données depuis le cache"""
    cache_path = get_cache_path(symbol, interval)

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        print(f"  [CACHE] Loaded from {cache_path}")
        print(f"  [CACHE] {cache_data['candles_count']} candles (cached: {cache_data['downloaded_at']})")

        return cache_data['data']
    except Exception as e:
        print(f"  [ERROR] Failed to load cache: {e}")
        return None

def get_historical_data(symbol, interval='1d', limit=2000, force_download=False):
    """
    Obtenir données historiques (cache, Binance, Yahoo, CryptoCompare)

    Stratégie optimale:
    - Daily: Binance (rapide, volume précis)
    - Weekly BTC/ETH: Yahoo Finance (plus d'historique) > CryptoCompare > Binance
    - Weekly SOL: CryptoCompare (plus de données) > Binance

    Args:
        symbol: Ex: 'BTCUSDT'
        interval: '1d', '1w', etc.
        limit: Nombre de candles
        force_download: Forcer téléchargement même si cache valide

    Returns:
        List de klines (format Binance)
    """
    cache_path = get_cache_path(symbol, interval)

    # Vérifier le cache
    if not force_download and is_cache_valid(cache_path):
        data = load_from_cache(symbol, interval)
        if data and len(data) >= limit * 0.5:  # Accept cache if it has at least 50% of requested
            print(f"  [OK] Using {len(data)} candles from cache")
            return data  # Return all cached data

    # Cache invalide ou inexistant -> télécharger
    print(f"  [INFO] Cache not found or expired for {symbol} {interval}")

    data = None

    # STRATÉGIE POUR WEEKLY
    if interval == '1w':
        # BTC/ETH: Yahoo Finance en priorité (meilleur historique)
        if symbol in YAHOO_MAPPING:
            print(f"  [INFO] Trying Yahoo Finance first for {symbol} weekly...")
            data = download_yahoo_data(symbol, interval, limit)

            if data and len(data) >= limit * 0.7:
                print(f"  [SUCCESS] Using Yahoo Finance data ({len(data)} candles)")
                save_to_cache(data, symbol, interval)
                return data

        # Fallback: CryptoCompare (gratuit, sans API key)
        print(f"  [INFO] Trying CryptoCompare for {symbol} weekly...")
        data_cc = download_cryptocompare_data(symbol, interval, limit)

        if data_cc and len(data_cc) >= (len(data) if data else 0):
            print(f"  [SUCCESS] Using CryptoCompare data ({len(data_cc)} candles)")
            data = data_cc

        # Dernier fallback: Binance
        if not data or len(data) < limit * 0.5:
            print(f"  [INFO] Trying Binance as last fallback...")
            data_binance = download_binance_data(symbol, interval, limit)

            if data_binance:
                if not data or len(data_binance) > len(data):
                    print(f"  [SUCCESS] Using Binance data ({len(data_binance)} candles)")
                    data = data_binance

    # STRATÉGIE POUR DAILY
    else:
        # Daily: Binance en priorité (rapide, précis)
        data = download_binance_data(symbol, interval, limit)

        # Fallback si Binance échoue
        if not data or len(data) < limit * 0.7:
            print(f"  [INFO] Binance insufficient, trying CryptoCompare...")
            data_cc = download_cryptocompare_data(symbol, interval, limit)

            if data_cc and len(data_cc) > (len(data) if data else 0):
                print(f"  [SUCCESS] Using CryptoCompare data ({len(data_cc)} candles)")
                data = data_cc

    if data:
        # Only save if it's better than existing cache
        existing_data = load_from_cache(symbol, interval)
        if not existing_data or len(data) > len(existing_data):
            save_to_cache(data, symbol, interval)
        else:
            print(f"  [INFO] Keeping existing cache ({len(existing_data)} candles) instead of new data ({len(data)} candles)")

    return data

def update_all_caches():
    """Mettre à jour tous les caches pour BTC + ETH + SOL (1d + 1w)"""
    configs = [
        ('BTCUSDT', '1d', 3000),  # ↑ Augmenté pour plus de données
        ('BTCUSDT', '1w', 800),   # ↑ Doublé pour weekly
        ('ETHUSDT', '1d', 3000),
        ('ETHUSDT', '1w', 800),
        ('SOLUSDT', '1d', 3000),
        ('SOLUSDT', '1w', 800)
    ]

    print("\n" + "="*70)
    print("MISE À JOUR DES CACHES - Bitcoin + Ethereum + Solana")
    print("="*70 + "\n")

    for symbol, interval, limit in configs:
        print(f"Processing {symbol} {interval}...")
        get_historical_data(symbol, interval, limit, force_download=True)
        print()

    print("="*70)
    print("✅ Tous les caches mis à jour!")
    print("="*70)

def clear_cache():
    """Supprimer tous les caches"""
    if os.path.exists(CACHE_DIR):
        for file in os.listdir(CACHE_DIR):
            if file.endswith('.json'):
                os.remove(os.path.join(CACHE_DIR, file))
                print(f"[CACHE] Deleted {file}")
    print("✅ Cache cleared")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'update':
            update_all_caches()
        elif command == 'clear':
            clear_cache()
        elif command == 'test':
            # Test avec Bitcoin 1d
            data = get_historical_data('BTCUSDT', '1d', 100)
            print(f"\nTest result: {len(data)} candles loaded")
            if data:
                print(f"Latest: {datetime.fromtimestamp(data[-1][0]/1000)}")
        else:
            print("Usage:")
            print("  python data_manager.py update  # Télécharger et cacher toutes les données")
            print("  python data_manager.py clear   # Supprimer le cache")
            print("  python data_manager.py test    # Test de chargement")
    else:
        # Par défaut: mettre à jour tous les caches
        update_all_caches()
