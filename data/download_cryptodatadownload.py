"""
Téléchargement des datasets de CryptoDataDownload.com
Données historiques complètes (2013+) pour BTC, ETH, SOL
"""

import os
import requests
import json
from datetime import datetime

# URLs des datasets CryptoDataDownload (Gemini Exchange - bonnes données)
DATASETS = {
    'BTC': {
        'daily': 'https://www.cryptodatadownload.com/cdd/Gemini_BTCUSD_d.csv',
        'hourly': 'https://www.cryptodatadownload.com/cdd/Gemini_BTCUSD_1h.csv'
    },
    'ETH': {
        'daily': 'https://www.cryptodatadownload.com/cdd/Gemini_ETHUSD_d.csv',
        'hourly': 'https://www.cryptodatadownload.com/cdd/Gemini_ETHUSD_1h.csv'
    }
}

# Note: Solana n'est pas sur CryptoDataDownload (trop récent)
# On garde CryptoCompare pour SOL

def download_csv_dataset(url, output_path):
    """Télécharger un dataset CSV"""

    print(f"[>>] Downloading from {url}...")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        # Sauvegarder
        with open(output_path, 'wb') as f:
            f.write(response.content)

        # Compter les lignes (enlever header)
        lines = response.text.split('\n')
        # Enlever les lignes de métadonnées (commencent souvent par #)
        data_lines = [l for l in lines if l and not l.startswith('#') and not l.startswith('Date')]

        print(f"  [OK] Downloaded {len(data_lines)} rows")
        print(f"  [OK] Saved to {output_path}")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to download: {e}")
        return False


def convert_csv_to_binance_format(csv_path, symbol):
    """Convertir CSV CryptoDataDownload vers format Binance pour compatibilité"""

    print(f"\n[>>] Converting {csv_path} to Binance format...")

    try:
        import csv
        from datetime import datetime

        binance_data = []

        with open(csv_path, 'r') as f:
            lines = f.readlines()

            # Skip first line (URL)
            if lines[0].startswith('https://'):
                lines = lines[1:]

            # Parse CSV starting from header
            reader = csv.DictReader(lines)

            for row in reader:
                # Skip empty rows or metadata
                if not row or 'date' not in row:
                    continue

                try:
                    # Utiliser le unix timestamp déjà présent
                    timestamp_str = row.get('unix', '').strip()
                    if not timestamp_str:
                        continue

                    timestamp = int(float(timestamp_str))

                    # Extraire OHLCV (colonnes en minuscules)
                    open_price = row.get('open', '0')
                    high = row.get('high', '0')
                    low = row.get('low', '0')
                    close = row.get('close', '0')
                    volume_btc = row.get('Volume BTC', '0')
                    if not volume_btc or volume_btc == '':
                        volume_btc = '0'
                    volume = volume_btc

                    # Format Binance
                    binance_data.append([
                        timestamp,
                        open_price,
                        high,
                        low,
                        close,
                        volume,
                        timestamp + 86400000,  # Close time (approximatif)
                        "0",  # Quote asset volume
                        0,    # Number of trades
                        "0",  # Taker buy base
                        "0",  # Taker buy quote
                        "0"   # Ignore
                    ])

                except Exception as e:
                    # Skip bad rows
                    continue

        if not binance_data:
            print(f"  [ERROR] No data converted")
            return None

        # Trier par timestamp (ascendant)
        binance_data.sort(key=lambda x: x[0])

        print(f"  [OK] Converted {len(binance_data)} candles")

        return binance_data

    except Exception as e:
        print(f"  [ERROR] Conversion failed: {e}")
        return None


def aggregate_to_weekly(daily_data):
    """Agréger données daily en weekly"""

    print(f"[>>] Aggregating to weekly...")

    weekly_data = []
    current_week = []

    for candle in daily_data:
        timestamp = candle[0]

        # Début de semaine (lundi)
        from datetime import datetime, timedelta
        dt = datetime.fromtimestamp(timestamp / 1000)

        # Calculer le lundi de cette semaine
        days_to_monday = (dt.weekday()) % 7
        week_start = dt - timedelta(days=days_to_monday)
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_timestamp = int(week_start.timestamp() * 1000)

        # Nouvelle semaine?
        if current_week and week_timestamp != current_week[0][0]:
            # Agréger semaine précédente
            weekly_candle = aggregate_candles(current_week)
            if weekly_candle:
                weekly_data.append(weekly_candle)
            current_week = []

        # Ajouter candle avec timestamp de début de semaine
        current_week.append([week_timestamp] + candle[1:])

    # Dernière semaine
    if current_week:
        weekly_candle = aggregate_candles(current_week)
        if weekly_candle:
            weekly_data.append(weekly_candle)

    print(f"  [OK] Aggregated to {len(weekly_data)} weekly candles")

    return weekly_data


def aggregate_candles(candles):
    """Agréger plusieurs candles en un"""

    if not candles:
        return None

    timestamp = candles[0][0]
    open_price = candles[0][1]
    high = max(float(c[2]) for c in candles)
    low = min(float(c[3]) for c in candles)
    close = candles[-1][4]
    volume = sum(float(c[5]) for c in candles)

    return [
        timestamp,
        open_price,
        str(high),
        str(low),
        close,
        str(volume),
        candles[-1][6],
        "0", 0, "0", "0", "0"
    ]


def main():
    print("\n" + "="*80)
    print("TÉLÉCHARGEMENT DATASETS - CRYPTODATADOWNLOAD.COM")
    print("="*80)

    # Créer dossier datasets
    os.makedirs('./datasets', exist_ok=True)

    results = {}

    for crypto, urls in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"{crypto} - Téléchargement")
        print('='*80)

        # Télécharger daily
        daily_csv = f'./datasets/{crypto}_daily_cdd.csv'

        if download_csv_dataset(urls['daily'], daily_csv):
            # Convertir en format Binance
            daily_data = convert_csv_to_binance_format(daily_csv, f'{crypto}USD')

            if daily_data:
                # Sauvegarder JSON
                daily_json = f'./cache/{crypto.lower()}_1d_cdd.json'
                with open(daily_json, 'w') as f:
                    json.dump({
                        'data': daily_data,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'CryptoDataDownload',
                        'count': len(daily_data)
                    }, f)

                print(f"  [OK] Saved {len(daily_data)} daily candles to {daily_json}")

                # Agréger en weekly
                weekly_data = aggregate_to_weekly(daily_data)

                if weekly_data:
                    weekly_json = f'./cache/{crypto.lower()}_1w_cdd.json'
                    with open(weekly_json, 'w') as f:
                        json.dump({
                            'data': weekly_data,
                            'timestamp': datetime.now().isoformat(),
                            'source': 'CryptoDataDownload (aggregated)',
                            'count': len(weekly_data)
                        }, f)

                    print(f"  [OK] Saved {len(weekly_data)} weekly candles to {weekly_json}")

                    results[crypto] = {
                        'daily': len(daily_data),
                        'weekly': len(weekly_data)
                    }

    # Résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ DES TÉLÉCHARGEMENTS")
    print('='*80)

    for crypto, counts in results.items():
        print(f"\n{crypto}:")
        print(f"  Daily:  {counts['daily']} candles")
        print(f"  Weekly: {counts['weekly']} candles")

    print("\n[NOTE] Solana n'est pas disponible sur CryptoDataDownload")
    print("       Utiliser CryptoCompare API pour SOL (déjà intégré)")
    print('='*80)


if __name__ == '__main__':
    main()
