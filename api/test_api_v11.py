"""
Test API with V11 Integration
==============================
Script to test the FastAPI endpoints with V11 models
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_api():
    """Test the API endpoints"""
    print("="*80)
    print("TESTING CRYPTO ADVISER API WITH V11")
    print("="*80)

    # Import after path setup
    from main import app, startup_event
    from fastapi.testclient import TestClient

    # Initialize
    print("\n[1/5] Initializing API and loading models...")
    await startup_event()

    # Create test client
    client = TestClient(app)

    # Test root endpoint
    print("\n[2/5] Testing root endpoint (/)...")
    response = client.get("/")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")

    # Test health endpoint
    print("\n[3/5] Testing health endpoint (/health)...")
    response = client.get("/health")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Status: {data['status']}")
    print(f"  Models loaded: {data['models_loaded']}")
    print(f"  Cryptos: {data['cryptos_available']}")

    # Test cryptos list
    print("\n[4/5] Testing cryptos endpoint (/api/cryptos)...")
    response = client.get("/api/cryptos")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Count: {data['count']}")
    for crypto_id, info in data['cryptos'].items():
        print(f"    - {info['name']} ({info['symbol']})")

    # Test predictions
    print("\n[5/5] Testing predictions endpoints...")

    # Test individual predictions
    for crypto in ['bitcoin', 'ethereum', 'solana']:
        print(f"\n  [{crypto.upper()}] /api/predictions/{crypto}")
        response = client.get(f"/api/predictions/{crypto}")

        if response.status_code == 200:
            result = response.json()
            print(f"    Signal: {result['signal']}")
            print(f"    Confidence: {result['confidence']:.4f}")
            print(f"    Threshold: {result['threshold']:.2f}")
            print(f"    Current Price: ${result['current_price']:.2f}")

            if result.get('risk_management'):
                rm = result['risk_management']
                print(f"    Target Price: ${rm['target_price']:.2f} (+{rm['potential_gain_percent']}%)")
                print(f"    Stop Loss: ${rm['stop_loss']:.2f} (-{rm['potential_loss_percent']}%)")
                print(f"    Risk/Reward: {rm['risk_reward_ratio']:.2f}")
        else:
            print(f"    ERROR: {response.status_code} - {response.text}")

    # Test predict all
    print(f"\n  [ALL] /api/predictions/all")
    response = client.get("/api/predictions/all")

    if response.status_code == 200:
        data = response.json()
        print(f"    Count: {data['count']}")
        for crypto_id, result in data['predictions'].items():
            if 'error' not in result:
                print(f"      {crypto_id.upper()}: {result['signal']} (conf={result['confidence']:.4f})")
            else:
                print(f"      {crypto_id.upper()}: ERROR - {result['error']}")
    else:
        print(f"    ERROR: {response.status_code} - {response.text}")

    # Test price endpoint
    print(f"\n  [PRICE] /api/price/bitcoin")
    response = client.get("/api/price/bitcoin")

    if response.status_code == 200:
        data = response.json()
        print(f"    Price: ${data['price']:.2f}")
    else:
        print(f"    ERROR: {response.status_code} - {response.text}")

    print("\n" + "="*80)
    print("API TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    asyncio.run(test_api())
