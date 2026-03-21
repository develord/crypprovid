"""
Test V11 Prediction Service
============================
Script de test pour vérifier que le service V11 fonctionne correctement.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from predictions_v11 import PredictionServiceV11

async def test_v11():
    """Test V11 prediction service"""
    print("="*80)
    print("TEST V11 TEMPORAL PREDICTION SERVICE")
    print("="*80)

    # Initialize service
    service = PredictionServiceV11()

    # Load models
    await service.load_models()

    # Test predictions for all cryptos
    print("\n" + "="*80)
    print("TESTING PREDICTIONS")
    print("="*80)

    for crypto_id in ['bitcoin', 'ethereum', 'solana']:
        print(f"\n[{crypto_id.upper()}]")
        print("-"*80)

        try:
            result = await service.predict_one(crypto_id)

            print(f"  Signal: {result['signal']}")
            print(f"  Confidence (P(TP)): {result['confidence']:.4f}")
            print(f"  Threshold: {result['threshold']:.2f}")
            print(f"  Current Price: ${result['current_price']:.2f}")

            if result['risk_management']:
                rm = result['risk_management']
                print(f"  Target Price: ${rm['target_price']:.2f} (+{rm['potential_gain_percent']}%)")
                print(f"  Stop Loss: ${rm['stop_loss']:.2f} (-{rm['potential_loss_percent']}%)")
                print(f"  Risk/Reward: {rm['risk_reward_ratio']:.2f}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    # Test predict_all
    print("\n" + "="*80)
    print("TESTING PREDICT_ALL")
    print("="*80)

    try:
        all_predictions = await service.predict_all()

        for crypto_id, result in all_predictions.items():
            if 'error' in result:
                print(f"\n{crypto_id.upper()}: ERROR - {result['error']}")
            else:
                print(f"\n{crypto_id.upper()}: {result['signal']} (confidence={result['confidence']:.4f})")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    asyncio.run(test_v11())
