# Crypto Predictions API

API REST FastAPI pour les prédictions de cryptomonnaies avec modèles XGBoost V6.

## Features

- ✅ Prédictions en temps réel pour 8 cryptos majeures
- ✅ Documentation Swagger automatique
- ✅ Support CORS pour apps mobiles
- ✅ Modèles XGBoost V6 (57 features)
- ✅ Probabilités de confiance
- ✅ Prix en temps réel depuis Binance

## Installation

```bash
cd api
pip install -r requirements.txt
```

## Démarrage

```bash
# Mode développement (avec auto-reload)
python main.py

# Ou avec uvicorn directement
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Documentation

Une fois le serveur démarré :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

## Endpoints

### GET /health
Health check du serveur
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00",
  "models_loaded": 8,
  "cryptos_available": ["bitcoin", "ethereum", ...]
}
```

### GET /api/cryptos
Liste des cryptos supportées

### GET /api/predictions/all
Toutes les prédictions en une seule requête

### GET /api/predictions/{crypto}
Prédiction pour une crypto spécifique

**Exemple** : `GET /api/predictions/bitcoin`
```json
{
  "crypto": "bitcoin",
  "symbol": "BTCUSDT",
  "name": "Bitcoin",
  "signal": "BUY",
  "confidence": 0.65,
  "probabilities": {
    "buy": 0.65,
    "sell": 0.15,
    "hold": 0.20
  },
  "current_price": 45000.50,
  "timestamp": "2025-01-01T12:00:00"
}
```

### GET /api/price/{crypto}
Prix actuel d'une crypto

## Cryptos Supportées

1. Bitcoin (BTC) - `bitcoin`
2. Ethereum (ETH) - `ethereum`
3. BNB (BNB) - `bnb`
4. XRP (XRP) - `xrp`
5. Cardano (ADA) - `cardano`
6. Avalanche (AVAX) - `avalanche`
7. Polkadot (DOT) - `polkadot`
8. Solana (SOL) - `solana`

## Configuration

Créer un fichier `.env` (optionnel) :
```env
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here
```

## Tests

```bash
# Test health check
curl http://localhost:8000/health

# Test prediction
curl http://localhost:8000/api/predictions/bitcoin

# Test all predictions
curl http://localhost:8000/api/predictions/all
```

## Déploiement

### Docker (Recommandé)
```bash
docker build -t crypto-api .
docker run -p 8000:8000 crypto-api
```

### Heroku
```bash
heroku create crypto-predictions-api
git push heroku main
```

### Railway / Render
- Connect GitHub repo
- Auto-deploy on push

## Structure

```
api/
├── main.py           # Serveur FastAPI principal
├── config.py         # Configuration
├── models.py         # Modèles Pydantic
├── predictions.py    # Service de prédictions
├── requirements.txt  # Dépendances Python
└── README.md         # Ce fichier
```

## Support

Pour toute question, consulter la documentation Swagger à `/docs`.
