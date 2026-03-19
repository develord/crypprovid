# CryptoAdviser - Projet ML Optimisé

## 🎯 Objectif
Maximiser les performances de prédiction crypto en optimisant individuellement 3 modèles ML (SNN, LSTM, XGBoost) sur Bitcoin, Ethereum et Solana, puis créer un algorithme de combinaison optimal.

## 📊 Configuration

### Cryptomonnaies (3)
- 🟠 **Bitcoin (BTC)** - BTCUSDT
- 🔵 **Ethereum (ETH)** - ETHUSDT
- 🟣 **Solana (SOL)** - SOLUSDT

### Timeframes (2)
- **1d** (Daily) - Swing trading - 7 jours lookahead
- **1w** (Weekly) - Position trading - 4 semaines lookahead

### Modèles ML (3 × 3 cryptos × 2 timeframes = 18 modèles)
1. **SNN** (Self-Normalizing Neural Network)
   - bitcoin_1d.keras, bitcoin_1w.keras
   - ethereum_1d.keras, ethereum_1w.keras
   - solana_1d.keras, solana_1w.keras

2. **LSTM** (Long Short-Term Memory)
   - bitcoin_1d_lstm.keras, bitcoin_1w_lstm.keras
   - ethereum_1d_lstm.keras, ethereum_1w_lstm.keras
   - solana_1d_lstm.keras, solana_1w_lstm.keras

3. **XGBoost** (Extreme Gradient Boosting)
   - bitcoin_1d_xgboost.pkl, bitcoin_1w_xgboost.pkl
   - ethereum_1d_xgboost.pkl, ethereum_1w_xgboost.pkl
   - solana_1d_xgboost.pkl, solana_1w_xgboost.pkl

## 📁 Structure du Projet

```
CryptoAdviser/
├── training/
│   ├── train_models.py          # Training SNN
│   ├── train_models_lstm.py     # Training LSTM
│   └── train_models_xgboost.py  # Training XGBoost
│
├── data/
│   ├── data_manager.py          # Gestionnaire de cache Binance
│   └── cache/                   # Données historiques en JSON
│       ├── btc_1d.json         (391 KB - 2000 candles)
│       ├── btc_1w.json         (80 KB - 400 candles)
│       ├── eth_1d.json         (384 KB - 2000 candles)
│       ├── eth_1w.json         (78 KB - 400 candles)
│       ├── sol_1d.json         (372 KB - 2000 candles)
│       └── sol_1w.json         (56 KB - 293 candles)
│
└── models/
    ├── snn/      (6 modèles)
    ├── lstm/     (6 modèles)
    └── xgboost/  (6 modèles)
```

## 🚀 Entraînement des Modèles

### Pré-requis
- Python 3.10+
- TensorFlow 2.x (avec GPU recommandé)
- XGBoost
- CUDA + cuDNN (pour GPU)
- requests (pour téléchargement Binance API)

### 📥 Étape 1: Télécharger les données historiques (une seule fois)

```bash
cd data
wsl -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/moham/Desktop/crypto/CryptoAdviser/data && python3 data_manager.py update"
```

Cela télécharge et cache:
- Bitcoin 1d (2000 candles, ~5.5 ans)
- Bitcoin 1w (400 candles, ~7.7 ans)
- Ethereum 1d (2000 candles, ~5.5 ans)
- Ethereum 1w (400 candles, ~7.7 ans)
- Solana 1d (2000 candles, ~5.5 ans)
- Solana 1w (293 candles, ~5.6 ans)

💡 **Avantages du cache:**
- ✓ Pas de rate limit Binance API
- ✓ Training 10x plus rapide
- ✓ Données consistantes entre entraînements
- ✓ Travail offline possible
- ✓ Auto-refresh après 24h

### 🎓 Étape 2: Entraîner les modèles

```bash
cd training

# Entraîner SNN (tous: BTC + ETH, 1d + 1w)
wsl -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/moham/Desktop/crypto/CryptoAdviser/training && python3 train_models.py"

# Entraîner LSTM
wsl -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/moham/Desktop/crypto/CryptoAdviser/training && python3 train_models_lstm.py"

# Entraîner XGBoost
wsl -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/moham/Desktop/crypto/CryptoAdviser/training && python3 train_models_xgboost.py"
```

### 🧹 Gestion du cache

```bash
cd data

# Mettre à jour le cache (forcer téléchargement)
python3 data_manager.py update

# Tester le cache
python3 data_manager.py test

# Effacer le cache
python3 data_manager.py clear
```

## 📈 Phase 1: Optimisation Individuelle

### Objectifs par modèle
- **SNN**: Maximiser précision avec régularisation optimale
- **LSTM**: Optimiser séquences temporelles et architecture
- **XGBoost**: Tuning hyperparamètres (max_depth, learning_rate, etc.)

### Métriques à optimiser
- ROI (Return on Investment)
- Win Rate (% trades gagnants)
- Sharpe Ratio
- Max Drawdown

## 🎯 Phase 2: Algorithme de Combinaison

### Stratégies envisagées
1. **Vote pondéré** - Pondération basée sur performances historiques
2. **Stacking** - Meta-model apprenant à combiner les 3 modèles
3. **Ensemble adaptatif** - Sélection dynamique selon conditions marché
4. **Moyenne bayésienne** - Poids ajustés avec incertitude

### Critères de combinaison
- Confiance de chaque modèle
- Performance récente (rolling window)
- Volatilité du marché
- Corrélation entre prédictions

## 📊 Données

**Source**: Binance API (téléchargement automatique)
- Bitcoin: BTCUSDT
- Ethereum: ETHUSDT
- Solana: SOLUSDT
- Historique: ~5-8 ans selon timeframe
- Features: 29 indicateurs techniques + Phase 1 features

## 🔧 Configuration Technique

### Features (29 total)
- RSI, MACD, Bollinger Bands
- Moyennes mobiles (SMA 7/25/99, EMA 9/21)
- ATR, Stochastic
- Volume relatif
- Prix relatifs (5 derniers)
- Phase 1: volatilité, momentum, mean reversion, etc.

### Entraînement
- Epochs: 100
- Batch size: 32
- Validation split: 20%
- Early stopping: patience 10

## 📝 Notes

- Modèles 4h supprimés (focus swing/position trading)
- Focus sur 3 cryptos: BTC + ETH + SOL
- Datasets CSV supprimés (utilise API Binance avec cache)
- Structure minimaliste pour performance maximale

## 🎯 Roadmap

- [x] Phase 0: Nettoyage et réorganisation du projet
- [ ] Phase 1.1: Optimiser SNN (Grid search hyperparamètres)
- [ ] Phase 1.2: Optimiser LSTM (Architecture, séquence length)
- [ ] Phase 1.3: Optimiser XGBoost (Feature importance, tuning)
- [ ] Phase 1.4: Backtesting complet de chaque modèle
- [ ] Phase 2.1: Design algorithme de combinaison
- [ ] Phase 2.2: Backtesting combinaison vs modèles individuels
- [ ] Phase 2.3: Déploiement en production

---

**Dernière mise à jour**: 18 Mars 2026
**Focus**: Bitcoin + Ethereum + Solana | Timeframes: 1d + 1w | Modèles: SNN + LSTM + XGBoost
