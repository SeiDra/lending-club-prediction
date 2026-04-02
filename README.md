# 🏦 Prédiction de Défaut de Paiement — LendingClub

Projet MLOps end-to-end de prédiction du risque de crédit sur les données historiques du Lending Club.

## Architecture du Projet

```
lending-club-prediction/
├── .github/
│   └── workflows/
│       ├── github-docker-cicd.yaml  ← Pipeline CI/CD (Docker Hub)
│       └── aws.yml                  ← Déploiement AWS ECS
├── pipeline/                        ← Application de déploiement
│   ├── credit_app_mlops.py          ← App Streamlit
│   ├── Dockerfile                   ← Image Docker
│   ├── requirements.txt             ← Dépendances légères (prod)
│   ├── test.py                      ← Tests unitaires (CI)
│   ├── .dockerignore
│   └── model/
│       ├── best_model.pkl           ← Modèle LightGBM (généré par P4)
│       └── features_config.json     ← Config features (généré par P4)
├── CONFIG/                          ← Configuration des notebooks
├── DATA/                            ← Données brutes (non versionnées)
├── mlruns/                          ← Expériences MLflow (non versionnées)
├── du_sda_ml2_P1_import_parquet_filtre-current.ipynb
├── du_sda_ml2_P2_data_explo.ipynb
├── du_sda_ml2_P3_feature_engineering.ipynb
├── du_sda_ml2_P4_modeling_mlops.ipynb  ← Modélisation + MLflow tracking
├── requirements.txt                 ← Dépendances complètes (dev)
└── README.md
```

## Pipeline MLOps

### 1. Expérimentation (Notebooks)
| Notebook | Rôle |
|---|---|
| P1 | Import CSV → Parquet, filtrage |
| P2 | EDA, nettoyage, variable cible |
| P3 | Feature engineering, normalisation, split |
| P4 | **MLflow** : 3 experiments (DT / LR / LightGBM), évaluation |

### 2. Tracking — MLflow
- **3 experiments** : `DecisionTree`, `LogisticRegression`, `LightGBM`
- Métriques loggées : ROC-AUC, AUPRC, Recall, F2-Score, KS Statistic
- Lancer l'UI : `mlflow ui` → http://127.0.0.1:5000

### 3. Application — Streamlit
```bash
streamlit run pipeline/credit_app_mlops.py
```

### 4. CI/CD — GitHub Actions → Docker Hub
Le pipeline se déclenche à chaque push sur `main` :
1. **CI** : Format (black) → Lint (pylint) → Tests (pytest)
2. **CD** : Build image Docker → Push sur Docker Hub

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/SeiDra/lending-club-prediction.git
cd lending-club-prediction

# 2. Créer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer les notebooks dans l'ordre P1 → P2 → P3 → P4
```

## Secrets GitHub requis (Settings → Secrets → Actions)

| Secret | Description |
|---|---|
| `DOCKER_USER` | Identifiant Docker Hub |
| `DOCKER_PASSWORD` | Token Docker Hub |
| `REPO_NAME` | Nom du repo Docker Hub (ex: `lending-club-mlops`) |
| `AWS_ACCESS_KEY_ID` | *(optionnel)* Pour déploiement AWS ECS |
| `AWS_SECRET_ACCESS_KEY` | *(optionnel)* Pour déploiement AWS ECS |
