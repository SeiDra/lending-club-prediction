# Prédiction de Défaut de Paiement — LendingClub

Projet MLOps end-to-end de prédiction du risque de crédit sur les données historiques du Lending Club.

## Structure du projet

```
lending-club-prediction/
├── .github/workflows/
│   ├── github-docker-cicd.yaml
│   └── aws.yml
├── pipeline/
│   ├── credit_app_mlops.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── test.py
│   └── model/
│       ├── best_model.pkl
│       └── features_config.json
├── model/
├── mlruns/
├── DATA/
├── du_sda_ml2_P1_import_parquet_filtre-current.ipynb
├── du_sda_ml2_P2_data_explo.ipynb
├── du_sda_ml2_P3_feature_engineering.ipynb
├── du_sda_ml2_P4_modeling_mlops.ipynb
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/SeiDra/lending-club-prediction.git
cd lending-club-prediction

python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

## Utilisation

### 1. Notebooks — dans l'ordre

| Notebook | Rôle |
|---|---|
| P1 | Import CSV → Parquet, filtrage |
| P2 | Exploration, nettoyage, variable cible |
| P3 | Feature engineering, normalisation, split |
| P4 | Modélisation (LightGBM), tracking MLflow, export des artefacts |

### 2. MLflow

```bash
mlflow ui
```
Interface disponible sur `http://127.0.0.1:5000`

### 3. Application Streamlit

Après exécution de P4 :

```bash
streamlit run pipeline/credit_app_mlops.py
```

L'app permet de saisir le profil d'un emprunteur et d'obtenir une probabilité de défaut, un score interne et une décision (approuvé / examen / refusé).

## CI/CD

Le pipeline se déclenche sur push ou pull request vers `main` :

1. **CI** : formatage (black) → lint (pylint) → tests (pytest)
2. **CD** : build image Docker → push sur Docker Hub
3. **AWS** : push sur ECR → déploiement sur ECS Fargate

Les secrets GitHub requis sont : `DOCKER_USER`, `DOCKER_PASSWORD`, `REPO_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

Pour le détail du déploiement AWS, voir `aws-setup.md`.
