# ── Image de base légère Python 3.10 ──────────────────────────────────────
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="SeiDra"
LABEL project="lending-club-mlops"
LABEL description="Credit Risk Analyzer — LendingClub Streamlit App"

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copier d'abord les dépendances (optimise le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application (app + model/)
COPY . .

# Port exposé par Streamlit
EXPOSE 8501

# Healthcheck pour vérifier que l'app répond
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Commande de lancement
CMD ["streamlit", "run", "credit_app_mlops.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
