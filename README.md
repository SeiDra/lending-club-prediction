# Prédiction de Défaut de Paiement - Lending Club

Ce projet d'analyse de données et de Machine Learning a pour objectif de prédire le risque de défaut de paiement des emprunteurs en utilisant les données historiques du Lending Club.

## Structure du projet
* `DATA/` : Dossier contenant les données brutes (non incluses sur Git).
* `lending-club-loan-default-prediction-eda.ipynb` : Notebook d'analyse exploratoire (EDA) et de modélisation.
* `requirements.txt` : Liste des dépendances et bibliothèques Python nécessaires.

## Installation et Configuration

Pour reproduire ce projet sur votre machine locale, veuillez suivre les étapes ci-dessous.

### 1. Créer l'environnement virtuel
Il est fortement recommandé d'isoler les dépendances du projet. À la racine du projet, ouvrez votre terminal et exécutez :

```bash
python -m venv .venv
```

### 2. Activer l'environnement virtuel (Windows / PowerShell)
Pour activer l'environnement que vous venez de créer, lancez la commande suivante :

```Bash
.\.venv\Scripts\Activate.ps1
```

L'indicateur (.venv) devrait maintenant apparaître au début de votre ligne de commande.

### 3. Installer les dépendances

Une fois l'environnement activé, installez l'ensemble des bibliothèques requises (Pandas, Scikit-Learn, XGBoost, LightGBM, etc.) avec la commande suivante :

```Bash
pip install -r requirements.txt
```

### Utilisation
Une fois l'installation terminée, vous pouvez sélectionner le noyau (kernel) .venv dans votre éditeur (comme VS Code) et exécuter les cellules du notebook lending-club-loan-default-prediction-eda.ipynb.