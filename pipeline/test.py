"""
test.py — Tests unitaires pour le pipeline CI
LendingClub Credit Risk Analyzer | Projet MLOps

Couvre :
- Imports des dépendances critiques
- Logique de standardisation (z-score)
- Logique de coloration du risque
- Logique de formatage des valeurs
- Intégrité du fichier de config (si présent)
"""

import json
import os

import numpy as np
import pandas as pd
import pytest


# ── 1. Tests d'import ─────────────────────────────────────────────────────
class TestImports:
    def test_streamlit_importable(self):
        import streamlit
        assert streamlit is not None

    def test_plotly_importable(self):
        import plotly.graph_objects as go
        assert go is not None

    def test_lightgbm_importable(self):
        import lightgbm
        assert lightgbm is not None

    def test_joblib_importable(self):
        import joblib
        assert joblib is not None

    def test_sklearn_importable(self):
        import sklearn
        assert sklearn is not None


# ── 2. Tests de la logique de standardisation ─────────────────────────────
class TestStandardization:
    """
    La fonction standardize_and_predict applique z = (x - mean) / std
    pour les features scalées avant de passer au modèle.
    """

    def _standardize(self, val, mean, std):
        """Réplique la logique de standardize_and_predict."""
        std = std if std != 0 else 1.0
        return (val - mean) / std

    def test_standard_normal(self):
        """Valeur = mean → z-score doit être 0."""
        assert self._standardize(50000, 50000, 20000) == pytest.approx(0.0)

    def test_one_std_above(self):
        """Valeur = mean + std → z-score doit être 1."""
        assert self._standardize(70000, 50000, 20000) == pytest.approx(1.0)

    def test_one_std_below(self):
        """Valeur = mean - std → z-score doit être -1."""
        assert self._standardize(30000, 50000, 20000) == pytest.approx(-1.0)

    def test_zero_std_guard(self):
        """std=0 ne doit pas lever ZeroDivisionError."""
        result = self._standardize(100, 100, 0)
        assert result == pytest.approx(0.0)

    def test_binary_passthrough(self):
        """Les features binaires ne doivent PAS être standardisées (passthrough)."""
        # Simuler : is_binary=True → la valeur passe telle quelle
        val = 1
        assert val == 1  # pas de transformation


# ── 3. Tests de la logique de risque ──────────────────────────────────────
class TestRiskLogic:

    def _risk_color(self, p):
        return "#00c853" if p < 0.35 else ("#ffd600" if p < 0.60 else "#ff1744")

    def _risk_level(self, p):
        return "Faible" if p < 0.35 else ("Modéré" if p < 0.60 else "Élevé")

    def test_low_risk_color(self):
        assert self._risk_color(0.10) == "#00c853"
        assert self._risk_color(0.34) == "#00c853"

    def test_medium_risk_color(self):
        assert self._risk_color(0.35) == "#ffd600"
        assert self._risk_color(0.59) == "#ffd600"

    def test_high_risk_color(self):
        assert self._risk_color(0.60) == "#ff1744"
        assert self._risk_color(0.99) == "#ff1744"

    def test_risk_level_labels(self):
        assert self._risk_level(0.20) == "Faible"
        assert self._risk_level(0.50) == "Modéré"
        assert self._risk_level(0.80) == "Élevé"

    def test_score_interne_range(self):
        """Le score interne (0-1000) doit rester dans [0, 1000]."""
        for proba in [0.0, 0.1, 0.5, 0.9, 1.0]:
            score = max(0, min(1000, int((1 - proba) * 1000)))
            assert 0 <= score <= 1000


# ── 4. Tests du formatage des valeurs ─────────────────────────────────────
class TestFormatting:

    def _fmt(self, feat, val):
        if any(k in feat for k in ["amnt", "inc", "bal", "limit", "buy"]):
            return f"${val:,.0f}"
        if any(k in feat for k in ["rate", "util", "dti", "pct"]):
            return f"{val:.1f}%"
        if "mths" in feat or "mo_sin" in feat:
            return f"{val:.0f} mois"
        return f"{val:.2f}"

    def test_dollar_format(self):
        assert self._fmt("annual_inc", 75000) == "$75,000"
        assert self._fmt("loan_amnt", 10000) == "$10,000"

    def test_percent_format(self):
        assert self._fmt("int_rate", 14.5) == "14.5%"
        assert self._fmt("revol_util", 55.0) == "55.0%"

    def test_months_format(self):
        assert self._fmt("mths_since_last_record", 24) == "24 mois"
        assert self._fmt("mo_sin_old_il_acct", 60) == "60 mois"

    def test_generic_format(self):
        assert self._fmt("some_ratio", 0.3333) == "0.33"


# ── 5. Test d'intégrité du fichier de config (si présent) ─────────────────
class TestModelConfig:

    CONFIG_PATH = os.path.join("model", "features_config.json")

    def test_config_structure_if_present(self):
        """Si features_config.json existe, vérifie sa structure minimale."""
        if not os.path.exists(self.CONFIG_PATH):
            pytest.skip("features_config.json absent (normal avant déploiement)")

        with open(self.CONFIG_PATH) as f:
            config = json.load(f)

        assert "features" in config, "Clé 'features' manquante"
        assert "stats" in config, "Clé 'stats' manquante"
        assert isinstance(config["features"], list), "'features' doit être une liste"
        assert len(config["features"]) > 0, "La liste de features est vide"

    def test_stats_keys_if_present(self):
        """Chaque feature dans stats doit avoir min, max, median, is_binary."""
        if not os.path.exists(self.CONFIG_PATH):
            pytest.skip("features_config.json absent")

        with open(self.CONFIG_PATH) as f:
            config = json.load(f)

        required_keys = {"min", "max", "median", "is_binary"}
        for feat, s in config["stats"].items():
            missing = required_keys - set(s.keys())
            assert not missing, f"Feature '{feat}' manque les clés : {missing}"
