"""
credit_app_mlops.py — Application Streamlit Prédiction Risque de Crédit
LendingClub | Projet MLOps
Correction v2 : les sliders affichent les valeurs REELLES (dollars, %, etc.)
La standardisation z=(x-mean)/std est appliquée AVANT la prédiction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#070d1a;}
[data-testid="stSidebar"]{background:#0d1526;border-right:1px solid #1e2d4a;}
[data-testid="stSidebar"] .stMarkdown h3{color:#4fc3f7;font-size:.75rem;font-weight:600;
  letter-spacing:.12em;text-transform:uppercase;margin-top:1.5rem;padding-bottom:.4rem;
  border-bottom:1px solid #1e2d4a;}
[data-testid="stSlider"] label{font-size:.82rem;color:#8899bb;font-family:'DM Mono',monospace;}
.mcard{background:#0d1526;border:1px solid #1e2d4a;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:.75rem;}
.mcard .lbl{font-size:.72rem;color:#5577aa;letter-spacing:.1em;text-transform:uppercase;font-weight:500;}
.mcard .val{font-size:2rem;font-weight:600;margin-top:.2rem;font-family:'DM Mono',monospace;}
.ba{background:#0a2e1a;border:1.5px solid #00c853;color:#00c853;padding:.6rem 1.5rem;border-radius:50px;font-weight:600;font-size:1rem;display:inline-block;}
.bw{background:#2e2200;border:1.5px solid #ffd600;color:#ffd600;padding:.6rem 1.5rem;border-radius:50px;font-weight:600;font-size:1rem;display:inline-block;}
.br{background:#2e0a0a;border:1.5px solid #ff1744;color:#ff1744;padding:.6rem 1.5rem;border-radius:50px;font-weight:600;font-size:1rem;display:inline-block;}
.main-title{font-size:2rem;font-weight:600;color:#e8f0fe;letter-spacing:-.02em;}
.main-sub{font-size:.9rem;color:#5577aa;margin-top:.25rem;}
hr{border:none;border-top:1px solid #1e2d4a;margin:1.5rem 0;}
.fr{display:flex;align-items:center;margin-bottom:.55rem;}
.fn{font-family:'DM Mono',monospace;font-size:.78rem;color:#8899bb;width:200px;flex-shrink:0;}
.fb{flex-grow:1;background:#1e2d4a;border-radius:3px;height:6px;overflow:hidden;}
.ff{height:100%;border-radius:3px;background:linear-gradient(90deg,#1565c0,#4fc3f7);}
.fv{font-family:'DM Mono',monospace;font-size:.75rem;color:#5577aa;width:60px;text-align:right;flex-shrink:0;}
.stButton>button{background:linear-gradient(135deg,#1565c0,#0288d1);color:white;border:none;
  border-radius:8px;padding:.6rem 2rem;font-weight:600;font-size:.95rem;width:100%;}
.stButton>button:hover{opacity:.88;color:white;}
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "model"

@st.cache_resource(show_spinner="Chargement du modèle…")
def load_artifacts():
    mp, cp = os.path.join(MODEL_DIR,"best_model.pkl"), os.path.join(MODEL_DIR,"features_config.json")
    if not os.path.exists(mp) or not os.path.exists(cp): return None, None
    return joblib.load(mp), json.load(open(cp))

model, config = load_artifacts()

LABELS = {
    "int_rate":"Taux d'intérêt (%)","dti":"Ratio dettes/revenus (DTI)","annual_inc":"Revenu annuel ($)",
    "loan_amnt":"Montant du prêt ($)","installment":"Mensualité ($)","revol_bal":"Solde revolving ($)",
    "revol_util":"Utilisation revolving (%)","loan_to_income":"Ratio prêt/revenu",
    "installment_to_income":"Ratio mensualité/revenu","fico_avg":"Score FICO moyen",
    "emp_length_yrs":"Ancienneté (ans)","open_acc":"Comptes ouverts","total_acc":"Total comptes",
    "all_util":"Utilisation globale (%)","il_util":"Util. prêts tempérament (%)","bc_util":"Util. bancaire (%)",
    "total_bc_limit":"Limite crédit bancaire ($)","total_derog_signals":"Signaux négatifs (0-4)",
    "mths_since_last_record":"Mois depuis incident","mths_since_recent_bc_dlq":"Mois depuis défaut BC",
    "mths_since_recent_bc":"Mois depuis BC récente","pct_tl_nvr_dlq":"% comptes sans défaut",
    "acc_open_past_24mths":"Comptes ouverts (24 mois)","mo_sin_old_il_acct":"Âge + ancien compte (mois)",
    "mo_sin_old_rev_tl_op":"Âge + ancienne ligne rev. (mois)","credit_age_years":"Âge crédit (ans)",
    "bc_open_to_buy":"Crédit bancaire disponible ($)","has_delinq":"Retards passés",
    "has_pub_rec":"Registre public","has_bankruptcy":"Faillite déclarée","term_ 60 months":"Durée 60 mois",
}
lbl = lambda f: LABELS.get(f, f.replace("_"," ").title())

def standardize_and_predict(raw, features, stats):
    X = {}
    for f in features:
        s = stats.get(f, {}); v = raw.get(f, s.get("median",0.0))
        if s.get("is_scaled") and not s.get("is_binary"):
            std = s.get("std",1.0) or 1.0
            X[f] = (v - s.get("mean",0.0)) / std
        else:
            X[f] = v
    df = pd.DataFrame([X])
    return float(model.predict_proba(df)[0,1]), int(model.predict(df)[0])

def fmt(f, v, stats):
    s = stats.get(f,{})
    if s.get("is_binary"): return "Oui" if v==1 else "Non"
    if any(k in f for k in ["amnt","inc","bal","limit","buy"]): return f"${v:,.0f}"
    if any(k in f for k in ["rate","util","dti","pct"]): return f"{v:.1f}%"
    if "mths" in f or "mo_sin" in f: return f"{v:.0f} mois"
    return f"{v:.2f}"

PROFILES = {
    "— Saisie manuelle —": None,
    "✅ Profil Faible Risque": {
        "int_rate":7.5,"dti":8.0,"annual_inc":110000,"loan_amnt":8000,"installment":250,
        "revol_bal":5000,"revol_util":15.0,"fico_avg":760,"emp_length_yrs":8,
        "loan_to_income":0.07,"installment_to_income":0.027,"all_util":18.0,"bc_util":12.0,
        "total_bc_limit":60000,"has_delinq":0,"term_ 60 months":0,"total_derog_signals":0,
        "pct_tl_nvr_dlq":98.0,"mo_sin_old_il_acct":120,"mo_sin_old_rev_tl_op":180,
        "bc_open_to_buy":45000,"acc_open_past_24mths":2,
    },
    "⚠️  Profil Risque Modéré": {
        "int_rate":14.5,"dti":22.0,"annual_inc":55000,"loan_amnt":18000,"installment":480,
        "revol_bal":18000,"revol_util":55.0,"fico_avg":685,"emp_length_yrs":3,
        "loan_to_income":0.33,"installment_to_income":0.105,"all_util":58.0,"bc_util":52.0,
        "total_bc_limit":25000,"has_delinq":0,"term_ 60 months":1,"total_derog_signals":1,
        "pct_tl_nvr_dlq":80.0,"mo_sin_old_il_acct":60,"mo_sin_old_rev_tl_op":84,
        "bc_open_to_buy":8000,"acc_open_past_24mths":5,
    },
    "❌ Profil Haut Risque": {
        "int_rate":24.0,"dti":38.0,"annual_inc":32000,"loan_amnt":30000,"installment":850,
        "revol_bal":42000,"revol_util":88.0,"fico_avg":625,"emp_length_yrs":1,
        "loan_to_income":0.94,"installment_to_income":0.319,"all_util":85.0,"bc_util":90.0,
        "total_bc_limit":10000,"has_delinq":1,"term_ 60 months":1,"total_derog_signals":3,
        "pct_tl_nvr_dlq":45.0,"mo_sin_old_il_acct":18,"mo_sin_old_rev_tl_op":24,
        "bc_open_to_buy":500,"acc_open_past_24mths":9,
    },
}

GROUPS = {
    "💰 Informations du Prêt":["loan_amnt","int_rate","installment","term_ 60 months"],
    "👤 Profil Emprunteur":["annual_inc","emp_length_yrs","dti","fico_avg","revol_bal","revol_util"],
    "📊 Ratios & Utilisation":["loan_to_income","installment_to_income","all_util","il_util",
                                "bc_util","total_bc_limit","bc_open_to_buy"],
    "⚠️ Historique Crédit":["has_delinq","has_pub_rec","has_bankruptcy","total_derog_signals",
                             "pct_tl_nvr_dlq","mths_since_last_record","mths_since_recent_bc_dlq",
                             "mths_since_recent_bc","acc_open_past_24mths","mo_sin_old_il_acct",
                             "mo_sin_old_rev_tl_op","credit_age_years"],
    "🗂️ Autres":[],
}

with st.sidebar:
    st.markdown("## 🏦 Credit Risk Analyzer")
    st.markdown("<div style='color:#5577aa;font-size:.8rem'>LendingClub · Modèle LightGBM</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    sel_prof = st.selectbox("Profil de démonstration", list(PROFILES.keys()))
    prof_vals = PROFILES[sel_prof]
    st.markdown("<hr/>", unsafe_allow_html=True)
    input_values = {}
    if model is None or config is None:
        st.error("⚠️ Modèle introuvable. Voir instructions ci-dessous.")
    else:
        features, stats = config["features"], config["stats"]
        grouped, assigned = {k:[] for k in GROUPS}, set()
        for g, gf in GROUPS.items():
            for f in gf:
                if f in features and f not in assigned:
                    grouped[g].append(f); assigned.add(f)
        grouped["🗂️ Autres"] = [f for f in features if f not in assigned]
        for gname, gfeats in grouped.items():
            if not gfeats: continue
            st.markdown(f"### {gname}")
            for feat in gfeats:
                if feat not in stats: continue
                s = stats[feat]; is_bin = s.get("is_binary", False)
                fmin, fmax, fmed = float(s["min"]), float(s["max"]), float(s["median"])
                default = float(np.clip(float(prof_vals.get(feat,fmed)) if prof_vals else fmed, fmin, fmax))
                if is_bin:
                    val = st.selectbox(lbl(feat), [0,1], index=int(round(default)),
                                       format_func=lambda x:"Oui (1)" if x else "Non (0)", key=f"feat_{feat}")
                else:
                    r = fmax - fmin
                    step, fmtstr = (float(round(r/200)),"%.0f") if r>10000 else ((round(r/200,1),"%.1f") if r>100 else (round(r/200,3),"%.3f"))
                    val = st.slider(lbl(feat), min_value=fmin, max_value=fmax,
                                    value=default, step=step, format=fmtstr, key=f"feat_{feat}")
                input_values[feat] = val
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.button("🔍 Analyser le Risque", use_container_width=True)

col_t, _ = st.columns([5,1])
with col_t:
    st.markdown('<div class="main-title">Analyse de Risque de Crédit</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">LendingClub · Prédiction de défaut · LightGBM optimisé via MLflow</div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

if model is None or config is None:
    st.warning("""
### ⚙️ Configuration requise
**1. Fin de P3** — ajouter :
```python
import joblib, os; os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")
```
**2. Fin de P4** — exécuter la cellule `snippet_P4_save_artifacts.py`  
**3.** Relancer : `streamlit run credit_app_mlops.py`
""")
    st.stop()

rc = lambda p: "#00c853" if p<.35 else ("#ffd600" if p<.60 else "#ff1744")
badge = lambda p: ('<span class="ba">✅ APPROUVÉ</span>' if p<.35 else
                   ('<span class="bw">⚠️ EXAMEN REQUIS</span>' if p<.60 else '<span class="br">❌ REFUSÉ</span>'))
dtxt = lambda p: ("Profil à faible risque. Prêt recommandé." if p<.35 else
                  ("Profil à risque modéré. Vérification manuelle conseillée." if p<.60 else
                   "Profil à risque élevé de défaut. Prêt déconseillé."))

proba, pred = standardize_and_predict(input_values, config["features"], config["stats"]) if input_values else (0.0,0)
c = rc(proba)

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="mcard"><div class="lbl">Probabilité de Défaut</div><div class="val" style="color:{c}">{proba:.1%}</div></div>', unsafe_allow_html=True)
with c2:
    score = max(0, min(1000, int((1-proba)*1000)))
    st.markdown(f'<div class="mcard"><div class="lbl">Score Interne (0–1000)</div><div class="val" style="color:{c}">{score}</div></div>', unsafe_allow_html=True)
with c3:
    niv = "Faible" if proba<.35 else ("Modéré" if proba<.60 else "Élevé")
    st.markdown(f'<div class="mcard"><div class="lbl">Niveau de Risque</div><div class="val" style="color:{c};font-size:1.4rem">{niv}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="mcard"><div class="lbl">Décision</div><div style="margin-top:.6rem">{badge(proba)}</div></div>', unsafe_allow_html=True)

st.markdown("")
cg, ci = st.columns(2)
with cg:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=proba*100,
        number={"suffix":"%","font":{"size":38,"color":c,"family":"DM Mono"}},
        gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#1e2d4a","tickfont":{"color":"#5577aa","size":11}},
               "bar":{"color":c,"thickness":.25},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
               "steps":[{"range":[0,35],"color":"#0a2e1a"},{"range":[35,60],"color":"#2e2200"},{"range":[60,100],"color":"#2e0a0a"}],
               "threshold":{"line":{"color":c,"width":3},"thickness":.8,"value":proba*100}},
        title={"text":"Risque de Défaut","font":{"size":14,"color":"#5577aa"}},
        domain={"x":[0,1],"y":[0,1]}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      height=280,margin=dict(l=20,r=20,t=40,b=10),font_color="#e8f0fe")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<p style='text-align:center;color:#8899bb;font-size:.85rem'>{dtxt(proba)}</p>", unsafe_allow_html=True)

with ci:
    st.markdown("<p style='color:#5577aa;font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600'>Top 10 — Importance des Features</p>", unsafe_allow_html=True)
    try:
        imps = pd.Series(model.feature_importances_, index=config["features"]).sort_values(ascending=False).head(10)
        imax = imps.max()
        for f, imp in imps.items():
            bp = int((imp/imax)*100) if imax>0 else 0
            st.markdown(f'<div class="fr"><div class="fn">{lbl(f)[:24]}</div><div class="fb"><div class="ff" style="width:{bp}%"></div></div><div class="fv">{imp:,.0f}</div></div>', unsafe_allow_html=True)
    except: st.info("Importances non disponibles.")

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<p style='color:#5577aa;font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600'>Valeurs du Profil Analysé</p>", unsafe_allow_html=True)
if input_values:
    items = [(lbl(f), fmt(f, v, config["stats"])) for f,v in input_values.items()]
    cols = st.columns(3); chunk = max(1,len(items)//3+1)
    for i, col in enumerate(cols):
        with col:
            for l, vs in items[i*chunk:(i+1)*chunk]:
                st.markdown(f"<div style='display:flex;justify-content:space-between;border-bottom:1px solid #1e2d4a;padding:.3rem 0;'><span style='color:#5577aa;font-size:.78rem'>{l}</span><span style='color:#c8d8f0;font-family:DM Mono,monospace;font-size:.78rem'>{vs}</span></div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#2a3a5a;font-size:.75rem'>Projet MLOps — LendingClub · LightGBM · MLflow · Streamlit</p>", unsafe_allow_html=True)
