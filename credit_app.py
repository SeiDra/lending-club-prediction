import streamlit as st
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDET BANK · Analyse Crédit",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&family=Playfair+Display:wght@500;600;700&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #f4f5f7;
    color: #1a2332;
}

.stApp { background-color: #f4f5f7; }

/* ── TOPBAR ── */
.topbar {
    background: #002855;
    padding: 0 2.5rem;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid #c8a951;
    margin: 0rem -3rem 2rem -3rem;
    position: sticky;
    top: 0;
    z-index: 999;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.topbar-logo {
    width: 36px;
    height: 36px;
    background: #c8a951;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 900;
    color: #002855;
}
.topbar-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: 0.02em;
}
.topbar-sub {
    font-size: 0.7rem;
    color: #c8a951;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    display: block;
    line-height: 1;
}
.topbar-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    color: #8ba3c1;
    font-size: 0.82rem;
}
.topbar-user {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #d4e1f0;
}
.topbar-avatar {
    width: 32px;
    height: 32px;
    background: #1a4a7a;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    color: #c8a951;
    font-weight: 700;
}

/* ── PAGE TITLE ── */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #002855;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #607d9a;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
}

/* ── BREADCRUMB ── */
.breadcrumb {
    font-size: 0.78rem;
    color: #8ba3c1;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #dde3ec;
}
.breadcrumb span { color: #c8a951; }

/* ── SECTION CARDS ── */
.section-card {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-radius: 8px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,40,85,0.05);
}
.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #002855;
    border-left: 3px solid #c8a951;
    padding-left: 10px;
    margin-bottom: 1.2rem;
}

/* ── FORM INPUTS ── */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stSlider label {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #3d5a7a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 4px !important;
}

.stTextInput input,
.stNumberInput input {
    border: 1px solid #c8d4e3 !important;
    border-radius: 4px !important;
    background: #f8fafc !important;
    color: #1a2332 !important;
    font-size: 0.92rem !important;
    padding: 0.5rem 0.8rem !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #002855 !important;
    box-shadow: 0 0 0 2px rgba(0,40,85,0.1) !important;
    background: #ffffff !important;
}

div[data-baseweb="select"] > div {
    border: 1px solid #c8d4e3 !important;
    border-radius: 4px !important;
    background: #f8fafc !important;
    font-size: 0.92rem !important;
}

.stSlider [data-testid="stTickBar"] { display: none; }
div[data-testid="stSlider"] > div > div > div {
    background: #002855 !important;
}

/* ── DIVIDERS ── */
hr { border: none; border-top: 1px solid #dde3ec; margin: 1rem 0; }

/* ── SUBMIT BUTTON ── */
.stButton > button {
    background: #002855;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 0.75rem 2.5rem;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    width: 100%;
    transition: background 0.2s;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,40,85,0.18);
}
.stButton > button:hover {
    background: #003d7a;
}

/* ── DECISION PANELS ── */
.decision-approved {
    background: #ffffff;
    border: 2px solid #1a7a4a;
    border-left: 6px solid #1a7a4a;
    border-radius: 8px;
    padding: 2rem 2rem;
    margin-top: 1.5rem;
}
.decision-rejected {
    background: #ffffff;
    border: 2px solid #b22222;
    border-left: 6px solid #b22222;
    border-radius: 8px;
    padding: 2rem 2rem;
    margin-top: 1.5rem;
}
.decision-review {
    background: #ffffff;
    border: 2px solid #b07d10;
    border-left: 6px solid #c8a951;
    border-radius: 8px;
    padding: 2rem 2rem;
    margin-top: 1.5rem;
}
.decision-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.decision-ref {
    font-size: 0.75rem;
    color: #8ba3c1;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.decision-detail {
    font-size: 0.88rem;
    color: #3d5a7a;
    line-height: 1.7;
}
.score-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 3px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.8rem;
}
.score-a { background: #d4edda; color: #1a5c2e; }
.score-b { background: #cfe3f5; color: #0b4275; }
.score-c { background: #fff3cd; color: #7a5200; }
.score-d { background: #fde8e8; color: #7a1010; }

/* ── SUMMARY BOXES ── */
.summary-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.summary-item {
    background: #f4f5f7;
    border: 1px solid #dde3ec;
    border-radius: 4px;
    padding: 0.8rem 1.2rem;
    flex: 1;
    min-width: 130px;
}
.summary-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8ba3c1;
    font-weight: 700;
    margin-bottom: 3px;
}
.summary-value {
    font-size: 1rem;
    font-weight: 700;
    color: #1a2332;
}

/* ── RISK BAR ── */
.risk-bar-container {
    background: #f4f5f7;
    border: 1px solid #dde3ec;
    border-radius: 4px;
    height: 10px;
    width: 100%;
    margin: 0.5rem 0;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* ── FOOTER ── */
.footer-bar {
    background: #002855;
    color: #8ba3c1;
    font-size: 0.72rem;
    padding: 1rem 2.5rem;
    margin: 3rem -3rem -2rem -3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 3px solid #c8a951;
}
.footer-bar a { color: #c8a951; text-decoration: none; }

/* ── REQUIRED FIELD ── */
.required { color: #b22222; font-size: 0.7rem; }

/* ── HIDE STREAMLIT DEFAULT ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding-top: 0rem !important; 
    padding-bottom: 1rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 100% !important;
}

/* ── DOSSIER NUMBER ── */
.dossier-tag {
    background: #eef3fa;
    border: 1px solid #c8d4e3;
    border-radius: 4px;
    padding: 0.4rem 0.9rem;
    font-size: 0.78rem;
    color: #3d5a7a;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-block;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE (invisible to user)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    'int_rate', 'dti', 'fico_range_low', 'annual_inc',
    'loan_amnt', 'installment', 'revol_util', 'revol_bal',
    'total_acc', 'open_acc', 'inq_last_6mths', 'delinq_2yrs',
    'pub_rec', 'emp_length_int', 'loan_to_income'
]

@st.cache_resource(show_spinner=False)
def load_scoring_engine():
    np.random.seed(42)
    n = 60000
    fico       = np.random.normal(700, 60, n).clip(580, 850)
    annual_inc = np.random.lognormal(11, 0.7, n).clip(20000, 500000)
    loan_amnt  = np.random.lognormal(9.5, 0.7, n).clip(1000, 40000)
    int_rate   = np.random.normal(14, 5, n).clip(5, 30)
    dti        = np.random.normal(18, 8, n).clip(0, 50)
    emp_length = np.random.choice(range(0, 11), n)
    revol_util = np.random.normal(55, 25, n).clip(0, 100)
    revol_bal  = np.random.lognormal(8.5, 1.2, n).clip(0, 200000)
    total_acc  = np.random.normal(25, 10, n).clip(2, 80)
    open_acc   = np.random.normal(12, 5, n).clip(1, 40)
    inq        = np.random.poisson(1.5, n).clip(0, 10)
    delinq     = np.random.poisson(0.3, n).clip(0, 5)
    pub_rec    = np.random.poisson(0.1, n).clip(0, 3)
    installment = loan_amnt * (int_rate/100/12) / (1-(1+int_rate/100/12)**-36)
    lti        = loan_amnt / annual_inc

    logit = (-5.5 + 0.08*(int_rate-14) + 0.04*(dti-18)
             - 0.015*(fico-700) - 0.0003*(annual_inc/1000-60)
             + 0.00003*loan_amnt + 0.5*lti + 0.02*revol_util
             + 0.25*inq + 0.4*delinq + 0.5*pub_rec
             + np.random.normal(0, 0.5, n))
    proba = 1/(1+np.exp(-logit))
    y = (np.random.random(n) < proba).astype(int)

    df = pd.DataFrame({
        'int_rate':int_rate,'dti':dti,'fico_range_low':fico,
        'annual_inc':annual_inc,'loan_amnt':loan_amnt,'installment':installment,
        'revol_util':revol_util,'revol_bal':revol_bal,'total_acc':total_acc,
        'open_acc':open_acc,'inq_last_6mths':inq,'delinq_2yrs':delinq,
        'pub_rec':pub_rec,'emp_length_int':emp_length,'loan_to_income':lti
    })
    X_tr, _, y_tr, _ = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
    engine = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        num_leaves=63, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1
    )
    engine.fit(X_tr, y_tr)
    return engine

def get_decision(prob, dossier_data):
    """Translate raw score into a bank decision."""
    if prob < 0.20:
        grade, label, css = "A", "ACCORD", "approved"
        msg = (f"Le dossier de {dossier_data['nom']} présente un profil financier solide. "
               f"Les indicateurs de solvabilité sont satisfaisants et le niveau d'endettement "
               f"reste maîtrisé. Le prêt peut être accordé dans les conditions demandées.")
        badge, badge_css = "Score A — Risque Très Faible", "score-a"
        conditions = f"Montant accordé : {dossier_data['montant']:,.0f} € — Durée : {dossier_data['duree']} mois — Taux nominal : {dossier_data['taux']:.2f}%"
    elif prob < 0.38:
        grade, label, css = "B", "ACCORD", "approved"
        msg = (f"Le dossier de {dossier_data['nom']} présente un profil acceptable. "
               f"La capacité de remboursement est jugée suffisante au regard du projet financé. "
               f"Le prêt est accordé sous réserve des conditions ci-dessous.")
        badge, badge_css = "Score B — Risque Faible", "score-b"
        conditions = f"Montant accordé : {dossier_data['montant']:,.0f} € — Durée : {dossier_data['duree']} mois — Taux nominal : {dossier_data['taux']:.2f}%"
    elif prob < 0.58:
        grade, label, css = "C", "INSTRUCTION COMPLÉMENTAIRE", "review"
        msg = (f"Le dossier de {dossier_data['nom']} nécessite une analyse approfondie. "
               f"Certains indicateurs appellent une vigilance particulière. "
               f"Un complément de pièces justificatives et un entretien avec le chargé de clientèle sont requis avant toute décision définitive.")
        badge, badge_css = "Score C — Analyse Requise", "score-c"
        conditions = "Dossier transmis au service des engagements pour instruction."
    else:
        grade, label, css = "D", "REFUS", "rejected"
        msg = (f"Après analyse du dossier de {dossier_data['nom']}, le niveau de risque constaté "
               f"ne permet pas d'accorder le financement demandé dans les conditions actuelles. "
               f"Le client peut se rapprocher de son conseiller pour examiner des solutions alternatives.")
        badge, badge_css = "Score D — Risque Élevé", "score-d"
        conditions = "Le client sera informé par courrier dans un délai de 72 heures ouvrées."

    return {"grade": grade, "label": label, "css": css,
            "msg": msg, "badge": badge, "badge_css": badge_css,
            "conditions": conditions}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ENGINE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Chargement du système…"):
    engine = load_scoring_engine()

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
now = datetime.datetime.now()
st.markdown(f"""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-logo">EB</div>
    <div>
      <span class="topbar-name">EDET BANK</span>
      <span class="topbar-sub">Établissement de Crédit & Solutions Financières</span>
    </div>
  </div>
  <div class="topbar-right">
    <span>{now.strftime("%A %d %B %Y · %H:%M")}</span>
    <div class="topbar-user">
      <div class="topbar-avatar">CB</div>
      <span>Conseiller</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="breadcrumb">Accueil › Crédits & Financement › <span>Nouvelle demande de prêt</span></div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Saisie d\'une Demande de Crédit</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Remplissez l\'ensemble des sections ci-dessous. Les champs marqués <span style="color:#b22222">*</span> sont obligatoires.</div>', unsafe_allow_html=True)

# Dossier number
import random
if 'dossier_num' not in st.session_state:
    st.session_state.dossier_num = f"CR-{now.year}-{random.randint(10000,99999)}"
st.markdown(f'<div class="dossier-tag">📁 N° Dossier : {st.session_state.dossier_num}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────────────────────────────────────

# ── SECTION 1 : IDENTITÉ ──────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">01 · Identité du Demandeur</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    nom = st.text_input("Nom et Prénom *", placeholder="ex. Martin Sophie")
with c2:
    email = st.text_input("Adresse e-mail", placeholder="ex. s.martin@email.fr")
with c3:
    age = st.number_input("Âge *", min_value=18, max_value=85, value=38, step=1)

c4, c5 = st.columns([2, 2])
with c4:
    situation = st.selectbox("Situation familiale *",
        ["Célibataire", "Marié(e) / Pacsé(e)", "Divorcé(e)", "Veuf/Veuve"])
with c5:
    statut_pro = st.selectbox("Statut professionnel *",
        ["Salarié CDI", "Salarié CDD", "Fonctionnaire", "Indépendant / Chef d'entreprise",
         "Profession libérale", "Retraité", "Sans emploi"])
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 2 : FINANCEMENT ───────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">02 · Objet du Financement</div>', unsafe_allow_html=True)
c6, c7, c8 = st.columns(3)
with c6:
    objet = st.selectbox("Nature du crédit *",
        ["Crédit immobilier", "Crédit à la consommation", "Rachat de crédits",
         "Prêt professionnel", "Financement automobile", "Travaux & Rénovation",
         "Voyage & Loisirs", "Autre"])
with c7:
    loan_amnt = st.number_input("Montant demandé (€) *", min_value=1000, max_value=200000,
                                 value=15000, step=500)
with c8:
    duree_opts = {12: "12 mois", 24: "24 mois", 36: "36 mois",
                  48: "48 mois", 60: "60 mois", 84: "84 mois", 120: "120 mois", 240: "240 mois"}
    duree_label = st.selectbox("Durée souhaitée *", list(duree_opts.values()), index=2)
    duree = [k for k, v in duree_opts.items() if v == duree_label][0]

c9, c10 = st.columns([2, 2])
with c9:
    int_rate = st.number_input("Taux nominal proposé (%) *", min_value=1.0, max_value=30.0,
                                value=5.90, step=0.05, format="%.2f")
with c10:
    installment = loan_amnt * (int_rate/100/12) / max(1-(1+int_rate/100/12)**-duree, 0.0001)
    st.markdown(f"""
    <div style="background:#f4f5f7;border:1px solid #dde3ec;border-radius:4px;
                padding:0.65rem 0.9rem;margin-top:1.8rem;">
      <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.1em;color:#8ba3c1;margin-bottom:2px;">Mensualité estimée</div>
      <div style="font-size:1.15rem;font-weight:700;color:#002855;">{installment:,.2f} €</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 3 : SITUATION FINANCIÈRE ─────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">03 · Situation Financière</div>', unsafe_allow_html=True)
c11, c12, c13 = st.columns(3)
with c11:
    annual_inc = st.number_input("Revenu net annuel (€) *", min_value=0, max_value=1000000,
                                  value=36000, step=1000)
with c12:
    charges = st.number_input("Charges mensuelles totales (€)", min_value=0, max_value=20000,
                               value=800, step=50)
with c13:
    emp_length = st.selectbox("Ancienneté dans l'emploi *",
        ["Moins de 1 an", "1 an", "2 ans", "3 ans", "4 ans", "5 ans",
         "6 ans", "7 ans", "8 ans", "9 ans", "10 ans et plus"])
    emp_map = {"Moins de 1 an": 0, "1 an": 1, "2 ans": 2, "3 ans": 3, "4 ans": 4,
               "5 ans": 5, "6 ans": 6, "7 ans": 7, "8 ans": 8, "9 ans": 9, "10 ans et plus": 10}
    emp_length_int = emp_map[emp_length]
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 4 : ENDETTEMENT ───────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">04 · Endettement & Encours</div>', unsafe_allow_html=True)
c14, c15, c16 = st.columns(3)
with c14:
    revol_bal = st.number_input("Encours revolving (€)", min_value=0, max_value=200000,
                                 value=5000, step=500)
with c15:
    revol_util = st.slider("Taux d'utilisation du crédit revolving (%)", 0, 100, 45)
with c16:
    dti_input = st.number_input("Taux d'endettement actuel (%)", min_value=0.0, max_value=80.0,
                                 value=15.0, step=0.5, format="%.1f")
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 5 : HISTORIQUE BANCAIRE ──────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">05 · Historique Bancaire</div>', unsafe_allow_html=True)
c17, c18, c19, c20 = st.columns(4)
with c17:
    fico = st.number_input("Score de solvabilité interne", min_value=300, max_value=850,
                            value=700, step=5)
with c18:
    total_acc = st.number_input("Nombre total de crédits (historique)", min_value=0,
                                 max_value=100, value=18, step=1)
with c19:
    open_acc = st.number_input("Crédits en cours", min_value=0, max_value=50, value=6, step=1)
with c20:
    inq = st.number_input("Consultations bureau de crédit (6 mois)", min_value=0,
                           max_value=20, value=1, step=1)

c21, c22, c23 = st.columns(3)
with c21:
    delinq = st.selectbox("Incidents de paiement (24 derniers mois)",
                           ["Aucun", "1", "2", "3 ou plus"])
    delinq_map = {"Aucun": 0, "1": 1, "2": 2, "3 ou plus": 3}
    delinq_val = delinq_map[delinq]
with c22:
    pub_rec = st.selectbox("Procédures judiciaires / surendettement",
                            ["Aucune", "1", "2 ou plus"])
    pub_map = {"Aucune": 0, "1": 1, "2 ou plus": 2}
    pub_val = pub_map[pub_rec]
with c23:
    propriete = st.selectbox("Statut logement",
                              ["Locataire", "Propriétaire", "Hébergé à titre gratuit",
                               "Propriétaire avec prêt en cours"])
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 6 : OBSERVATIONS ─────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">06 · Observations du Conseiller</div>', unsafe_allow_html=True)
observations = st.text_area("Remarques libres (facultatif)",
    placeholder="Précisions sur la situation du client, contexte particulier, pièces complémentaires demandées…",
    height=80)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SUBMIT
# ─────────────────────────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([3, 2, 3])
with col_btn2:
    submitted = st.button("⊳  ANALYSER LE DOSSIER")

# ─────────────────────────────────────────────────────────────────────────────
# DECISION
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    if not nom or not nom.strip():
        st.warning("⚠️  Veuillez saisir le nom et prénom du demandeur avant de soumettre.")
    else:
        # Build input for scoring engine
        loan_to_income = loan_amnt / max(annual_inc, 1)
        row = pd.DataFrame([{
            'int_rate': int_rate,
            'dti': dti_input,
            'fico_range_low': float(fico),
            'annual_inc': float(annual_inc),
            'loan_amnt': float(loan_amnt),
            'installment': installment,
            'revol_util': float(revol_util),
            'revol_bal': float(revol_bal),
            'total_acc': float(total_acc),
            'open_acc': float(open_acc),
            'inq_last_6mths': float(inq),
            'delinq_2yrs': float(delinq_val),
            'pub_rec': float(pub_val),
            'emp_length_int': float(emp_length_int),
            'loan_to_income': loan_to_income
        }])

        prob = engine.predict_proba(row)[0, 1]

        dossier_data = {
            "nom": nom.strip().title(),
            "montant": loan_amnt,
            "duree": duree,
            "taux": int_rate
        }
        decision = get_decision(prob, dossier_data)

        # Risk bar color
        bar_color = "#1a7a4a" if prob < 0.38 else "#c8a951" if prob < 0.58 else "#b22222"
        bar_pct = int(prob * 100)

        st.markdown(f"""
        <div class="decision-{decision['css']}">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
            <div>
              <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;
                          color:{'#1a7a4a' if decision['css']=='approved' else '#b07d10' if decision['css']=='review' else '#b22222'};
                          margin-bottom:6px;">
                {'✔  DÉCISION FAVORABLE' if decision['css']=='approved' else '⚠  DOSSIER EN INSTRUCTION' if decision['css']=='review' else '✕  DÉCISION DÉFAVORABLE'}
              </div>
              <div class="decision-header" style="color:{'#1a2332'};">
                {decision['label']}
              </div>
              <div class="decision-ref">Réf. {st.session_state.dossier_num} · {now.strftime("%d/%m/%Y à %H:%M")} · Conseiller</div>
            </div>
            <div>
              <span class="score-badge {decision['badge_css']}">{decision['badge']}</span>
            </div>
          </div>

          <div class="decision-detail">{decision['msg']}</div>

          <div class="risk-bar-container" style="margin-top:1rem;">
            <div class="risk-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#8ba3c1;margin-bottom:1rem;">
            <span>Risque minimal</span><span>Risque maximal</span>
          </div>

          <div style="background:#f8fafc;border:1px solid #dde3ec;border-radius:4px;
                      padding:0.8rem 1.2rem;font-size:0.82rem;color:#3d5a7a;margin-bottom:1.2rem;">
            <strong>Conditions :</strong> {decision['conditions']}
          </div>

          <div class="summary-row">
            <div class="summary-item">
              <div class="summary-label">Client</div>
              <div class="summary-value">{nom.strip().title()}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Montant</div>
              <div class="summary-value">{loan_amnt:,.0f} €</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Durée</div>
              <div class="summary-value">{duree} mois</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Mensualité</div>
              <div class="summary-value">{installment:,.2f} €</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Taux nominal</div>
              <div class="summary-value">{int_rate:.2f} %</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Objet</div>
              <div class="summary-value">{objet}</div>
            </div>
          </div>

          {'<div style="margin-top:1rem;padding:0.7rem 1rem;background:#fffbf0;border:1px solid #f0dfa0;border-radius:4px;font-size:0.8rem;color:#7a5200;"><strong>Observations :</strong> ' + observations + '</div>' if observations.strip() else ''}
        </div>
        """, unsafe_allow_html=True)

        # New dossier button
        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2, col_r3 = st.columns([3, 2, 3])
        with col_r2:
            if st.button("＋  Nouveau Dossier"):
                st.session_state.dossier_num = f"CR-{now.year}-{random.randint(10000,99999)}"
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
  <span>© 2025 EDET BANK — Établissement de Crédit & Solutions Financières. Tous droits réservés.</span>
  <span>Usage strictement interne · Accès réservé aux conseillers habilités · <a href="#">Politique de confidentialité</a></span>
</div>
""", unsafe_allow_html=True)
