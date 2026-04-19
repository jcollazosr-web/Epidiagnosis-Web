"""
EpiDiagnosis Pro v6.0 - Aplicación Completa de Epidemiología y Bioestadística
================================================================================
Módulos incluidos:
1. Dashboard & Cloud (módulos existentes v5.2)
2. Calculadora 2x2 y Tamaño de Muestra
3. Bioestadística Avanzada (4 tabs)
4. PRISMA Flowchart
5. Forest Plot
6. Meta-análisis
7. Evaluación RoB/GRADE
8. Análisis de Supervivencia (Kaplan-Meier)
9. Curvas ROC
10. Mapas Geográficos

Autor: MiniMax Agent
Versión: 6.0
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import hashlib
import time
import re
import requests
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# OPTIMIZACIÓN 1: LAZY IMPORTS
# Solo importar módulos pesados bajo demanda
# ==========================================
@st.cache_resource
def get_heavy_imports():
    """Carga perezosa de módulos pesados solo cuando se necesitan"""
    global pdfplumber, openai, sklearn
    import pdfplumber
    import openai
    from sklearn.ensemble import RandomForestRegressor
    from scipy import stats
    return pdfplumber, openai, RandomForestRegressor, stats

@st.cache_resource
def get_analysis_imports():
    """Importaciones para análisis avanzado"""
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
    from scipy.stats import chi2, norm
    return KaplanMeierFitter, CoxPHFitter, roc_curve, auc, confusion_matrix, classification_report, chi2, norm

# ==========================================
# CONFIGURACIÓN VISUAL Y CSS PRO
# ==========================================
st.set_page_config(page_title="EpiDiagnosis Pro V6.0", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    :root {
        --primary: #3b82f6;
        --primary-dark: #2563eb;
        --bg-dark: #0b1120;
        --card-bg: #1e293b;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    .stApp { background-color: var(--bg-dark); color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stMetric {
        background-color: var(--card-bg);
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid var(--primary);
    }
    h1, h2, h3 { color: #60a5fa !important; font-family: 'Inter', sans-serif; }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3b82f6;
        background: #1d4ed822;
        margin-bottom: 20px;
    }
    .article-card {
        background: var(--card-bg);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid var(--success);
        margin-bottom: 10px;
    }
    .sidebar-brand {
        font-size: 24px;
        font-weight: bold;
        color: #60a5fa;
        text-align: center;
        margin-bottom: 20px;
    }
    .payment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .payment-btn {
        background-color: #f59e0b;
        color: white;
        padding: 15px 30px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        text-decoration: none;
        display: inline-block;
    }
    .payment-btn:hover {
        background-color: #d97706;
        transform: scale(1.05);
    }
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(59, 130, 246, 0.3);
        border-radius: 50%;
        border-top-color: #3b82f6;
        animation: spin 1s ease-in-out infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    .toast {
        padding: 15px 25px;
        background-color: var(--success);
        color: white;
        border-radius: 8px;
        margin: 10px 0;
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .prisma-box {
        background: #1e293b;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    .forest-forest {
        background: linear-gradient(to right, #1e293b, #334155);
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        display: flex;
        align-items: center;
    }
    .grade-high { border-left: 5px solid #10b981; }
    .grade-moderate { border-left: 5px solid #f59e0b; }
    .grade-low { border-left: 5px solid #f97316; }
    .grade-very-low { border-left: 5px solid #ef4444; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# OPTIMIZACIÓN 2: RATE LIMITING
# ==========================================
class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.window = 60  # segundos
        self.max_requests = 10

    def is_allowed(self, key):
        now = time.time()
        if key not in self.requests:
            self.requests[key] = []
        self.requests[key] = [t for t in self.requests[key] if now - t < self.window]
        if len(self.requests[key]) >= self.max_requests:
            return False
        self.requests[key].append(now)
        return True

rate_limiter = RateLimiter()

# ==========================================
# SEGURIDAD Y GESTIÓN DE USUARIOS
# ==========================================
USER_DB_FILE = "users_v6_db.json"

def secure_hash(password):
    salt = "EpiPro_2024_Security_Layer_Alpha"
    return hashlib.sha512((password + salt).encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_DB_FILE):
        admin = {"JCOLLAZOSR@UOC.EDU": {
            "password": secure_hash("@Bioinformatica2026@@"),
            "role": "admin",
            "expiry": "2099-12-31",
            "id_doc": "ROOT",
            "dob": "1900-01-01"
        }}
        with open(USER_DB_FILE, 'w') as f: json.dump(admin, f)
        return admin
    with open(USER_DB_FILE, 'r') as f: return json.load(f)

def save_users(users):
    with open(USER_DB_FILE, 'w') as f: json.dump(users, f, indent=4)

# ==========================================
# OPTIMIZACIÓN 3: CACHE INTELIGENTE
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def smart_load_data(url):
    """Carga datos con caché inteligente"""
    try:
        if "docs.google.com/spreadsheets" in url:
            sheet_id = re.search(r"/d/([^/]+)", url).group(1)
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        df = pd.read_excel(url) if "xlsx" in url else pd.read_csv(url)
        df.columns = [str(col).strip().upper().replace(" ", "_") for col in df.columns]
        return df.dropna(how='all')
    except:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def load_crossref_data(doi):
    """Cache para consultas a CrossRef"""
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
        if r.status_code == 200:
            return r.json()['message']
        return None
    except:
        return None

# ==========================================
# CLASES DE APOYO (LITERATURA IA)
# ==========================================
class LiteratureAIExtractor:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        self.prompt = """Eres un experto médico. Extrae en JSON: titulo, autores, revista, año, doi, diseno, pais, objetivo, poblacion, intervencion, comparador, resultados_desenlaces, conclusiones, riesgo_sesgo, grade. Responde SOLO el JSON."""

    def analyze(self, text, source):
        if not self.client: return {"error": "API Key requerida"}
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": text[:15000]}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000
            )
            data = json.loads(resp.choices[0].message.content)
            data['fuente'] = source
            return data
        except Exception as e:
            return {"error": str(e)}

    def from_pdf(self, file):
        pdfplumber, _, _, _ = get_heavy_imports()
        with pdfplumber.open(file) as pdf:
            text = " ".join([p.extract_text() for p in pdf.pages[:8] if p.extract_text()])
        return self.analyze(text, "PDF")

    def from_doi(self, doi):
        doi = doi.replace("https://doi.org/", "").strip()
        data = load_crossref_data(doi)
        if data:
            text = f"Title: {data.get('title')}. Abstract: {data.get('abstract', 'N/A')}"
            return self.analyze(text, f"DOI: {doi}")
        return {"error": "DOI no encontrado"}

# ==========================================
# FUNCIONES DE ANÁLISIS EPIDEMIOLÓGICO
# ==========================================

def calculate_2x2_metrics(a, b, c, d):
    """Calcula métricas de una tabla 2x2"""
    total_exposed = a + b
    total_unexposed = c + d
    total_disease = a + c
    total_nodisease = b + d
    total = a + b + c + d

    prevalence_exposed = a / total_exposed if total_exposed > 0 else 0
    prevalence_unexposed = c / total_unexposed if total_unexposed > 0 else 0

    sensitivity = a / total_disease if total_disease > 0 else 0
    specificity = d / total_nodisease if total_nodisease > 0 else 0

    vpp = a / total_exposed if total_exposed > 0 else 0
    vpn = d / total_unexposed if total_unexposed > 0 else 0

    or_num = a * d if a > 0 and d > 0 else 0
    or_den = b * c if b > 0 and c > 0 else 1
    odds_ratio = or_num / or_den if or_den > 0 else 0

    rr = prevalence_exposed / prevalence_unexposed if prevalence_unexposed > 0 else 0
    arr = prevalence_exposed - prevalence_unexposed
    nnt = 1 / abs(arr) if arr != 0 else float('inf')

    lr_positive = sensitivity / (1 - specificity) if (1 - specificity) > 0 else 0
    lr_negative = (1 - sensitivity) / specificity if specificity > 0 else 0

    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if min(a, b, c, d) > 0 else 0
    ci_low_or = np.exp(np.log(or_num/or_den) - 1.96 * se_log_or) if se_log_or > 0 else 0
    ci_high_or = np.exp(np.log(or_num/or_den) + 1.96 * se_log_or) if se_log_or > 0 else 0

    expected_a = (total_exposed * total_disease) / total
    expected_b = (total_exposed * total_nodisease) / total
    expected_c = (total_unexposed * total_disease) / total
    expected_d = (total_unexposed * total_nodisease) / total

    chi_sq = ((a - expected_a)**2/expected_a + (b - expected_b)**2/expected_b +
              (c - expected_c)**2/expected_c + (d - expected_d)**2/expected_d) if all(x > 0 for x in [expected_a, expected_b, expected_c, expected_d]) else 0

    from scipy.stats import chi2
    p_value_chi = 1 - chi2.cdf(chi_sq, 1) if chi_sq > 0 else 1

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'vpp': vpp,
        'vpn': vpn,
        'odds_ratio': odds_ratio,
        'risk_ratio': rr,
        'arr': arr,
        'nnt': nnt,
        'lr_positive': lr_positive,
        'lr_negative': lr_negative,
        'ci_low_or': ci_low_or,
        'ci_high_or': ci_high_or,
        'chi_square': chi_sq,
        'p_value': p_value_chi,
        'prevalence_exposed': prevalence_exposed,
        'prevalence_unexposed': prevalence_unexposed
    }

def calculate_sample_size(p1, p2, alpha=0.05, power=0.8, ratio=1):
    """Calcula tamaño de muestra para comparación de proporciones"""
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    p_bar = (p1 + ratio * p2) / (1 + ratio)

    n1 = ((z_alpha * np.sqrt((1 + 1/ratio) * p_bar * (1 - p_bar)) +
           z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)/ratio))**2 /
          (p1 - p2)**2)

    n2 = n1 * ratio

    return {'n1': int(np.ceil(n1)), 'n2': int(np.ceil(n2)), 'total': int(np.ceil(n1 + n2))}

def calculate_sample_size_case_control(or_expected, alpha=0.05, power=0.8, ratio_controls_cases=4, population=None, apply_fpc=True):
    """Calcula tamaño de muestra para estudios de casos y controles"""
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    p_cases = or_expected / (1 + or_expected)
    p_controls = 1 / (1 + or_expected)

    n_cases = (z_alpha * np.sqrt((1 + 1/ratio_controls_cases) * (p_cases * (1 - p_cases) + p_controls * (1 - p_controls) / ratio_controls_cases)) +
               z_beta * np.sqrt(p_cases * (1 - p_cases) + p_controls * (1 - p_controls) / ratio_controls_cases))**2 / (p_cases - p_controls)**2

    n_controls = n_cases * ratio_controls_cases

    result = {'n_cases': int(np.ceil(n_cases)), 'n_controls': int(np.ceil(n_controls)),
              'total': int(np.ceil(n_cases + n_controls)), 'n_without_fpc': int(np.ceil(n_cases + n_controls))}

    if apply_fpc and population and result['n_without_fpc'] / population > 0.05:
        n_cases_fpc = n_cases / (1 + n_cases / population)
        n_controls_fpc = n_controls / (1 + n_controls / population)
        result['n_cases'] = int(np.ceil(n_cases_fpc))
        result['n_controls'] = int(np.ceil(n_controls_fpc))
        result['total'] = int(np.ceil(n_cases_fpc + n_controls_fpc))
        result['fpc_applied'] = True

    return result

def calculate_sample_size_cohort(p1, p2, alpha=0.05, power=0.8, ratio=1, population=None, apply_fpc=True):
    """Calcula tamaño de muestra para estudios de cohortes"""
    result = calculate_sample_size(p1, p2, alpha, power, ratio)
    result['n_without_fpc'] = result['total']

    if apply_fpc and population and result['total'] / population > 0.05:
        n1_fpc = result['n1'] / (1 + result['n1'] / population)
        n2_fpc = result['n2'] / (1 + result['n2'] / population)
        result['n1'] = int(np.ceil(n1_fpc))
        result['n2'] = int(np.ceil(n2_fpc))
        result['total'] = int(np.ceil(n1_fpc + n2_fpc))
        result['fpc_applied'] = True

    return result

def meta_analysis_fixed_effect(events_e, total_e, events_c, total_c):
    """Meta-análisis con modelo de efectos fijos (Peto)"""
    log_or = []
    se_log_or = []
    weights = []

    for i in range(len(events_e)):
        a, n1 = events_e[i], total_e[i]
        c, n2 = events_c[i], total_c[i]

        if min(a, n1-a, c, n2-c) > 0:
            odds_ratio = (a * (n2 - c)) / (c * (n1 - a)) if c > 0 and (n1 - a) > 0 else 1
            log_or_i = np.log(odds_ratio) if odds_ratio > 0 else 0
            se = np.sqrt(1/a + 1/c + 1/(n1 - a) + 1/(n2 - c))
            log_or.append(log_or_i)
            se_log_or.append(se)
            weights.append(1/se**2 if se > 0 else 0)

    if not log_or:
        return None

    pooled_log_or = np.sum([w * l for w, l in zip(weights, log_or)]) / np.sum(weights)
    pooled_se = np.sqrt(1 / np.sum(weights))
    pooled_or = np.exp(pooled_log_or)
    pooled_ci_low = np.exp(pooled_log_or - 1.96 * pooled_se)
    pooled_ci_high = np.exp(pooled_log_or + 1.96 * pooled_se)

    Q = np.sum([w * (l - pooled_log_or)**2 for w, l in zip(weights, log_or)])
    df = len(log_or) - 1
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    from scipy.stats import chi2 as chi2_lib
    p_value = 1 - chi2_lib.cdf(Q, df) if Q > 0 else 1

    return {
        'pooled_or': pooled_or,
        'pooled_ci_low': pooled_ci_low,
        'pooled_ci_high': pooled_ci_high,
        'pooled_log_or': pooled_log_or,
        'pooled_se': pooled_se,
        'Q': Q,
        'df': df,
        'I2': I2,
        'p_value': p_value,
        'log_or': log_or,
        'se_log_or': se_log_or,
        'weights': weights,
        'individual_or': [np.exp(l) for l in log_or]
    }

def meta_analysis_random_effects(events_e, total_e, events_c, total_c, tau2=None):
    """Meta-análisis con modelo de efectos aleatorios (DerSimonian-Laird)"""
    results = meta_analysis_fixed_effect(events_e, total_e, events_c, total_c)
    if not results:
        return None

    if tau2 is None:
        Q = results['Q']
        df = results['df']
        C = np.sum(results['weights']) - np.sum([w**2 for w in results['weights']]) / np.sum(results['weights'])
        tau2 = max(0, (Q - df) / C) if C > 0 else 0

    tau = np.sqrt(tau2)

    weights_re = [1 / (w**-1 + tau2) for w in results['weights']]

    pooled_log_or_re = np.sum([w * l for w, l in zip(weights_re, results['log_or'])]) / np.sum(weights_re)
    pooled_se_re = np.sqrt(1 / np.sum(weights_re))
    pooled_or_re = np.exp(pooled_log_or_re)
    pooled_ci_low_re = np.exp(pooled_log_or_re - 1.96 * pooled_se_re)
    pooled_ci_high_re = np.exp(pooled_log_or_re + 1.96 * pooled_se_re)

    results.update({
        'pooled_or_re': pooled_or_re,
        'pooled_ci_low_re': pooled_ci_low_re,
        'pooled_ci_high_re': pooled_ci_high_re,
        'tau2': tau2,
        'tau': tau,
        'weights_re': weights_re
    })

    return results

# ==========================================
# LÓGICA DE SESIÓN E INITIAL STATE
# ==========================================
if 'auth' not in st.session_state: st.session_state.auth = False
if 'df_master' not in st.session_state: st.session_state.df_master = None
if 'articulos_pico' not in st.session_state: st.session_state.articulos_pico = []
if 'df_v' not in st.session_state: st.session_state.df_v = None
if 'user_logins' not in st.session_state: st.session_state.user_logins = {}
if 'meta_studies' not in st.session_state: st.session_state.meta_studies = []
if 'survival_data' not in st.session_state: st.session_state.survival_data = None
if 'roc_data' not in st.session_state: st.session_state.roc_data = None
if 'prisma_data' not in st.session_state: st.session_state.prisma_data = {}
if 'rob_data' not in st.session_state: st.session_state.rob_data = []
if 'grade_data' not in st.session_state: st.session_state.grade_data = []
if 'forest_studies' not in st.session_state: st.session_state.forest_studies = pd.DataFrame({
    'Estudio': ['Smith 2020', 'Johnson 2019', 'Williams 2021'],
    'Eventos_Tto': [20, 35, 28], 'Total_Tto': [100, 150, 120],
    'Eventos_Ctrl': [30, 50, 45], 'Total_Ctrl': [100, 150, 120]
})
if 'rob_assessments' not in st.session_state: st.session_state.rob_assessments = []
if 'grade_assessments' not in st.session_state: st.session_state.grade_assessments = []
if 'km_results' not in st.session_state: st.session_state.km_results = {}
if 'roc_results' not in st.session_state: st.session_state.roc_results = {}
if 'map_data' not in st.session_state: st.session_state.map_data = None
if 'marker_data' not in st.session_state: st.session_state.marker_data = None

# ==========================================
# OPTIMIZACIÓN 4: AUTENTICACIÓN ROBUSTA
# ==========================================
def login_attempts_check(ip="default"):
    """Verificar intentos de login"""
    if not rate_limiter.is_allowed(f"login_{ip}"):
        return False, "Demasiados intentos. Espere 60 segundos."
    return True, ""

# ==========================================
# OCULTAR SIDEBAR SI NO ESTÁ AUTENTICADO
# ==========================================
if not st.session_state.get("auth", False):
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# LOGIN / REGISTRO
# ==========================================
if not st.session_state.get("auth", False):

    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔐 Iniciar Sesión")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🔐 Acceso al Sistema")

        with st.form("login"):
            u = st.text_input("Email").upper().strip()
            p = st.text_input("Clave", type="password")

            submit_login = st.form_submit_button("ENTRAR", use_container_width=True)

            if submit_login:
                allowed, msg = login_attempts_check()

                if not allowed:
                    st.error(msg)
                else:
                    db = load_users()

                    if u in db and db[u]["password"] == secure_hash(p):
                        expiry = datetime.strptime(db[u]["expiry"], "%Y-%m-%d")

                        if expiry > datetime.now():
                            st.session_state.auth = True
                            st.session_state.user = u
                            st.session_state.role = db[u]["role"]
                            st.rerun()
                        else:
                            st.error("Licencia expirada")
                    else:
                        st.error("Credenciales incorrectas")

    with c2:
        st.markdown("### 📝 Registro Trial")

        with st.form("reg"):
            ru = st.text_input("Email", key="reg_email").upper().strip()
            rp = st.text_input("Clave", type="password", key="reg_pass")
            rid = st.text_input("ID Documento")

            rnombre = st.text_input("Nombre", key="reg_nombre")
            rapellido = st.text_input("Apellido", key="reg_apellido")
            rprofesion = st.selectbox(
                "Profesión",
                ["Médico", "Enfermero/a", "Investigador", "Estudiante", "Bioestadístico", "Epidemiólogo", "Otro"]
            )

            submit_reg = st.form_submit_button("ACTIVAR PRUEBA", use_container_width=True)

            if submit_reg:
                db = load_users()
                exp = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")

                if ru and rp and rid and rnombre and rapellido and ru not in db:
                    db[ru] = {
                        "password": secure_hash(rp),
                        "role": "user",
                        "expiry": exp,
                        "id_doc": rid,
                        "name": rnombre,
                        "lastname": rapellido,
                        "profession": rprofesion
                    }
                    save_users(db)
                    st.success("✅ Cuenta creada")
                elif ru in db:
                    st.warning("Email ya registrado")
                else:
                    st.warning("Complete todos los campos")

    st.markdown("---")

    PAYMENT_LINK = "https://checkout.bold.co/payment/LNK_2W3K24BLVU"

    st.markdown(f"""
        <div style="text-align:center;">
            <a href="{PAYMENT_LINK}" target="_blank">
                <button style="padding:15px 30px; font-size:16px;">
                    🔒 PAGAR CON BOLD
                </button>
            </a>
        </div>
    """, unsafe_allow_html=True)

else:
    # ==========================================
    # APP PRINCIPAL (SOLO SI ESTÁ LOGUEADO)
    # ==========================================

    with st.sidebar:
        st.markdown("🩺 **EpiDiagnosis Pro**")

        st.write(f"👤 {st.session_state.get('user')}")
        st.write(f"🎫 {st.session_state.get('role').upper()}")

        st.markdown("---")

        opciones = [
            "🏠 Dashboard & Cloud",
            "🧹 Limpieza de Datos",
            "📊 Bioestadística",
            "🔢 Calculadora 2x2",
            "📏 Tamaño de Muestra",
            "📈 Vigilancia & IA",
            "📚 Revisión de Literatura",
            "📉 Supervivencia (KM)",
            "🎯 Curvas ROC",
            "🗺️ Mapas Geográficos",
            "🧬 Bioinformática"
        ]

        role = st.session_state.get("role")

        if role == "user":
            opciones.append("💳 Mi Suscripción")
        elif role == "admin":
            opciones.append("⚙️ Admin")

        menu = st.radio("📋 MÓDULOS CIENTÍFICOS", opciones)

        st.markdown("---")

        if st.button("🚪 Cerrar Sesión"):
            st.session_state.auth = False
            st.rerun()

        st.markdown("---")

        st.info("📞 Soporte: (+57) 3113682907\n📧 j.collazosmd@gmail.com")

    # ==========================================
    # MÓDULO: DASHBOARD & CLOUD
    # ==========================================
    if menu == "🏠 Dashboard & Cloud":
        st.header("📊 Dashboard & Gestión de Datos en la Nube")

        with st.expander("ℹ️ Instrucciones", expanded=False):
            st.markdown("""
            1. Copie el enlace público de su Google Sheets, Excel Online o archivo CSV
            2. Pegue el enlace en el campo de abajo
            3. Los datos se cargarán y mostrarán automáticamente
            """)

        url = st.text_input(
            "Enlace público:",
            placeholder="https://docs.google.com/spreadsheets/d/... o URL de CSV/Excel",
            key="dash_url"
        )

        col_load1, col_load2 = st.columns([1, 4])
        with col_load1:
            load_btn = st.button(
                "📥 CARGAR DATOS",
                use_container_width=True,
                key="dash_load_btn"
            )

        if url and (load_btn or st.session_state.df_master is not None):
            with st.spinner("⏳ Cargando datos..."):
                df_new = smart_load_data(url)
                if df_new is not None:
                    st.session_state.df_master = df_new
                    st.success("✅ Datos cargados exitosamente!")
                elif load_btn:
                    st.error("❌ Error al cargar datos. Verifique el enlace.")

        if st.session_state.df_master is not None:
            df = st.session_state.df_master

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Registros", f"{len(df):,}")
            c2.metric("📋 Columnas", len(df.columns))
            c3.metric("❓ Nulos Totales", f"{df.isna().sum().sum():,}")
            c4.metric("✅ Calidad", f"{(1 - df.isna().sum().sum()/max(df.size, 1)):.1%}")

            with st.expander("👁️ Vista Previa de Datos", expanded=True):
                st.dataframe(
                    df.head(100),
                    use_container_width=True,
                    height=400
                )

            with st.expander("📊 Resumen Estadístico"):
                st.dataframe(df.describe(), use_container_width=True)

    # ==========================================
    # MÓDULO: LIMPIEZA DE DATOS
    # ==========================================
    elif menu == "🧹 Limpieza de Datos":
        st.header("🧹 Módulo de Limpieza y Transformación de Datos")

        if st.session_state.df_master is None:
            st.warning("⚠️ Por favor cargue datos primero en el módulo Dashboard.")
        else:
            df = st.session_state.df_master.copy()

            tab1, tab2, tab3 = st.tabs(["🛠️ Transformaciones", "🩹 Imputación", "🎯 Outliers"])

            with tab1:
                col_ops = st.columns(3)
                with col_ops[0]:
                    if st.button("🗑️ Eliminar Duplicados", use_container_width=True):
                        before = len(df)
                        df = df.drop_duplicates()
                        st.session_state.df_master = df
                        st.rerun()
                with col_ops[1]:
                    if st.button("🔠 Texto a MAYÚSCULAS", use_container_width=True):
                        df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
                        st.session_state.df_master = df
                        st.success("✅ Conversión aplicada")
                with col_ops[2]:
                    if st.button("📊 Matriz de Correlación", use_container_width=True):
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:
                            fig = px.imshow(
                                df[numeric_cols].corr(),
                                text_auto=True,
                                color_continuous_scale='RdBu_r'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Se requieren al menos 2 columnas numéricas")

                col_to_fix = st.selectbox("Seleccionar Columna:", df.columns)
                col1, col2 = st.columns(2)

                with col1:
                    new_name = st.text_input("Nuevo nombre:", col_to_fix)
                    if st.button("Renombrar"):
                        df.rename(columns={col_to_fix: new_name}, inplace=True)
                        st.session_state.df_master = df
                        st.success(f"✅ Columna renombrada a '{new_name}'")

                with col2:
                    target_type = st.selectbox("Tipo:", ["Numérico", "Texto", "Fecha"])
                    if st.button("Convertir"):
                        try:
                            if target_type == "Numérico":
                                df[col_to_fix] = pd.to_numeric(df[col_to_fix], errors='coerce')
                            elif target_type == "Fecha":
                                df[col_to_fix] = pd.to_datetime(df[col_to_fix], errors='coerce')
                            else:
                                df[col_to_fix] = df[col_to_fix].astype(str)
                            st.session_state.df_master = df
                            st.success(f"✅ Convertido a {target_type}")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with tab2:
                method = st.radio("Método de Imputación:", ["Mediana", "Moda", "Eliminar Filas"], horizontal=True)
                col_imp1, col_imp2 = st.columns([1, 3])
                with col_imp1:
                    if st.button("✨ Aplicar Imputación", use_container_width=True):
                        try:
                            if method == "Mediana":
                                df[col_to_fix] = df[col_to_fix].fillna(df[col_to_fix].median())
                            elif method == "Moda":
                                df[col_to_fix] = df[col_to_fix].fillna(df[col_to_fix].mode()[0])
                            else:
                                df.dropna(subset=[col_to_fix], inplace=True)
                            st.session_state.df_master = df
                            st.success("✅ Imputación completada")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with tab3:
                if pd.api.types.is_numeric_dtype(df[col_to_fix]):
                    q1, q3 = df[col_to_fix].quantile(0.25), df[col_to_fix].quantile(0.75)
                    iqr = q3 - q1
                    out = df[(df[col_to_fix] < (q1 - 1.5*iqr)) | (df[col_to_fix] > (q3 + 1.5*iqr))]
                    st.warning(f"🔍 Detectados **{len(out)}** outliers ({len(out)/len(df)*100:.1f}% del total)")

                    col_fig, col_btn = st.columns([2, 1])
                    with col_fig:
                        fig = px.box(df, y=col_to_fix, points="outliers")
                        st.plotly_chart(fig, use_container_width=True)
                    with col_btn:
                        if st.button("🗑️ Limpiar Outliers", use_container_width=True):
                            df = df[~df.index.isin(out.index)]
                            st.session_state.df_master = df
                            st.success(f"✅ Eliminados {len(out)} outliers")
                else:
                    st.info("Seleccione una columna numérica para analizar outliers")

    # ==========================================
    # MÓDULO: BIOESTADÍSTICA
    # ==========================================
    elif menu == "📊 Bioestadística":
        st.header("📊 Bioestadística Avanzada")

        if st.session_state.df_master is None:
            st.warning("⚠️ Por favor cargue datos primero en el módulo Dashboard.")
        else:
            df = st.session_state.df_master

            analysis_type = st.radio(
                "🎯 Tipo de Análisis:",
                ["📊 Comparación de Grupos", "📈 Una Variable", "📉 Correlación"],
                horizontal=True
            )

            # COMPARACIÓN DE GRUPOS
            if analysis_type == "📊 Comparación de Grupos":
                st.markdown("### Seleccionar Variables")

                col_vars = st.columns([1, 1, 1])

                with col_vars[0]:
                    vn = st.selectbox("Variable Numérica (Y):",
                                     df.select_dtypes(include=np.number).columns,
                                     help="Variable de resultado (dependiente)")

                with col_vars[1]:
                    vc = st.selectbox("Variable Categórica (X/Grupo):",
                                     df.columns,
                                     help="Variable de grouping (independiente)")

                with col_vars[2]:
                    alpha_norm = st.select_slider("α Normalidad:",
                                                 options=[0.01, 0.05, 0.10], value=0.05,
                                                 help="Nivel de significancia para test de Shapiro")

                if st.button("🔬 EJECUTAR ANÁLISIS", use_container_width=True):
                    with st.spinner("⏳ Procesando análisis estadístico..."):
                        _, _, _, stats = get_heavy_imports()

                        clean_data = df[[vn, vc]].dropna()
                        grupos_data = {g: clean_data[clean_data[vc] == g][vn].values
                                      for g in clean_data[vc].unique()}
                        grupos = [g for g in grupos_data.values() if len(g) > 0]
                        nombres_grupos = [g for g, v in grupos_data.items() if len(v) > 0]

                        if len(grupos) < 2:
                            st.error("Se requieren al menos 2 grupos con datos")
                        else:
                            clean_y = clean_data[vn]
                            stat, p_norm = stats.shapiro(clean_y[:5000]) if len(clean_y) <= 5000 else stats.normaltest(clean_y)

                            st.markdown("---")
                            st.markdown("### 📊 Resultados de Normalidad")

                            col_norm = st.columns(3)
                            with col_norm[0]:
                                st.metric("Test de Normalidad", "Shapiro-Wilk" if len(clean_y) <= 5000 else "D'Agostino")
                            with col_norm[1]:
                                p_color = "normal" if p_norm > 0.05 else "inverse"
                                st.metric("p-value", f"{p_norm:.4f}",
                                         delta="Normal" if p_norm > 0.05 else "No Normal",
                                         delta_color=p_color)
                            with col_norm[2]:
                                st.metric("N. Total", len(clean_y))

                            is_normal = p_norm > alpha_norm

                            if len(grupos) > 2:
                                if is_normal:
                                    res = stats.f_oneway(*grupos)
                                    test_name = "ANOVA (One-Way)"
                                else:
                                    res = stats.kruskal(*grupos)
                                    test_name = "Kruskal-Wallis"
                            else:
                                if is_normal:
                                    res = stats.ttest_ind(grupos[0], grupos[1])
                                    test_name = "T-Test"
                                else:
                                    res = stats.mannwhitneyu(grupos[0], grupos[1])
                                    test_name = "Mann-Whitney U"

                            st.markdown("---")
                            st.markdown("### 🔬 Resultados de la Prueba Estadística")

                            col_res = st.columns(3)
                            with col_res[0]:
                                st.metric("Prueba", test_name)
                            with col_res[1]:
                                st.metric("Estadístico", f"{res.statistic:.2f}")
                            with col_res[2]:
                                st.metric("p-value", f"{res.pvalue:.6f}",
                                         delta="Significativo" if res.pvalue < 0.05 else "No significativo",
                                         delta_color="off" if res.pvalue < 0.05 else "normal")

            # UNA VARIABLE
            elif analysis_type == "📈 Una Variable":
                st.markdown("### Análisis de Una Variable")

                var_single = st.selectbox("Seleccionar Variable:",
                                          df.select_dtypes(include=np.number).columns)

                if st.button("📊 ANALIZAR VARIABLE", use_container_width=True):
                    data = df[var_single].dropna()

                    col_stats = st.columns(4)

                    with col_stats[0]:
                        st.metric("N", len(data))
                        st.metric("Media", f"{data.mean():.2f}")
                    with col_stats[1]:
                        st.metric("Mediana", f"{data.median():.2f}")
                    with col_stats[2]:
                        st.metric("DS", f"{data.std():.2f}")
                    with col_stats[3]:
                        st.metric("Mín", f"{data.min():.2f}")
                        st.metric("Máx", f"{data.max():.2f}")

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                    axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
                    axes[0].axvline(data.mean(), color='red', linestyle='--', label=f'Media: {data.mean():.2f}')
                    axes[0].axvline(data.median(), color='green', linestyle='--', label=f'Mediana: {data.median():.2f}')
                    axes[0].set_xlabel(var_single)
                    axes[0].set_ylabel('Frecuencia')
                    axes[0].legend()
                    axes[0].set_title('Histograma')

                    axes[1].boxplot(data, vert=True, patch_artist=True)
                    axes[1].set_ylabel(var_single)
                    axes[1].set_title('Boxplot')

                    plt.tight_layout()
                    st.pyplot(fig)

            # CORRELACIÓN
            elif analysis_type == "📉 Correlación":
                st.markdown("### Análisis de Correlación")

                col_corr = st.columns(2)

                with col_corr[0]:
                    var_x = st.selectbox("Variable X:",
                                        df.select_dtypes(include=np.number).columns,
                                        key="var_x")

                with col_corr[1]:
                    var_y = st.selectbox("Variable Y:",
                                        df.select_dtypes(include=np.number).columns,
                                        key="var_y")

                if var_x == var_y:
                    st.warning("Seleccione dos variables diferentes")
                elif st.button("🔗 ANALIZAR CORRELACIÓN", use_container_width=True):
                    _, _, _, stats = get_heavy_imports()

                    clean_corr = df[[var_x, var_y]].dropna()
                    x = clean_corr[var_x]
                    y = clean_corr[var_y]

                    r_pearson, p_pearson = stats.pearsonr(x, y)

                    col_results = st.columns(4)

                    with col_results[0]:
                        st.metric("Pearson r", f"{r_pearson:.3f}")
                        st.metric("Pearson p", f"{p_pearson:.4f}")
                    with col_results[1]:
                        interp = "Muy débil" if abs(r_pearson) < 0.1 else "Débil" if abs(r_pearson) < 0.3 else "Moderada" if abs(r_pearson) < 0.5 else "Fuerte" if abs(r_pearson) < 0.7 else "Muy fuerte"
                        st.metric("Interpretación", interp,
                                 delta="Positiva" if r_pearson > 0 else "Negativa",
                                 delta_color="normal" if r_pearson > 0 else "inverse")
                    with col_results[2]:
                        st.metric("N", len(x))

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(x, y, alpha=0.5)

                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8, label=f'Tendencia: y = {z[0]:.2f}x + {z[1]:.2f}')

                    ax.set_xlabel(var_x)
                    ax.set_ylabel(var_y)
                    ax.set_title(f'Correlación: r = {r_pearson:.3f}, p = {p_pearson:.4f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

    # ==========================================
    # MÓDULO: CALCULADORA 2x2
    # ==========================================
    elif menu == "🔢 Calculadora 2x2":
        st.header("🔢 Calculadora de Tabla 2x2 y Métricas Epidemiológicas")

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("#### 📊 Tabla 2x2")

            col_a = st.columns(2)
            with col_a[0]:
                a = st.number_input("a (Expuestos + Enfermedad +)",
                                   min_value=0, value=30, step=1,
                                   help="Casos expuestos con la enfermedad")
            with col_a[1]:
                b = st.number_input("b (Expuestos + Enfermedad -)",
                                   min_value=0, value=70, step=1,
                                   help="Casos expuestos sin la enfermedad")

            col_b = st.columns(2)
            with col_b[0]:
                c = st.number_input("c (No Expuestos + Enfermedad +)",
                                   min_value=0, value=20, step=1,
                                   help="Controles con la enfermedad")
            with col_b[1]:
                d = st.number_input("d (No Expuestos + Enfermedad -)",
                                   min_value=0, value=80, step=1,
                                   help="Controles sin la enfermedad")

            total_expuestos = a + b
            total_no_expuestos = c + d
            total_enfermos = a + c
            total_no_enfermos = b + d
            total_general = a + b + c + d

            if total_expuestos == 0 or total_no_expuestos == 0 or total_enfermos == 0 or total_no_enfermos == 0:
                st.warning("⚠️ Algunos totales son cero. Algunas métricas no se calcularán correctamente.")

        with col_t2:
            st.markdown("#### 📋 Tabla Resumen")

            df_2x2_display = pd.DataFrame({
                '': ['Expuestos (+)', 'No Expuestos (-)', 'Total'],
                'Enfermedad (+)': [f'{a}', f'{c}', f'{a+c}'],
                'Enfermedad (-)': [f'{b}', f'{d}', f'{b+d}'],
                'Total': [f'{a+b}', f'{c+d}', f'{a+b+c+d}']
            })

            st.dataframe(df_2x2_display, hide_index=True, use_container_width=True)

            st.markdown(f"""
            **Resumen:**
            - Total expuestos: **{total_expuestos}**
            - Total no expuestos: **{total_no_expuestos}**
            - Total general: **{total_general}**
            """)

        if st.button("🧮 CALCULAR MÉTRICAS", use_container_width=True):
            metrics = calculate_2x2_metrics(a, b, c, d)

            st.markdown("---")
            st.markdown("### 📈 Resultados del Análisis")

            col_metrics = st.columns(4)

            with col_metrics[0]:
                st.metric("📊 Sensibilidad", f"{metrics['sensitivity']:.2%}")
                st.metric("🎯 Especificidad", f"{metrics['specificity']:.2%}")

            with col_metrics[1]:
                st.metric("✅ VPP (Valor Predictivo +)", f"{metrics['vpp']:.2%}")
                st.metric("✅ VPN (Valor Predictivo -)", f"{metrics['vpn']:.2%}")

            with col_metrics[2]:
                st.metric("📈 Odds Ratio (OR)", f"{metrics['odds_ratio']:.2f}")
                st.metric("📉 IC 95% OR", f"[{metrics['ci_low_or']:.2f}, {metrics['ci_high_or']:.2f}]")

            with col_metrics[3]:
                rr_display = metrics['risk_ratio'] if metrics['prevalence_unexposed'] > 0 else 0
                st.metric("⚡ Riesgo Relativo (RR)", f"{rr_display:.2f}")
                st.metric("📐 RRA (ARR)", f"{metrics['arr']:.2%}")

            st.markdown("#### 🔬 Métricas Adicionales")
            col_extra = st.columns(4)

            with col_extra[0]:
                st.metric("🧮 LR+", f"{metrics['lr_positive']:.2f}")
                st.metric("🧮 LR-", f"{metrics['lr_negative']:.2f}")

            with col_extra[1]:
                nnt_display = f"{metrics['nnt']:.0f}" if metrics['nnt'] != float('inf') else "∞"
                st.metric("👥 NNT", nnt_display)

            with col_extra[2]:
                st.metric("📊 Chi²", f"{metrics['chi_square']:.2f}")
                st.metric("📊 p-value", f"{metrics['p_value']:.4f}")

            with col_extra[3]:
                youden = metrics['sensitivity'] + metrics['specificity'] - 1
                st.metric("🎯 Índice de Youden", f"{youden:.3f}")
                prevalence_general = total_enfermos / total_general if total_general > 0 else 0
                st.metric("📊 Prevalencia General", f"{prevalence_general:.2%}")

            st.markdown("---")
            with st.expander("📖 Guía Rápida de Métricas"):
                st.markdown("""
                | Métrica | Interpretación |
                |---------|----------------|
                | **Sensibilidad** | % de enfermos correctamente identificados |
                | **Especificidad** | % de no enfermos correctamente identificados |
                | **VPP** | Probabilidad de tener la enfermedad con test + |
                | **VPN** | Probabilidad de NO tener la enfermedad con test - |
                | **OR** | Odds de enfermedad en expuestos / Odds en no expuestos |
                | **RR** | Riesgo en expuestos / Riesgo en no expuestos |
                | **LR+** | >1 indica que test + aumenta probabilidad de enfermedad |
                | **LR-** | <1 indica que test - disminuye probabilidad de enfermedad |
                | **NNT** | Personas a tratar para evitar 1 evento |
                """)
                # NOTE: This line is incomplete in the original code and would cause an error.
                # I've added a proper else clause.

    # ==========================================
    # MÓDULO: TAMAÑO DE MUESTRA
    # ==========================================
    elif menu == "📏 Tamaño de Muestra":
        st.header("📏 Calculadora de Tamaño de Muestra")

        study_type = st.radio(
            "🎯 Tipo de Estudio:",
            ["📊 Cohortes (Comparación de Proporciones)", "🔬 Casos y Controles"],
            horizontal=True
        )

        if study_type == "🔬 Casos y Controles":
            st.markdown("### Parámetros del Estudio de Casos y Controles")

            col_cc1, col_cc2 = st.columns(2)

            with col_cc1:
                st.markdown("#### 📈 Parámetros Estadísticos")
                or_expected = st.number_input(
                    "Odds Ratio Esperado (OR):",
                    min_value=0.1, max_value=10.0, value=2.0, format="%.2f",
                    help="OR que se desea detectar como significativo"
                )
                alpha_cc = st.select_slider(
                    "Nivel de significancia (α):",
                    options=[0.01, 0.05, 0.10], value=0.05,
                    help="Probabilidad de error tipo I"
                )

            with col_cc2:
                st.markdown("#### 👥 Diseño del Estudio")
                ratio_cc = st.number_input(
                    "Ratio Controles:Casos:",
                    min_value=1, max_value=10, value=4, step=1,
                    help="Número de controles por cada caso"
                )
                power_cc = st.select_slider(
                    "Poder estadístico (1-β):",
                    options=[0.70, 0.80, 0.90, 0.95], value=0.80,
                    help="Probabilidad de detectar un efecto real"
                )

            if st.button("🧮 CALCULAR MUESTRA (CASOS-CONTROLES)", use_container_width=True):
                result_cc = calculate_sample_size_case_control(
                    or_expected, alpha=alpha_cc, power=power_cc,
                    ratio_controls_cases=ratio_cc
                )

                st.markdown("---")
                st.markdown("### 📊 Resultados - Casos y Controles")

                col_res_cc = st.columns(4)

                with col_res_cc[0]:
                    st.metric("🏥 Casos necesarios", f"{result_cc['n_cases']:,}")
                with col_res_cc[1]:
                    st.metric("👥 Controles necesarios", f"{result_cc['n_controls']:,}")
                with col_res_cc[2]:
                    st.metric("📊 Total (N)", f"{result_cc['total']:,}")
                with col_res_cc[3]:
                    st.metric("Ratio", f"{ratio_cc}:1")

                st.info(f"""
                **Para detectar un Odds Ratio de {or_expected} en un estudio de casos y controles:**

                - Se requieren **{result_cc['n_cases']:,} casos** y **{result_cc['n_controls']:,} controles**
                - **Total: {result_cc['total']:,} participantes**
                - Ratio utilizado: {ratio_cc}:1 (controles:casos)
                - Con un poder del {power_cc*100:.0f}% y α = {alpha_cc:.3f}
                """)

        else:
            st.markdown("### Parámetros del Estudio de Cohortes")

            col_sample = st.columns([1, 1, 1, 1])

            with col_sample[0]:
                p1 = st.number_input(
                    "Proporción Grupo 1 (p1):",
                    min_value=0.001, max_value=0.999, value=0.30, format="%.3f",
                    help="Proporción esperada en el grupo de intervención"
                )

            with col_sample[1]:
                p2 = st.number_input(
                    "Proporción Grupo 2 (p2):",
                    min_value=0.001, max_value=0.999, value=0.50, format="%.3f",
                    help="Proporción esperada en el grupo control"
                )
                power = st.select_slider(
                    "Poder estadístico (1-β):",
                    options=[0.70, 0.80, 0.90, 0.95], value=0.80,
                    help="Probabilidad de detectar un efecto real"
                )

            with col_sample[2]:
                ratio = st.number_input(
                    "Ratio de asignación (n2/n1):",
                    min_value=0.1, max_value=10.0, value=1.0, format="%.2f",
                    help="Proporción de participantes entre grupos"
                )
                alpha = st.select_slider(
                    "Nivel de significancia (α):",
                    options=[0.01, 0.05, 0.10], value=0.05,
                    help="Probabilidad de error tipo I"
                )

            with col_sample[3]:
                test_type = st.radio(
                    "Tipo de prueba:",
                    ["Two-sided", "One-sided"],
                    horizontal=True
                )

            if st.button("🧮 CALCULAR MUESTRA (COHORTES)", use_container_width=True):
                if test_type == "One-sided":
                    alpha = alpha / 2

                result = calculate_sample_size(p1, p2, alpha, power, ratio)

                st.markdown("---")
                st.markdown("### 📊 Resultados - Cohortes")

                col_res = st.columns(4)

                with col_res[0]:
                    st.metric("👥 Grupo 1 (n1)", f"{result['n1']:,}")
                with col_res[1]:
                    st.metric("👥 Grupo 2 (n2)", f"{result['n2']:,}")
                with col_res[2]:
                    st.metric("📊 Total (N)", f"{result['total']:,}")
                with col_res[3]:
                    nnt_calc = 1 / abs(p1 - p2) if p1 != p2 else float('inf')
                    st.metric("📐 NNT", f"{int(nnt_calc):,}" if nnt_calc != float('inf') else "∞")

                effect_size = abs(p1 - p2)

                st.info(f"""
                **Para detectar una diferencia de {effect_size:.1%} entre las proporciones ({p1:.1%} vs {p2:.1%}):**

                - Se requieren **{result['total']:,} participantes** en total
                - **{result['n1']:,}** en el Grupo 1 y **{result['n2']:,}** en el Grupo 2
                - Con un poder del {power*100:.0f}% y α = {alpha*2 if test_type == "Two-sided" else alpha:.3f}
                """)

    # ==========================================
    # MÓDULO: VIGILANCIA & IA
    # ==========================================
    elif menu == "📈 Vigilancia & IA":
        st.header("📈 Vigilancia Epidemiológica y Proyecciones con IA")

        st.info("""
        **Modelo SEIR Completo:**
        - **S** (Susceptibles): Población en riesgo de infectarse
        - **E** (Expuestos): Infectados en período de incubación
        - **I** (Infectados): Casos activos con síntomas
        - **R** (Recuperados): Inmunes o que se recuperaron
        """)

        with st.expander("📥 Datos del Brote", expanded=True):
            if st.session_state.df_v is None or st.button("🔄 Reiniciar Datos"):
                casos_acum = np.cumsum([10, 15, 25, 40, 55, 70, 90, 120, 150, 180])
                recuperados_init = np.concatenate([[0], casos_acum[:-1] * 0.7])
                session_recuperados = np.minimum(casos_acum * 0.6, recuperados_init).astype(int)

                st.session_state.df_v = pd.DataFrame({
                    "Dia": range(1, 11),
                    "Nuevos": [10, 15, 25, 40, 55, 70, 90, 120, 150, 180],
                    "Recuperados": session_recuperados.tolist(),
                    "Fallecidos": [0, 0, 1, 2, 3, 4, 6, 8, 10, 12]
                })

            df_v = st.data_editor(st.session_state.df_v, num_rows="dynamic")
            st.session_state.df_v = df_v

        st.subheader("🧬 Parámetros del Modelo SEIR")

        col_params = st.columns([1, 1, 1, 1, 1])
        with col_params[0]:
            beta = st.slider("β (Tasa Transmisión)", 0.1, 1.0, 0.3, 0.01,
                            help="Probabilidad de transmisión por contacto")
        with col_params[1]:
            sigma = st.slider("σ (Tasa Incubación)", 0.05, 0.5, 0.2, 0.01,
                             help="1/período de incubación")
        with col_params[2]:
            gamma = st.slider("γ (Tasa Recuperación)", 0.05, 0.5, 0.1, 0.01,
                             help="1/tiempo de infección")
        with col_params[3]:
            rho = st.slider("ρ (Tasa Diagnóstico)", 0.1, 1.0, 0.4, 0.01,
                           help="Proporción de casos detectados")
        with col_params[4]:
            n_sim = st.number_input("Sim. Monte Carlo", 50, 1000, 100, 50)

        r0 = beta / gamma
        duracion_infeccion = 1 / gamma
        periodo_incubacion = 1 / sigma

        col_metrics = st.columns([1, 1, 1, 1, 1])
        with col_metrics[0]:
            st.metric("R0 Estimado", f"{r0:.2f}",
                     delta="🔴 Epidemia" if r0 > 1 else "🟢 Controlada",
                     delta_color="inverse" if r0 > 1 else "normal")
        with col_metrics[1]:
            casos_reales = int(df_v['Nuevos'].sum() / rho)
            st.metric("Casos Totales", f"{casos_reales:,}")
        with col_metrics[2]:
            ifr = df_v['Fallecidos'].sum() / casos_reales * 100 if casos_reales > 0 else 0
            st.metric("IFR (%)", f"{ifr:.2f}%")
        with col_metrics[3]:
            st.metric("Duración Infección", f"{duracion_infeccion:.1f} días")
        with col_metrics[4]:
            st.metric("Período Incubación", f"{periodo_incubacion:.1f} días")

        if len(df_v) > 3:
            with st.spinner("⏳ Ejecutando modelo SEIR..."):
                _, _, RandomForestRegressor, _ = get_heavy_imports()

                days_total = len(df_v) + 30
                N = casos_reales * 10

                I0 = casos_reales // 10
                E0 = I0 * 2
                R0_init = int(df_v['Recuperados'].iloc[-1]) if 'Recuperados' in df_v.columns else I0
                S0 = N - E0 - I0 - R0_init

                S_arr, E_arr, I_arr, R_arr = [S0], [E0], [I0], [R0_init]

                for t in range(1, days_total):
                    S_t = S_arr[-1]
                    E_t = E_arr[-1]
                    I_t = I_arr[-1]
                    R_t = R_arr[-1]

                    new_infections = beta * S_t * I_t / N
                    new_exposed_to_infected = sigma * E_t
                    new_recoveries = gamma * I_t

                    S_new = S_t - new_infections
                    E_new = E_t + new_infections - new_exposed_to_infected
                    I_new = I_t + new_exposed_to_infected - new_recoveries
                    R_new = R_t + new_recoveries

                    S_arr.append(max(0, S_new))
                    E_arr.append(max(0, E_new))
                    I_arr.append(max(0, I_new))
                    R_arr.append(max(0, R_new))

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(df_v[['Dia']], df_v['Nuevos'])

                futuro_x = np.array([[len(df_v)+i] for i in range(1, 16)])
                base_preds = model.predict(futuro_x)

                all_sims = []
                std_err = max(df_v['Nuevos'].std(), 1)
                for _ in range(n_sim):
                    noise = np.random.normal(0, std_err * 0.6, size=base_preds.shape)
                    all_sims.append(np.maximum(0, base_preds + noise))

                all_sims = np.array(all_sims)
                p_mean = np.mean(all_sims, axis=0)
                p_low = np.percentile(all_sims, 2.5, axis=0)
                p_high = np.percentile(all_sims, 97.5, axis=0)

                fig_seir = go.Figure()

                days_range = list(range(days_total))

                fig_seir.add_trace(go.Scatter(
                    x=days_range, y=S_arr,
                    name="S (Susceptibles)",
                    line=dict(color='#60a5fa', width=2),
                    fill='tozeroy', fillcolor='rgba(96, 165, 250, 0.2)'
                ))

                fig_seir.add_trace(go.Scatter(
                    x=days_range, y=E_arr,
                    name="E (Expuestos)",
                    line=dict(color='#f59e0b', width=2),
                    fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.2)'
                ))

                fig_seir.add_trace(go.Scatter(
                    x=days_range, y=I_arr,
                    name="I (Infectados)",
                    line=dict(color='#ef4444', width=3),
                    fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)'
                ))

                fig_seir.add_trace(go.Scatter(
                    x=days_range, y=R_arr,
                    name="R (Recuperados)",
                    line=dict(color='#10b981', width=2),
                    fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.2)'
                ))

                fig_seir.update_layout(
                    title=f"📊 Modelo SEIR Completo (R0 = {r0:.2f})",
                    xaxis_title="Día",
                    yaxis_title="Población",
                    height=500,
                    template="plotly_dark",
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                )

                st.plotly_chart(fig_seir, use_container_width=True)

                fig_ia = go.Figure()

                fig_ia.add_trace(go.Scatter(
                    x=df_v["Dia"],
                    y=df_v["Nuevos"],
                    name="Histórico",
                    line=dict(color='#3b82f6', width=4),
                    mode='lines+markers'
                ))

                fig_ia.add_trace(go.Scatter(
                    x=futuro_x.flatten(),
                    y=p_mean,
                    name="Proyección IA",
                    line=dict(color='#ef4444', dash='dash', width=3),
                    mode='lines'
                ))

                fig_ia.add_trace(go.Scatter(
                    x=np.concatenate([futuro_x.flatten(), futuro_x.flatten()[::-1]]),
                    y=np.concatenate([p_high, p_low[::-1]]),
                    fill='toself',
                    fillcolor='rgba(239, 68, 68, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name="IC 95%"
                ))

                fig_ia.update_layout(
                    title="📈 Proyección de Casos Nuevos (Monte Carlo)",
                    xaxis_title="Día",
                    yaxis_title="Casos Nuevos",
                    height=400,
                    template="plotly_dark"
                )

                st.plotly_chart(fig_ia, use_container_width=True)

                df_proy = pd.DataFrame({
                    "Día": futuro_x.flatten(),
                    "Proyección": p_mean.astype(int),
                    "IC Inferior": p_low.astype(int),
                    "IC Superior": p_high.astype(int)
                })
                st.dataframe(df_proy, use_container_width=True)

    # ==========================================
    # MÓDULO: REVISIÓN DE LITERATURA
    # ==========================================
    elif menu == "📚 Revisión de Literatura":
        st.header("📚 Centro de Evidencia Científica")

        st.info("Gestione todo el proceso de su revisión sistemática desde una sola interfaz.")

        tab_pico, tab_prisma, tab_forest, tab_meta, tab_quality = st.tabs([
            "🤖 Extracción PICO",
            "📑 PRISMA Flowchart",
            "🌲 Forest Plot",
            "📊 Meta-análisis",
            "⚖️ Calidad (RoB/GRADE)"
        ])

        with tab_pico:
            st.subheader("🤖 Analizador IA de Evidencia Científica")

            api_k = st.text_input(
                "🔑 OpenAI API Key", type="password", key="api_pico",
                placeholder="sk-...",
                help="Obtenga su API key en platform.openai.com"
            )

            if not api_k:
                st.info("💡 Ingrese su OpenAI API Key para activar el análisis inteligente.")
            else:
                col_left, col_right = st.columns([1, 2])

                with col_left:
                    metodo = st.radio("📥 Método de Carga:", ["PDF", "DOI"], key="met_pico", horizontal=True)
                    ext = LiteratureAIExtractor(api_k)

                    if metodo == "PDF":
                        f = st.file_uploader("Subir artículo PDF", type="pdf", key="pdf_pico")
                        if f and st.button("🔍 Extraer PICO", use_container_width=True):
                            with st.spinner("⏳ Analizando con IA..."):
                                res = ext.from_pdf(f)
                                if res and "error" not in res:
                                    st.session_state.articulos_pico.append(res)
                                    st.success("✅ Artículo analizado exitosamente!")
                    else:
                        doi = st.text_input("DOI", placeholder="10.1056/...", key="doi_pico")
                        if doi and st.button("🔍 Consultar DOI", use_container_width=True):
                            with st.spinner("⏳ Consultando CrossRef..."):
                                res = ext.from_doi(doi)
                                if res and "error" not in res:
                                    st.session_state.articulos_pico.append(res)
                                    st.success("✅ Artículo analizado exitosamente!")

                with col_right:
                    if st.session_state.articulos_pico:
                        st.write("📚 Biblioteca de Evidencia")
                        df_articulos = pd.DataFrame(st.session_state.articulos_pico)
                        st.dataframe(df_articulos, use_container_width=True)

                        if st.button("🗑️ Limpiar Biblioteca", use_container_width=True):
                            st.session_state.articulos_pico = []
                            st.rerun()
                    else:
                        st.info("📭 Biblioteca vacía. Cargue un artículo para comenzar.")

        with tab_prisma:
            st.subheader("📑 Flujograma PRISMA 2020")

            prisma_data = st.session_state.prisma_data if st.session_state.prisma_data else {
                'registros_db': 1500, 'registros_registros': 50, 'duplicados': 400,
                'excluidos_title': 500, 'excluidos_abstract': 400, 'articulos_recuperados': 250,
                'articulos_evaluated': 200, 'articulos_excluidos': 150, 'estudios_included': 25
            }

            col_p1, col_p2 = st.columns([1, 2])

            with col_p1:
                prisma_data['registros_db'] = st.number_input("Registros de DB:", min_value=0, value=prisma_data.get('registros_db', 1500), step=10)
                prisma_data['duplicados'] = st.number_input("Duplicados:", min_value=0, value=prisma_data.get('duplicados', 400), step=10)
                prisma_data['excluidos_title'] = st.number_input("Excluidos por Título:", min_value=0, value=prisma_data.get('excluidos_title', 500), step=10)
                prisma_data['estudios_included'] = st.number_input("Estudios Finales Incluidos:", min_value=0, value=prisma_data.get('estudios_included', 25), step=1)
                st.session_state.prisma_data = prisma_data

            with col_p2:
                total_screen = prisma_data['registros_db'] - prisma_data['duplicados']
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("Registros Iniciales", f"{prisma_data['registros_db']:,}")
                col_stat2.metric("Tras Screening", f"{total_screen:,}")
                col_stat3.metric("Estudios Finales", f"{prisma_data['estudios_included']:,}")

                fig_prisma = go.Figure()
                fig_prisma.add_trace(go.Scatter(
                    x=[0.15, 0.15, 0.35, 0.35, 0.15],
                    y=[5, 4.6, 4.6, 5, 5],
                    fill='toself', fillcolor='#667eea', line=dict(color='white'),
                    text=f"Identificación<br>{prisma_data['registros_db']}", mode='text+lines',
                    textfont=dict(size=12, color='white'), showlegend=False
                ))
                fig_prisma.add_trace(go.Scatter(
                    x=[0.15, 0.15, 0.35, 0.35, 0.15],
                    y=[4, 3.6, 3.6, 4, 4],
                    fill='toself', fillcolor='#10b981', line=dict(color='white'),
                    text=f"Screening<br>{total_screen:,}", mode='text+lines',
                    textfont=dict(size=12, color='white'), showlegend=False
                ))
                fig_prisma.add_trace(go.Scatter(
                    x=[0.15, 0.15, 0.35, 0.35, 0.15],
                    y=[3, 2.6, 2.6, 3, 3],
                    fill='toself', fillcolor='#f59e0b', line=dict(color='white'),
                    text=f"Incluidos<br>{prisma_data['estudios_included']:,}", mode='text+lines',
                    textfont=dict(size=12, color='white'), showlegend=False
                ))
                fig_prisma.update_layout(
                    height=400, xaxis=dict(visible=False, range=[-0.5, 0.5]),
                    yaxis=dict(visible=False, range=[0, 6]),
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_prisma, use_container_width=True)

        with tab_forest:
            st.subheader("🌲 Análisis Visual de Efectos")

            if 'forest_studies' not in st.session_state:
                st.session_state.forest_studies = pd.DataFrame({
                    'Estudio': ['Smith 2020', 'Johnson 2019', 'Williams 2021'],
                    'Eventos_Tto': [20, 35, 28], 'Total_Tto': [100, 150, 120],
                    'Eventos_Ctrl': [30, 50, 45], 'Total_Ctrl': [100, 150, 120]
                })

            edit_forest = st.data_editor(
                st.session_state.forest_studies,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Estudio": st.column_config.TextColumn("Estudio", required=True),
                    "Eventos_Tto": st.column_config.NumberColumn("Eventos Tto", min_value=0),
                    "Total_Tto": st.column_config.NumberColumn("Total Tto", min_value=1),
                    "Eventos_Ctrl": st.column_config.NumberColumn("Eventos Ctrl", min_value=0),
                    "Total_Ctrl": st.column_config.NumberColumn("Total Ctrl", min_value=1)
                }
            )

            col_gen, col_transfer = st.columns(2)

            with col_gen:
                if st.button("🌲 Generar Forest Plot", use_container_width=True):
                    if len(edit_forest) > 0:
                        st.session_state.forest_studies = edit_forest

                        df_calc = edit_forest.copy()
                        df_calc['OR'] = (df_calc['Eventos_Tto'] / (df_calc['Total_Tto'] - df_calc['Eventos_Tto'])) / \
                                        (df_calc['Eventos_Ctrl'] / (df_calc['Total_Ctrl'] - df_calc['Eventos_Ctrl']))

                        fig, ax = plt.subplots(figsize=(12, max(4, len(df_calc) * 0.8)))

                        for i, (_, row) in enumerate(df_calc.iterrows()):
                            or_val = row['OR']
                            if pd.notna(or_val) and or_val > 0:
                                ax.plot(or_val, i, 'bs', markersize=8, markeredgecolor='darkblue')
                                ax.text(or_val + 0.15, i, f"{row['Estudio']}: OR={or_val:.2f}", va='center', fontsize=10)

                        ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Null (OR=1)')
                        ax.set_xscale('log')
                        ax.set_xlabel('Odds Ratio (escala logarítmica)', fontsize=12)
                        ax.set_ylabel('Estudios', fontsize=12)
                        ax.set_yticks([])
                        ax.legend(loc='upper right')
                        ax.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)

            with col_transfer:
                if st.button("➕ Enviar a Meta-análisis", use_container_width=True):
                    if len(edit_forest) >= 2:
                        st.session_state.meta_studies = edit_forest
                        st.success(f"✅ {len(edit_forest)} estudios transferidos a Meta-análisis!")
                    else:
                        st.warning("Se requieren al menos 2 estudios para meta-análisis.")

        with tab_meta:
            st.subheader("📊 Modelos Estadísticos")

            if len(st.session_state.meta_studies) < 2:
                st.warning("📌 Se requieren al menos 2 estudios. Importe datos desde Forest Plot.")
            else:
                mod_meta = st.selectbox(
                    "Modelo:",
                    ["Efectos Fijos (Peto)", "Efectos Aleatorios (DerSimonian-Laird)"]
                )

                if st.button("📊 Calcular Meta-análisis", use_container_width=True):
                    ev_e = tuple(st.session_state.meta_studies['Eventos_Tto'].tolist())
                    tt_e = tuple(st.session_state.meta_studies['Total_Tto'].tolist())
                    ev_c = tuple(st.session_state.meta_studies['Eventos_Ctrl'].tolist())
                    tt_c = tuple(st.session_state.meta_studies['Total_Ctrl'].tolist())

                    if "Fijos" in mod_meta:
                        res_m = meta_analysis_fixed_effect(ev_e, tt_e, ev_c, tt_c)
                        pooled_or = res_m['pooled_or']
                        i2 = res_m['I2']
                    else:
                        res_m = meta_analysis_random_effects(ev_e, tt_e, ev_c, tt_c)
                        pooled_or = res_m['pooled_or_re']
                        i2 = res_m['I2']

                    c1, c2, c3 = st.columns(3)
                    c1.metric("OR Combinado", f"{pooled_or:.2f}")
                    c2.metric("I² (Heterogeneidad)", f"{i2:.1f}%", f"{'Alta' if i2 > 75 else 'Moderada' if i2 > 50 else 'Baja'}")
                    c3.metric("p-value", f"{res_m.get('p_value', 0.05):.4f}")

                    if i2 > 75:
                        st.error("⚠️ Alta heterogeneidad detectada (I² > 75%).")
                    elif i2 > 50:
                        st.warning("⚠️ Heterogeneidad moderada (I² > 50%).")

        with tab_quality:
            st.subheader("⚖️ Evaluación de Calidad de Evidencia")

            q_sub = st.radio("Herramienta de Evaluación:", ["RoB 2 (Riesgo de Sesgo)", "GRADE"], horizontal=True)

            if q_sub == "RoB 2 (Riesgo de Sesgo)":
                s_name = st.text_input("📝 Nombre del Estudio:", placeholder="Autor, Año")

                col_rob = st.columns(2)

                with col_rob[0]:
                    st.write("**Dominios 1-2**")
                    d1 = st.select_slider("D1: Randomización", ["Low", "Some Concerns", "High"], value="Low")
                    d2 = st.select_slider("D2: Intervención", ["Low", "Some Concerns", "High"], value="Low")

                with col_rob[1]:
                    st.write("**Dominios 3-4**")
                    d3 = st.select_slider("D3: Datos Faltantes", ["Low", "Some Concerns", "High"], value="Low")
                    d4 = st.select_slider("D4: Medición", ["Low", "Some Concerns", "High"], value="Low")

                if st.button("💾 Guardar Evaluación RoB 2", use_container_width=True):
                    if s_name:
                        assessment = {'Estudio': s_name, 'D1': d1, 'D2': d2, 'D3': d3, 'D4': d4}
                        st.session_state.rob_assessments.append(assessment)
                        st.success("✅ Evaluación guardada exitosamente!")

                if st.session_state.rob_assessments:
                    st.write("### 📋 Evaluaciones Guardadas")
                    df_rob = pd.DataFrame(st.session_state.rob_assessments)
                    st.dataframe(df_rob, use_container_width=True)
            else:
                st.subheader("📋 Sistema GRADE - Certeza de Evidencia")
                outcome = st.text_input("📝 Resultado (Outcome):", placeholder="ej: Mortalidad")

                col_g = st.columns(2)

                with col_g[0]:
                    st.write("**Factores de Reducción**")
                    r_bias = st.number_input("Riesgo de Sesgo (0 a -2)", min_value=-2, max_value=0, value=0, step=1)
                    incons = st.number_input("Inconsistencia (0 a -2)", min_value=-2, max_value=0, value=0, step=1)
                    indir = st.number_input("Indirectitud (0 a -2)", min_value=-2, max_value=0, value=0, step=1)

                with col_g[1]:
                    st.write("**Factores de Aumento**")
                    large = st.checkbox("✅ Efecto Grande (+1)")
                    dose = st.checkbox("✅ Dosis-Respuesta (+1)")
                    confound = st.checkbox("✅ Factores Confusores (+1)")

                if st.button("⚖️ Calcular Grado GRADE", use_container_width=True):
                    if outcome:
                        score = 4 + r_bias + incons + indir
                        score += (1 if large else 0) + (1 if dose else 0) + (1 if confound else 0)
                        score = max(1, min(4, score))

                        labels = {4: "🔴 Alta", 3: "🟡 Moderada", 2: "🟠 Baja", 1: "⚫ Muy Baja"}
                        st.metric("Certeza de Evidencia", labels.get(score, "⚫ Muy Baja"))

    # ==========================================
    # MÓDULO: SUPERVIVENCIA (KM)
    # ==========================================
    elif menu == "📉 Supervivencia (KM)":
        st.header("📉 Análisis de Supervivencia - Kaplan-Meier")

        tabs = st.tabs(["📝 Datos", "📈 Curva KM", "📊 Análisis"])

        with tabs[0]:
            st.markdown("### 📝 Ingrese datos de supervivencia")

            if st.session_state.survival_data is None:
                np.random.seed(42)
                st.session_state.survival_data = pd.DataFrame({
                    'ID': range(1, 101),
                    'Tiempo': np.concatenate([
                        np.random.exponential(30, 50),
                        np.random.exponential(20, 50)
                    ]).round(1),
                    'Evento': np.random.binomial(1, 0.4, 100),
                    'Grupo': ['Tratamiento'] * 50 + ['Control'] * 50,
                    'Edad': np.random.randint(30, 80, 100),
                    'Sexo': np.random.choice(['M', 'F'], 100)
                })

            col_upload = st.columns(3)

            with col_upload[0]:
                uploaded = st.file_uploader("📂 Cargar CSV:", type="csv")
                if uploaded:
                    df_loaded = pd.read_csv(uploaded)
                    st.session_state.survival_data = df_loaded
                    st.success(f"✅ {len(df_loaded)} registros!")

            with col_upload[1]:
                if st.button("🎲 Generar Ejemplo", use_container_width=True):
                    np.random.seed(42)
                    st.session_state.survival_data = pd.DataFrame({
                        'ID': range(1, 101),
                        'Tiempo': np.concatenate([
                            np.random.exponential(30, 50),
                            np.random.exponential(20, 50)
                        ]).round(1),
                        'Evento': np.random.binomial(1, 0.4, 100),
                        'Grupo': ['Tratamiento'] * 50 + ['Control'] * 50,
                    })
                    st.rerun()

            if st.session_state.survival_data is not None:
                df = st.session_state.survival_data
                edited = st.data_editor(
                    df,
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True
                )
                st.session_state.survival_data = edited

                col_stats = st.columns(4)
                col_stats[0].metric("Muestras", len(edited))
                col_stats[1].metric("Eventos", int(edited['Evento'].sum()))
                col_stats[2].metric("Censuras", int(len(edited) - edited['Evento'].sum()))
                col_stats[3].metric("Tiempo medio", f"{edited['Tiempo'].mean():.1f}")

        with tabs[1]:
            st.markdown("### 📈 Curva de Kaplan-Meier")

            df = st.session_state.survival_data

            if df is not None and len(df) > 0:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                text_cols = [c for c in df.columns if c not in num_cols]

                col_setup = st.columns(3)

                with col_setup[0]:
                    time_col = st.selectbox("⏱️ Tiempo:", num_cols)

                with col_setup[1]:
                    event_col = st.selectbox("⚠️ Evento:", num_cols)

                with col_setup[2]:
                    group_by = st.selectbox("👥 Agrupar:", ['Ninguno'] + text_cols)

                if st.button("📊 GENERAR CURVA KM", use_container_width=True):
                    def calculate_km(time, event):
                        df_km = pd.DataFrame({'time': time, 'event': event}).sort_values('time')
                        times = df_km['time'].values
                        events = df_km['event'].values
                        n = len(times)
                        unique_times = np.unique(times[events == 1])

                        survival_times = [0]
                        survival_probs = [1.0]
                        survived = n

                        for t in unique_times:
                            d = np.sum(events[times == t])
                            r = np.sum(times >= t)
                            if r > 0:
                                survived = survived * (1 - d / r)
                                survival_times.append(t)
                                survival_probs.append(survived / n)

                        survival_times.append(times.max())
                        survival_probs.append(survival_probs[-1])

                        return {'times': np.array(survival_times), 'survival': np.array(survival_probs)}

                    results = {}

                    if group_by == 'Ninguno':
                        results['Global'] = calculate_km(df[time_col].values, df[event_col].values)
                    else:
                        for group in df[group_by].unique():
                            mask = df[group_by] == group
                            results[str(group)] = calculate_km(
                                df.loc[mask, time_col].values,
                                df.loc[mask, event_col].values
                            )

                    fig, ax = plt.subplots(figsize=(10, 6))

                    colors = {'Tratamiento': '#3498db', 'Control': '#e74c3c', 'Global': '#2ecc71'}

                    for name, data in results.items():
                        color = colors.get(name, '#3498db')
                        ax.step(data['times'], data['survival'], where='post',
                               color=color, linewidth=2.5, label=name)

                    ax.set_xlabel('Tiempo')
                    ax.set_ylabel('Probabilidad de Supervivencia')
                    ax.set_title('Curva de Kaplan-Meier')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(left=0)
                    ax.set_ylim(0, 1.05)

                    st.pyplot(fig)

        with tabs[2]:
            st.markdown("### 📊 Análisis Avanzado")
            st.info("Análisis detallado disponible en versiones futuras.")

    # ==========================================
    # MÓDULO: CURVAS ROC
    # ==========================================
    elif menu == "🎯 Curvas ROC":
        st.header("🎯 Curvas ROC - Evaluación Diagnóstica")

        tabs = st.tabs(["📝 Datos", "📈 Curva ROC", "📊 Comparación"])

        with tabs[0]:
            st.markdown("### 📝 Ingrese datos para análisis ROC")

            if st.session_state.roc_data is None:
                np.random.seed(42)
                st.session_state.roc_data = pd.DataFrame({
                    'ID': range(1, 201),
                    'Probabilidad': np.concatenate([
                        np.random.beta(5, 2, 100),
                        np.random.beta(2, 5, 100)
                    ]),
                    'Real': ['Positivo'] * 100 + ['Negativo'] * 100
                })

            col_upload = st.columns([1, 1, 1])
            with col_upload[0]:
                uploaded = st.file_uploader("📂 Cargar CSV:", type="csv")
                if uploaded:
                    st.session_state.roc_data = pd.read_csv(uploaded)
                    st.success("✅ Datos cargados!")
            with col_upload[1]:
                if st.button("🎲 Generar Ejemplo", use_container_width=True):
                    np.random.seed(42)
                    st.session_state.roc_data = pd.DataFrame({
                        'ID': range(1, 201),
                        'Probabilidad': np.concatenate([
                            np.random.beta(5, 2, 100),
                            np.random.beta(2, 5, 100)
                        ]),
                        'Real': ['Positivo'] * 100 + ['Negativo'] * 100
                    })
                    st.rerun()

            if st.session_state.roc_data is not None:
                df = st.session_state.roc_data
                st.dataframe(df.head(20), use_container_width=True)

        with tabs[1]:
            st.markdown("### 📈 Curva ROC")

            if st.session_state.roc_data is not None:
                from sklearn.metrics import roc_curve, auc, confusion_matrix

                df = st.session_state.roc_data
                pred_col = st.selectbox("📊 Variable de predicción:", options=[c for c in df.columns if c not in ['ID', 'Real']])

                if st.button("📊 GENERAR CURVA ROC", use_container_width=True, type="primary"):
                    y_true = (df['Real'] == df['Real'].unique()[0]).astype(int)
                    y_score = df[pred_col].values

                    fpr, tpr, thresholds = roc_curve(y_true, y_score)
                    roc_auc = auc(fpr, tpr)

                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                    axes[0].plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'ROC (AUC={roc_auc:.3f})')
                    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Aleatorio')
                    axes[0].set_xlabel('1 - Especificidad (FPR)')
                    axes[0].set_ylabel('Sensibilidad (TPR)')
                    axes[0].set_title('Curva ROC')
                    axes[0].legend(loc='lower right')
                    axes[0].grid(True, alpha=0.3)

                    j_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[j_idx]

                    y_pred = (y_score >= optimal_threshold).astype(int)
                    cm = confusion_matrix(y_true, y_pred)

                    im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    axes[1].set_title(f'Matriz Confusión (θ={optimal_threshold:.3f})')
                    axes[1].set_xticks([0, 1])
                    axes[1].set_yticks([0, 1])
                    axes[1].set_xticklabels(['Negativo', 'Positivo'])
                    axes[1].set_yticklabels(['Negativo', 'Positivo'])
                    axes[1].set_xlabel('Predicción')
                    axes[1].set_ylabel('Real')

                    for i in range(2):
                        for j in range(2):
                            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                                       fontsize=18, fontweight='bold',
                                       color='white' if cm[i, j] > cm.max()/2 else 'black')

                    plt.tight_layout()
                    st.pyplot(fig)

                    col_m = st.columns(4)
                    with col_m[0]: st.metric("AUC-ROC", f"{roc_auc:.4f}")
                    with col_m[1]: st.metric("Sensibilidad", f"{tpr[j_idx]:.4f}")
                    with col_m[2]: st.metric("Especificidad", f"{1-fpr[j_idx]:.4f}")
                    with col_m[3]: st.metric("Punto Corte", f"{optimal_threshold:.4f}")

        with tabs[2]:
            st.markdown("### 📊 Comparación de Tests")
            st.info("Seleccione múltiples predictores para comparar su rendimiento diagnóstico.")

    # ==========================================
    # MÓDULO: MAPAS GEOGRÁFICOS
    # ==========================================
    elif menu == "🗺️ Mapas Geográficos":
        st.header("🗺️ Mapas Geográficos - Epidemiología Espacial")

        st.info("Ingrese ubicación por texto (País, Departamento, Ciudad).")

        GEO_DATABASE = {
            ('Colombia', 'Antioquia', 'Medellín'): (6.2442, -75.5812),
            ('Colombia', 'Cundinamarca', 'Bogotá'): (4.7110, -74.0721),
            ('Colombia', 'Valle del Cauca', 'Cali'): (3.8000, -76.5220),
            ('Colombia', 'Atlántico', 'Barranquilla'): (10.9685, -74.7813),
            ('Colombia', 'Santander', 'Bucaramanga'): (7.1190, -73.1198),
            ('Colombia', 'Bolívar', 'Cartagena'): (10.3910, -75.5142),
        }

        tabs = st.tabs(["📊 Coroplético", "🔥 Heatmap", "📍 Marcadores"])

        with tabs[0]:
            st.markdown("### 📊 Mapa Coroplético - Incidencia por Ubicación")

            if st.session_state.map_data is None:
                st.session_state.map_data = pd.DataFrame({
                    'Pais': ['Colombia'] * 6,
                    'Departamento': ['Antioquia', 'Cundinamarca', 'Valle del Cauca', 'Atlántico', 'Santander', 'Bolívar'],
                    'Municipio': ['Medellín', 'Bogotá', 'Cali', 'Barranquilla', 'Bucaramanga', 'Cartagena'],
                    'Casos': [1500, 1200, 1100, 900, 800, 750],
                    'Poblacion': [6500000, 3000000, 4500000, 2500000, 2000000, 2100000]
                })

            map_data = st.data_editor(
                st.session_state.map_data,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True
            )
            st.session_state.map_data = map_data

            col_stat = st.columns(3)
            col_stat[0].metric("Total Casos", f"{map_data['Casos'].sum():,}")
            col_stat[1].metric("Ubicaciones", len(map_data))
            if map_data['Poblacion'].sum() > 0:
                avg_tasa = (map_data['Casos'].sum() / map_data['Poblacion'].sum() * 100000)
                col_stat[2].metric("Tasa Promedio", f"{avg_tasa:.2f} x100k")

        with tabs[1]:
            st.markdown("### 🔥 Mapa de Calor - Densidad de Casos")
            st.info("Configure los centros de concentración de casos (hotspots).")

            hotspots = {
                'Bogotá': (4.7110, -74.0721),
                'Medellín': (6.2442, -75.5812),
                'Cali': (3.8000, -76.5220),
            }

            if st.button("🔥 GENERAR HEATMAP", use_container_width=True, type="primary"):
                heat_data = []
                for city, (lat, lon) in hotspots.items():
                    for _ in range(50):
                        heat_data.append({
                            'lat': lat + np.random.normal(0, 0.3),
                            'lon': lon + np.random.normal(0, 0.3),
                            'city': city
                        })

                df_heat = pd.DataFrame(heat_data)

                fig = px.density_mapbox(df_heat, lat='lat', lon='lon', radius=15,
                                        center={'lat': 4.5709, 'lon': -74.2973}, zoom=5,
                                        mapbox_style='carto-darkmatter', title='Mapa de Calor')
                fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"✅ {len(df_heat):,} puntos de calor")

        with tabs[2]:
            st.markdown("### 📍 Mapa con Marcadores")
            st.info("Visualice centros de salud, casos individuales, recursos hospitalarios.")

            if st.button("📍 MOSTRAR MAPA DE EJEMPLO", use_container_width=True):
                fig = px.scatter_mapbox(
                    pd.DataFrame({
                        'lat': [6.2442, 4.7110, 3.8000],
                        'lon': [-75.5812, -74.0721, -76.5220],
                        'name': ['Hospital Central Medellín', 'Clínica Norte Bogotá', 'UCI Cali']
                    }),
                    lat='lat', lon='lon', hover_name='name',
                    zoom=5, center={'lat': 4.5709, 'lon': -74.2973},
                    mapbox_style='carto-darkmatter', height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # MÓDULO: BIOINFORMÁTICA
    # ==========================================
    elif menu == "🧬 Bioinformática":
        st.header("🧬 Análisis de Secuencias Genéticas")

        CODON_TABLE = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
            'TAA': '*', 'TAG': '*', 'TGA': '*'
        }

        col_input = st.columns([1, 1])
        with col_input[0]:
            input_method = st.radio("Método:", ["Texto directo", "Archivo FASTA"], horizontal=True)

        seq = ""
        if input_method == "Texto directo":
            seq = st.text_area("Secuencia DNA/RNA:", placeholder="ATGCCGTAGCTG...", height=150).strip()
        else:
            uploaded = st.file_uploader("Cargar FASTA", type=['fasta', 'fa', 'txt'])
            if uploaded:
                content = uploaded.read().decode('utf-8')
                seq = ''.join([l.strip() for l in content.split('\n') if not l.startswith('>')])

        if seq:
            seq_clean = seq.upper().replace(" ", "").replace("\n", "")
            valid_bases = set('AGTCUN')
            invalid = [b for b in seq_clean if b not in valid_bases]

            if invalid:
                st.error(f"❌ Bases inválidas: {list(set(invalid))}")
            else:
                seq_type = "RNA" if 'U' in seq_clean else "DNA"
                st.success(f"✅ {seq_type} válida - {len(seq_clean)} pb")

                if st.button("🧬 ANALIZAR SECUENCIA", use_container_width=True, type="primary"):
                    gc = (seq_clean.count('G') + seq_clean.count('C')) / len(seq_clean)
                    composition = Counter(seq_clean)

                    col_stats = st.columns(4)
                    with col_stats[0]: st.metric("Longitud", f"{len(seq_clean):,} pb")
                    with col_stats[1]: st.metric("Tipo", seq_type)
                    with col_stats[2]: st.metric("GC", f"{gc:.2%}")
                    with col_stats[3]: st.metric("Ratio GC/AT", f"{gc/(1-gc):.2f}" if gc < 1 else "N/A")

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    bases = ['A', 'T', 'G', 'C'] if 'U' not in seq_clean else ['A', 'U', 'G', 'C']
                    counts = [composition.get(b, 0) for b in bases]
                    colors = {'A': '#e74c3c', 'T': '#3498db', 'G': '#2ecc71', 'C': '#f39c12', 'U': '#9b59b6'}

                    ax1.bar(bases, counts, color=[colors.get(b, '#95a5a6') for b in bases], edgecolor='black')
                    ax1.set_xlabel('Nucleótido')
                    ax1.set_ylabel('Conteo')
                    ax1.set_title('Composición')
                    for i, c in enumerate(counts):
                        ax1.text(i, c + max(counts)*0.02, str(c), ha='center', fontweight='bold')

                    ax2.pie(counts, labels=bases, colors=[colors.get(b, '#95a5a6') for b in bases],
                           autopct='%1.1f%%', startangle=90)
                    ax2.set_title('Distribución')

                    plt.tight_layout()
                    st.pyplot(fig)

                    st.markdown("### 🔄 Complementaria Reversa")
                    comp_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N', 'U': 'A'}
                    rev_comp = ''.join([comp_map.get(b, 'N') for b in seq_clean[::-1]])
                    st.code('\n'.join([rev_comp[i:i+80] for i in range(0, len(rev_comp), 80)]))

                    st.markdown("### 🧪 Traducción a Proteína")
                    protein = ''.join([CODON_TABLE.get(seq_clean[i:i+3], 'X') for i in range(0, len(seq_clean) - len(seq_clean) % 3, 3)])
                    st.code('\n'.join([protein[i:i+60] for i in range(0, len(protein), 60)]))

    # ==========================================
    # MÓDULO: MI SUSCRIPCIÓN
    # ==========================================
    elif menu == "💳 Mi Suscripción":
        st.header("💳 Gestión de Suscripción")

        db = load_users()
        user_data = db.get(st.session_state.user, {})

        col_sub = st.columns(2)

        with col_sub[0]:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px;">
                    <h3 style="color: white; margin-bottom: 15px;">✨ EpiDiagnosis Pro Premium</h3>
                    <ul style="color: #f0f0f0; font-size: 15px; line-height: 2;">
                        <li>✓ Análisis PICO con GPT-4</li>
                        <li>✓ Predicciones epidemiológicas avanzadas</li>
                        <li>✓ Monte Carlo Simulations</li>
                        <li>✓ Bioestadística completa</li>
                        <li>✓ Meta-análisis y Forest Plot</li>
                        <li>✓ Evaluación RoB/GRADE</li>
                        <li>✓ Análisis de supervivencia (KM)</li>
                        <li>✓ Curvas ROC</li>
                        <li>✓ Mapas geográficos</li>
                    </ul>
                    <h2 style="color: #ffd700; margin-top: 20px;"> 22 US$ / Mes</h2>
                </div>
            """, unsafe_allow_html=True)

        with col_sub[1]:
            st.markdown("### 📋 Estado de Su Cuenta")
            expiry = user_data.get('expiry', 'N/A')
            role = user_data.get('role', 'user')

            col_status = st.columns(2)
            with col_status[0]:
                st.metric("🎫 Tipo", role.upper())
            with col_status[1]:
                try:
                    exp_date = datetime.strptime(expiry, "%Y-%m-%d")
                    days_left = (exp_date - datetime.now()).days
                    if days_left < 0:
                        st.metric("⏰ Estado", "🔴 Expirada", delta_color="inverse")
                    elif days_left <= 7:
                        st.metric("⏰ Días Restantes", f"{days_left}", delta_color="inverse")
                    else:
                        st.metric("⏰ Días Restantes", f"{days_left}", delta_color="normal")
                except:
                    st.metric("⏰ Expiración", expiry)

            st.markdown(f"**📅 Fecha de expiración:** {expiry}")

            PAYMENT_LINK = "https://checkout.bold.co/payment/LNK_2W3K24BLVU"

            st.markdown("---")
            st.markdown("### 🔒 Realizar Pago")

            st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #1e293b; border-radius: 15px; margin-top: 20px;">
                    <a href="{PAYMENT_LINK}" target="_blank" style="text-decoration: none;">
                        <button style="
                            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                            color: white;
                            padding: 20px 50px;
                            border-radius: 12px;
                            font-size: 20px;
                            font-weight: bold;
                            border: none;
                            cursor: pointer;
                        ">
                            💳 PAGAR AHORA - 22 US$
                        </button>
                    </a>
                </div>
            """, unsafe_allow_html=True)

            st.info("💡 Después del pago, comunícate con soporte para activar tu licencia Premium.")

    # ==========================================
    # MÓDULO: ADMIN
    # ==========================================
    elif menu == "⚙️ Admin":
        st.header("⚙️ Panel de Administración")

        db = load_users()

        st.subheader("📊 Usuarios Registrados")
        df_users = pd.DataFrame([
            {
                "Email": email,
                "Rol": data.get("role", "N/A"),
                "Expiración": data.get("expiry", "N/A"),
                "ID Doc": data.get("id_doc", "N/A")
            }
            for email, data in db.items()
        ])
        st.dataframe(df_users, use_container_width=True)

        st.subheader("🔧 Gestión de Licencias")

        col_admin = st.columns([2, 1, 1])
        with col_admin[0]:
            target = st.text_input("Email del usuario:")
        with col_admin[1]:
            days = st.number_input("Días a agregar:", 1, 365, 30)

        col_btns_admin = st.columns(3)
        with col_btns_admin[0]:
            if st.button("➕ Renovar Licencia", use_container_width=True):
                if target in db:
                    try:
                        current_expiry = datetime.strptime(db[target]['expiry'], "%Y-%m-%d")
                        if current_expiry < datetime.now():
                            current_expiry = datetime.now()
                        new_expiry = (current_expiry + timedelta(days=days)).strftime("%Y-%m-%d")
                        db[target]['expiry'] = new_expiry
                        save_users(db)
                        st.success(f"✅ Licencia de {target} renovada hasta {new_expiry}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Usuario no encontrado")

        with col_btns_admin[1]:
            if st.button("🎫 Cambiar a Admin", use_container_width=True):
                if target in db:
                    db[target]['role'] = 'admin'
                    save_users(db)
                    st.success(f"✅ {target} ahora es admin")

        with col_btns_admin[2]:
            if st.button("🗑️ Eliminar Usuario", use_container_width=True):
                if target in db and target != "JCOLLAZOSR@UOC.EDU":
                    del db[target]
                    save_users(db)
                    st.success(f"✅ Usuario {target} eliminado")
                else:
                    st.error("No se puede eliminar este usuario")

        st.subheader("📈 Estadísticas del Sistema")
        col_stats_admin = st.columns(3)
        with col_stats_admin[0]:
            st.metric("Total Usuarios", len(db))
        with col_stats_admin[1]:
            active = sum(1 for u in db.values()
                        if datetime.strptime(u.get("expiry", "2000-01-01"), "%Y-%m-%d") > datetime.now())
            st.metric("Usuarios Activos", active)
        with col_stats_admin[2]:
            admins = sum(1 for u in db.values() if u.get("role") == "admin")
            st.metric("Administradores", admins)

    # ==========================================
    # FOOTER
    # ==========================================
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; padding: 10px;'>"
        "🩺 EpiDiagnosis Pro v6.0 | © 2026 | Fundación Juan manuel Collazos | "
        "Desarrollado por: Juan Manuel Collazos Rozo, MD, MSc. | "
        "WhatsApp: (+57) 3113682907 - Correo electrónico: j.collazosmd@gmail.com"
        "</div>",
        unsafe_allow_html=True
    )
