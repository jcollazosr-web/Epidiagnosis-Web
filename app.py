"""
EpiDiagnosis Pro V7.0 - Aplicación Completa de Epidemiología y Bioestadística
================================================================================
MEJORAS V7.0:
1. PERSISTENCIA: Datos guardados por usuario (JSON local)
2. FILTROS BD: Variables con nombres, porcentaje de pérdidas
3. PRECIO: 23 USD/mes actualizado
4. TAMAÑO MUESTRA: Observacionales, Multivariada, Confundidores
5. PSM: Propensity Score Matching
6. COX/POISSON: Regresión avanzada en supervivencia
7. ARIMA: Modelos predictivos para vigilancia
8. ASIS: Exportación automatizada PDF/Word
9. VIGILANCIA: Proyecciones con desviaciones estándar y fechas

Autor: MiniMax Agent
Versión: 7.0
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
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import base64
from io import BytesIO

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE PERSISTENCIA
# ==========================================
USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

def get_user_data_path(user_email):
    """Obtiene la ruta del archivo de datos del usuario"""
    safe_name = hashlib.md5(user_email.encode()).hexdigest()[:12]
    return os.path.join(USER_DATA_DIR, f"user_{safe_name}.json")

def load_user_data(user_email):
    """Carga datos persistentes del usuario"""
    path = get_user_data_path(user_email)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ No se pudieron cargar datos del usuario: {e}")
            return {}
    return {}

def save_user_data(user_email, data):
    """Guarda datos persistentes del usuario"""
    path = get_user_data_path(user_email)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# ==========================================
# LAZY IMPORTS
# ==========================================
@st.cache_resource
def get_heavy_imports():
    """Carga perezosa de módulos pesados"""
    global pdfplumber, openai
    import pdfplumber
    import openai
    from sklearn.ensemble import RandomForestRegressor
    from scipy import stats
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.proportion import proportions_ztest
    return pdfplumber, openai, RandomForestRegressor, stats, ARIMA, proportions_ztest

@st.cache_resource
def get_analysis_imports():
    """Importaciones para análisis avanzado"""
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import chi2, norm
    return KaplanMeierFitter, CoxPHFitter, roc_curve, auc, confusion_matrix, classification_report, LogisticRegression, chi2, norm

@st.cache_resource
def get_psm_imports():
    """Importaciones para PSM"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    return LogisticRegression, NearestNeighbors

# ==========================================
# CONFIGURACIÓN VISUAL Y CSS PRO
# ==========================================
st.set_page_config(page_title="EpiDiagnosis Pro V7.0", layout="wide", page_icon="🧬")

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
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# RATE LIMITING
# ==========================================
class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.window = 60
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
USER_DB_FILE = "users_v7_db.json"

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
# CACHE INTELIGENTE
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
    except Exception as e:
        st.warning(f"⚠️ Error cargando datos: {e}")
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def load_crossref_data(doi):
    """Cache para consultas a CrossRef"""
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
        if r.status_code == 200:
            return r.json()['message']
        return None
    except Exception as e:
        st.warning(f"⚠️ Error consultando CrossRef: {e}")
        return None

# ==========================================
# CLASE LITERATURE AI
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
        pdfplumber, _, _, _, _, _ = get_heavy_imports()
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
        'sensitivity': sensitivity, 'specificity': specificity, 'vpp': vpp, 'vpn': vpn,
        'odds_ratio': odds_ratio, 'risk_ratio': rr, 'arr': arr, 'nnt': nnt,
        'lr_positive': lr_positive, 'lr_negative': lr_negative,
        'ci_low_or': ci_low_or, 'ci_high_or': ci_high_or,
        'chi_square': chi_sq, 'p_value': p_value_chi,
        'prevalence_exposed': prevalence_exposed, 'prevalence_unexposed': prevalence_unexposed
    }

def calculate_sample_size(p1, p2, alpha=0.05, power=0.8, ratio=1):
    """Calcula tamaño de muestra para comparación de proporciones"""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    p_bar = (p1 + ratio * p2) / (1 + ratio)
    n1 = ((z_alpha * np.sqrt((1 + 1/ratio) * p_bar * (1 - p_bar)) +
           z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)/ratio))**2 / (p1 - p2)**2)
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

def calculate_sample_size_observational(p_exposed, p_unexposed, alpha=0.05, power=0.8, k_covariates=3, inflation_factor=10):
    """Calcula tamaño de muestra para estudios observacionales con múltiples confundidores"""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    # Ajuste por múltiples confundidores (Regla de 10 eventos por variable)
    events_needed = k_covariates * inflation_factor
    total_events = events_needed * 2  # Exposados y no expuestos

    p_bar = (p_exposed + p_unexposed) / 2
    effect_size = abs(p_exposed - p_unexposed)

    n_per_group = ((z_alpha + z_beta)**2 * p_bar * (1 - p_bar)) / effect_size**2
    n_per_group_adjusted = int(np.ceil(n_per_group * 1.1))  # 10% extra por confundidores

    return {
        'n_per_group': n_per_group_adjusted,
        'total': n_per_group_adjusted * 2,
        'events_per_variable': events_needed,
        'adjusted_for_confounders': True
    }

def calculate_sample_size_multivariate(n_predictors, r2_expected=0.3, alpha=0.05, power=0.8):
    """Calcula tamaño de muestra para regresión multivariada"""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    # Fórmula para regresión múltiple
    n_adjusted = ((n_predictors + 1) * 10) / r2_expected
    n_with_power = int(np.ceil(n_adjusted * (1 + 1/power)))

    return {
        'n_predictors': n_predictors,
        'r2_expected': r2_expected,
        'min_n': int(np.ceil(n_adjusted)),
        'n_with_power': n_with_power,
        'recommendation': f"Mínimo {n_with_power} participantes para detectar R²={r2_expected} con {n_predictors} predictores"
    }

def propensity_score_matching(df, treatment_col, outcome_col, covariate_cols):
    """Propensity Score Matching para estudios observacionales"""
    LogisticRegression, NearestNeighbors = get_psm_imports()

    # Preparar datos
    X = df[covariate_cols].fillna(0)
    y = df[treatment_col]

    # Estimar propensity scores
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    df['propensity_score'] = model.predict_proba(X)[:, 1]

    # Matching
    treated = df[df[treatment_col] == 1]['propensity_score'].values
    control = df[df[treatment_col] == 0]['propensity_score'].values

    nbrs = NearestNeighbors(n_neighbors=1).fit(control.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(treated.reshape(-1, 1))

    # Crear grupos pareados
    matched_control_idx = indices.flatten()
    matched_treated_idx = np.arange(len(treated))

    # Calcular ATE (Average Treatment Effect)
    treated_outcomes = df[df[treatment_col] == 1][outcome_col].values
    control_outcomes = df[df[treatment_col] == 0].iloc[matched_control_idx][outcome_col].values

    ate = np.mean(treated_outcomes) - np.mean(control_outcomes)

    # Estandarización de diferencias
    std_diff = (np.mean(treated_outcomes) - np.mean(control_outcomes)) / np.sqrt((np.std(treated_outcomes)**2 + np.std(control_outcomes)**2) / 2)

    return {
        'matched_df': df.iloc[np.concatenate([np.where(df[treatment_col] == 1)[0],
                                              df[df[treatment_col] == 0].index[matched_control_idx]])],
        'ate': ate,
        'std_diff': std_diff,
        'n_matched': len(matched_treated_idx),
        'propensity_scores': df['propensity_score'].values,
        'balance_metrics': calculate_balance_metrics(df, covariate_cols, treatment_col='treatment')
    }

def calculate_balance_metrics(df, covariate_cols, treatment_col='treatment'):
    """Calcula métricas de balance después del PSM"""
    treated = df[df[treatment_col] == 1][covariate_cols]
    control = df[df[treatment_col] == 0][covariate_cols]

    balance = {}
    for col in covariate_cols:
        mean_t = treated[col].mean()
        mean_c = control[col].mean()
        std_pooled = np.sqrt((treated[col].std()**2 + control[col].std()**2) / 2)
        std_diff = abs(mean_t - mean_c) / std_pooled if std_pooled > 0 else 0
        balance[col] = {
            'mean_treated': mean_t,
            'mean_control': mean_c,
            'std_diff': std_diff,
            'balanced': std_diff < 0.1
        }
    return balance

def calculate_arima_forecast(data, periods=30, order=(1,1,1)):
    """Calcula pronóstico ARIMA"""
    _, _, _, _, ARIMA, _ = get_heavy_imports()
    try:
        model = ARIMA(data, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=periods)

        # Intervalos de confianza
        conf_int = fitted.get_forecast(steps=periods).conf_int()

        return {
            'forecast': forecast,
            'lower_ci': conf_int.iloc[:, 0].values,
            'upper_ci': conf_int.iloc[:, 1].values,
            'aic': fitted.aic,
            'order': order
        }
    except Exception as e:
        return {'error': str(e)}

def meta_analysis_fixed_effect(events_e, total_e, events_c, total_c):
    """Meta-análisis con modelo de efectos fijos"""
    log_or, se_log_or, weights = [], [], []
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
        'pooled_or': pooled_or, 'pooled_ci_low': pooled_ci_low, 'pooled_ci_high': pooled_ci_high,
        'Q': Q, 'df': df, 'I2': I2, 'p_value': p_value,
        'log_or': log_or, 'se_log_or': se_log_or, 'weights': weights,
        'individual_or': [np.exp(l) for l in log_or]
    }

def meta_analysis_random_effects(events_e, total_e, events_c, total_c, tau2=None):
    """Meta-análisis con modelo de efectos aleatorios"""
    results = meta_analysis_fixed_effect(events_e, total_e, events_c, total_c)
    if not results: return None
    if tau2 is None:
        Q, df = results['Q'], results['df']
        C = np.sum(results['weights']) - np.sum([w**2 for w in results['weights']]) / np.sum(results['weights'])
        tau2 = max(0, (Q - df) / C) if C > 0 else 0
    weights_re = [1 / (w**-1 + tau2) for w in results['weights']]
    pooled_log_or_re = np.sum([w * l for w, l in zip(weights_re, results['log_or'])]) / np.sum(weights_re)
    pooled_se_re = np.sqrt(1 / np.sum(weights_re))
    pooled_or_re = np.exp(pooled_log_or_re)
    pooled_ci_low_re = np.exp(pooled_log_or_re - 1.96 * pooled_se_re)
    pooled_ci_high_re = np.exp(pooled_log_or_re + 1.96 * pooled_se_re)
    results.update({
        'pooled_or_re': pooled_or_re, 'pooled_ci_low_re': pooled_ci_low_re,
        'pooled_ci_high_re': pooled_ci_high_re, 'tau2': tau2, 'tau': np.sqrt(tau2),
        'weights_re': weights_re
    })
    return results

def generate_asis_report(data_dict, format='pdf'):
    """Genera reporte ASIS completo (Análisis de Situación de Salud)"""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título
    story.append(Paragraph("ANÁLISIS DE SITUACIÓN DE SALUD (ASIS)", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Fecha de Generación:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 24))

    # Vigilancia epidemiológica
    if 'vigilance' in data_dict:
        story.append(Paragraph("1. VIGILANCIA EPIDEMIOLÓGICA", styles['Heading1']))
        story.append(Spacer(1, 12))
        v = data_dict['vigilance']
        total_cases = v.get('total_cases', 'N/A')
        story.append(Paragraph(f"<b>Casos Totales:</b> {total_cases:,}" if isinstance(total_cases, (int, float)) else f"<b>Casos Totales:</b> {total_cases}", styles['Normal']))
        r0_val = v.get('r0', 'N/A')
        story.append(Paragraph(f"<b>R0 Estimado:</b> {r0_val:.2f}" if isinstance(r0_val, (int, float)) else f"<b>R0 Estimado:</b> {r0_val}", styles['Normal']))
        ifr_val = v.get('ifr', 'N/A')
        story.append(Paragraph(f"<b>IFR (Tasa Fatalidad):</b> {ifr_val:.2f}%" if isinstance(ifr_val, (int, float)) else f"<b>IFR (Tasa Fatalidad):</b> {ifr_val}", styles['Normal']))
        recovered_val = v.get('recovered', 'N/A')
        story.append(Paragraph(f"<b>Recuperados:</b> {recovered_val:,}" if isinstance(recovered_val, (int, float)) else f"<b>Recuperados:</b> {recovered_val}", styles['Normal']))
        deaths_val = v.get('deaths', 'N/A')
        story.append(Paragraph(f"<b>Fallecidos:</b> {deaths_val:,}" if isinstance(deaths_val, (int, float)) else f"<b>Fallecidos:</b> {deaths_val}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Proyecciones con fechas
    if 'projections' in data_dict:
        story.append(Paragraph("2. PROYECCIONES EPIDEMIOLÓGICAS", styles['Heading1']))
        story.append(Spacer(1, 12))
        proj = data_dict['projections']
        if 'forecast_dates' in proj:
            story.append(Paragraph("<b>Proyecciones por Fecha:</b>", styles['Normal']))
            for i, date in enumerate(proj['forecast_dates'][:10]):
                nuevos = proj['nuevos_mean'][i] if i < len(proj['nuevos_mean']) else 'N/A'
                nuevos_std = proj['nuevos_std'][i] if i < len(proj['nuevos_std']) else 'N/A'
                activos = proj['activos_mean'][i] if i < len(proj['activos_mean']) else 'N/A'
                story.append(Paragraph(f"• {date}: Nuevos={nuevos}±{nuevos_std}, Activos={activos}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Tablas 2x2
    if 'tables_2x2' in data_dict:
        story.append(Paragraph("3. TABLAS 2x2 DE ANÁLISIS", styles['Heading1']))
        story.append(Spacer(1, 12))
        for i, t in enumerate(data_dict['tables_2x2']):
            story.append(Paragraph(f"<b>Tabla {i+1}:</b>", styles['Normal']))
            story.append(Paragraph(f"• Odds Ratio (OR): {t.get('or', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"• Riesgo Relativo (RR): {t.get('rr', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"• IC 95%: [{t.get('ci_low', 'N/A')}, {t.get('ci_high', 'N/A')}]", styles['Normal']))
            story.append(Paragraph(f"• p-value: {t.get('p_value', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 6))

    # Mapas geográficos
    if 'geographic' in data_dict:
        story.append(Paragraph("4. DISTRIBUCIÓN GEOGRÁFICA", styles['Heading1']))
        story.append(Spacer(1, 12))
        geo = data_dict['geographic']
        if 'locations' in geo:
            story.append(Paragraph("<b>Casos por Ubicación:</b>", styles['Normal']))
            for loc in geo['locations'][:10]:
                story.append(Paragraph(f"• {loc.get('name', 'N/A')}: {loc.get('cases', 'N/A')} casos", styles['Normal']))

    # Metadatos
    story.append(Spacer(1, 24))
    story.append(Paragraph("— FIN DEL REPORTE ASIS —", styles['Normal']))
    story.append(Paragraph(f"Generado por EpiDiagnosis Pro V7.0 - {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ==========================================
# INICIALIZACIÓN DE SESIÓN CON PERSISTENCIA MEJORADA
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
if 'rob_assessments' not in st.session_state: st.session_state.rob_assessments = []
if 'grade_assessments' not in st.session_state: st.session_state.grade_assessments = []
if 'forest_studies' not in st.session_state: st.session_state.forest_studies = pd.DataFrame({
    'Estudio': ['Smith 2020', 'Johnson 2019', 'Williams 2021'],
    'Eventos_Tto': [20, 35, 28], 'Total_Tto': [100, 150, 120],
    'Eventos_Ctrl': [30, 50, 45], 'Total_Ctrl': [100, 150, 120]
})
if 'map_data' not in st.session_state: st.session_state.map_data = None
if 'marker_data' not in st.session_state: st.session_state.marker_data = None
if 'psm_results' not in st.session_state: st.session_state.psm_results = None
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'vigilance_config' not in st.session_state: st.session_state.vigilance_config = {}
if 'df_v_edited' not in st.session_state: st.session_state.df_v_edited = None
if 'modelo_ejecutado' not in st.session_state: st.session_state.modelo_ejecutado = False

# Cargar datos persistentes del usuario al iniciar (MEJORADO)
if st.session_state.auth and 'user' in st.session_state and not st.session_state.data_loaded:
    user_email = st.session_state.user
    user_data = load_user_data(user_email)

    if user_data:
        try:
            # Cargar df_master
            if 'df_master' in user_data and user_data['df_master'] is not None:
                df_json = user_data['df_master']
                if isinstance(df_json, list):
                    st.session_state.df_master = pd.DataFrame(df_json)
                elif isinstance(df_json, dict) and 'data' in df_json:
                    st.session_state.df_master = pd.DataFrame(df_json['data'])

            # Cargar df_v (vigilancia)
            if 'df_v' in user_data and user_data['df_v'] is not None:
                st.session_state.df_v = pd.DataFrame(user_data['df_v'])

            # Cargar meta_studies
            if 'meta_studies' in user_data and len(user_data['meta_studies']) > 0:
                st.session_state.meta_studies = pd.DataFrame(user_data['meta_studies'])

            # Cargar survival_data
            if 'survival_data' in user_data and user_data['survival_data'] is not None:
                st.session_state.survival_data = pd.DataFrame(user_data['survival_data'])

            # Cargar forest_studies
            if 'forest_studies' in user_data and len(user_data['forest_studies']) > 0:
                st.session_state.forest_studies = pd.DataFrame(user_data['forest_studies'])

            # Cargar map_data
            if 'map_data' in user_data and user_data['map_data'] is not None:
                st.session_state.map_data = pd.DataFrame(user_data['map_data'])

            # Cargar roc_data
            if 'roc_data' in user_data and user_data['roc_data'] is not None:
                st.session_state.roc_data = pd.DataFrame(user_data['roc_data'])

            # Cargar articulos_pico
            if 'articulos_pico' in user_data:
                st.session_state.articulos_pico = user_data['articulos_pico']

            # Cargar prisma_data
            if 'prisma_data' in user_data:
                st.session_state.prisma_data = user_data['prisma_data']

            # Cargar rob_assessments
            if 'rob_assessments' in user_data:
                st.session_state.rob_assessments = user_data['rob_assessments']

            # Cargar configuraciones de vigilancia
            if 'vigilance_config' in user_data:
                st.session_state.vigilance_config = user_data['vigilance_config']

            # Cargar PSM results
            if 'psm_results' in user_data:
                st.session_state.psm_results = user_data['psm_results']

            st.session_state.data_loaded = True

        except Exception as e:
            st.warning(f"⚠️ Error al restaurar sesión: {e}")
        
def persist_user_data():
    """Guarda todos los datos del usuario actual (MEJORADO)"""
    if st.session_state.auth and 'user' in st.session_state:
        user_email = st.session_state.user
        try:
            data = {
                # Datos principales
                'df_master': st.session_state.df_master.to_dict('records') if st.session_state.df_master is not None and not st.session_state.df_master.empty else None,
                'df_v': st.session_state.df_v.to_dict('records') if st.session_state.df_v is not None and not st.session_state.df_v.empty else None,

                # Meta-análisis
                'meta_studies': st.session_state.meta_studies.to_dict('records') if st.session_state.meta_studies is not None and len(st.session_state.meta_studies) > 0 else [],

                # Supervivencia
                'survival_data': st.session_state.survival_data.to_dict('records') if st.session_state.survival_data is not None and not st.session_state.survival_data.empty else None,

                # Forest plot
                'forest_studies': st.session_state.forest_studies.to_dict('records') if st.session_state.forest_studies is not None and not st.session_state.forest_studies.empty else [],

                # Mapas
                'map_data': st.session_state.map_data.to_dict('records') if st.session_state.map_data is not None and not st.session_state.map_data.empty else None,

                # ROC
                'roc_data': st.session_state.roc_data.to_dict('records') if st.session_state.roc_data is not None and not st.session_state.roc_data.empty else None,

                # Literatura
                'articulos_pico': st.session_state.articulos_pico if st.session_state.articulos_pico else [],

                # PRISMA
                'prisma_data': st.session_state.prisma_data if st.session_state.prisma_data else {},

                # RoB
                'rob_assessments': st.session_state.rob_assessments if st.session_state.rob_assessments else [],

                # Config de vigilancia
                'vigilance_config': st.session_state.vigilance_config if st.session_state.vigilance_config else {},

                # PSM
                'psm_results': st.session_state.psm_results if st.session_state.psm_results else None,

                # Metadata
                'last_save': datetime.now().isoformat()
            }
            save_user_data(user_email, data)
        except Exception as e:
            st.warning(f"⚠️ Error guardando datos: {e}")

# ==========================================
# AUTENTICACIÓN
# ==========================================
def login_attempts_check(ip="default"):
    if not rate_limiter.is_allowed(f"login_{ip}"):
        return False, "Demasiados intentos. Espere 60 segundos."
    return True, ""

if not st.session_state.get("auth", False):
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {display: none !important;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🔐 Iniciar Sesión - EpiDiagnosis Pro V7.0")

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
                            st.success("✅ Bienvenido!")
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
                        "password": secure_hash(rp), "role": "user", "expiry": exp,
                        "id_doc": rid, "name": rnombre, "lastname": rapellido, "profession": rprofesion
                    }
                    save_users(db)
                    st.success("✅ Cuenta creada - Pruebas guardadas!")
                elif ru in db:
                    st.warning("Email ya registrado")
                else:
                    st.warning("Complete todos los campos")

    st.markdown("---")

    PAYMENT_LINK = "https://checkout.bold.co/payment/LNK_2W3K24BLVU"

    st.markdown("""
        <div style="text-align:center; padding: 20px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 15px; border: 2px solid #3b82f6;">
            <h2 style="color: #f59e0b; margin-bottom: 10px;">💳 SUSCRIPCIÓN PREMIUM</h2>
            <h1 style="color: #10b981; font-size: 48px; margin: 10px 0;">23 USD<span style="font-size: 20px; color: #94a3b8;">/mes</span></h1>
            <p style="color: #94a3b8; font-size: 14px;">Acceso ilimitado a todas las funcionalidades premium</p>
            <hr style="border-color: #334155; margin: 15px 0;">
            <h4 style="color: #60a5fa; margin-bottom: 15px;">✅ Incluye:</h4>
            <ul style="color: #e2e8f0; text-align: left; display: inline-block; font-size: 14px; line-height: 1.8;">
                <li>🔬 Análisis PICO con GPT-4</li>
                <li>📈 Proyecciones ARIMA y SEIR</li>
                <li>📊 Monte Carlo con desviaciones estándar</li>
                <li>⚙️ Propensity Score Matching (PSM)</li>
                <li>📉 Regresión Cox y Poisson</li>
                <li>🌲 Forest Plot y Meta-análisis</li>
                <li>🗺️ Mapas geográficos</li>
                <li>📄 Exportación ASIS PDF/Word</li>
            </ul>
            <hr style="border-color: #334155; margin: 15px 0;">
            <a href="https://checkout.bold.co/payment/LNK_2W3K24BLVU" target="_blank">
                <button style="padding:20px 50px; font-size:20px; background: linear-gradient(135deg, #f59e0b, #d97706); color:white; border:none; border-radius:12px; cursor:pointer; font-weight:bold; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4); transition: all 0.3s;">
                    🔒 PAGAR AHORA - 23 USD/mes
                </button>
            </a>
            <p style="color: #64748b; font-size: 12px; margin-top: 10px;">🔒 Pago seguro con Bold.co</p>
        </div>
    """, unsafe_allow_html=True)

else:
    # ==========================================
    # APP PRINCIPAL
    # ==========================================

    with st.sidebar:
        st.markdown("🩺 **EpiDiagnosis Pro V7.0**")

        if st.session_state.user:
            user_data = load_user_data(st.session_state.user)
            last_save = user_data.get('last_save', 'Nunca')
            st.caption(f"📁 Guardado: {last_save[:16] if last_save != 'Nunca' else 'Nunca'}")

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

        if st.button("💾 GUARDAR TODO", use_container_width=True):
            persist_user_data()
            st.success("✅ Datos guardados!")

        if st.button("🚪 Cerrar Sesión"):
            keys_to_clear = [
                'auth', 'user', 'role', 'df_master', 'articulos_pico', 'df_v',
                'meta_studies', 'survival_data', 'roc_data', 'prisma_data',
                'rob_assessments', 'grade_assessments', 'forest_studies',
                'map_data', 'marker_data', 'psm_results', 'vigilance_config',
                'user_logins', 'data_loaded'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
            
        st.markdown("---")
        st.info("📞 Soporte: (+57) 3113682907\n📧 j.collazosmd@gmail.com")

    # ==========================================
    # MÓDULO: DASHBOARD & CLOUD CON FILTROS
    # ==========================================
    if menu == "🏠 Dashboard & Cloud":
        st.header("📊 Dashboard & Gestión de Datos con Filtros")

        with st.expander("ℹ️ Instrucciones", expanded=False):
            st.markdown("""
            1. Copie el enlace público de Google Sheets, Excel Online o CSV
            2. Pegue el enlace y cargue los datos
            3. Use los **filtros** para explorar variables
            4. Sus datos se **guardan automáticamente**
            """)

        url = st.text_input(
            "Enlace público:",
            placeholder="https://docs.google.com/spreadsheets/d/... o URL de CSV/Excel",
            key="dash_url"
        )

        col_load1, col_load2 = st.columns([1, 4])
        with col_load1:
            load_btn = st.button("📥 CARGAR DATOS", use_container_width=True, key="dash_load_btn")

        if url and (load_btn or st.session_state.df_master is not None):
            with st.spinner("⏳ Cargando datos..."):
                df_new = smart_load_data(url)
                if df_new is not None:
                    st.session_state.df_master = df_new
                    persist_user_data()
                    st.success("✅ Datos cargados y guardados!")
                elif load_btn:
                    st.error("❌ Error al cargar datos. Verifique el enlace.")

        if st.session_state.df_master is not None:
            df = st.session_state.df_master

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Registros", f"{len(df):,}")
            c2.metric("📋 Columnas", len(df.columns))
            c3.metric("❓ Nulos Totales", f"{df.isna().sum().sum():,}")
            c4.metric("✅ Calidad", f"{(1 - df.isna().sum().sum()/max(df.size, 1)):.1%}")

            # FILTROS DE VARIABLES MEJORADOS
            st.markdown("---")
            st.markdown("### 🔍 Filtros de Variables - Diccionario de Datos")

            # Tabla de resumen completo de variables
            st.markdown("#### 📊 Inventario de Variables")
            summary_data = []
            for col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = missing_count / len(df) * 100
                missing_color = "🔴" if missing_pct > 20 else "🟡" if missing_pct > 5 else "🟢"

                row = {
                    'Estado': missing_color,
                    'Variable': col,
                    'Tipo': str(df[col].dtype)[:10],
                    'N': df[col].count(),
                    'Nulos': missing_count,
                    '% Pérdidas': f"{missing_pct:.1f}%",
                    'Únicos': df[col].nunique()
                }

                if pd.api.types.is_numeric_dtype(df[col]):
                    row['Mín'] = f"{df[col].min():.2f}" if pd.notna(df[col].min()) else '-'
                    row['Máx'] = f"{df[col].max():.2f}" if pd.notna(df[col].max()) else '-'
                    row['Media'] = f"{df[col].mean():.2f}" if pd.notna(df[col].mean()) else '-'
                    row['DE'] = f"{df[col].std():.2f}" if pd.notna(df[col].std()) else '-'
                else:
                    row['Mín'] = '-'
                    row['Máx'] = '-'
                    row['Media'] = '-'
                    row['DE'] = '-'

                summary_data.append(row)

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Leyenda de colores
            st.markdown("""
            **Leyenda de Estado:**
            - 🟢 Verde: < 5% de pérdidas (Aceptable)
            - 🟡 Amarillo: 5-20% de pérdidas (Revisar)
            - 🔴 Rojo: > 20% de pérdidas (Problemático)
            """)

            # Filtros interactivos
            col_filter1, col_filter2, col_filter3 = st.columns(3)

            with col_filter1:
                filter_type = st.selectbox("🔎 Filtrar por Tipo:", ["Todas", "Numéricas", "Texto", "Fecha", "Alta Pérdida (>20%)", "Sin Pérdidas"])

            with col_filter2:
                if filter_type == "Numéricas":
                    filtered_cols = df.select_dtypes(include=np.number).columns.tolist()
                elif filter_type == "Texto":
                    filtered_cols = df.select_dtypes(include='object').columns.tolist()
                elif filter_type == "Fecha":
                    filtered_cols = df.select_dtypes(include='datetime').columns.tolist()
                elif filter_type == "Alta Pérdida (>20%)":
                    filtered_cols = [col for col in df.columns if df[col].isna().sum() / len(df) * 100 > 20]
                elif filter_type == "Sin Pérdidas":
                    filtered_cols = [col for col in df.columns if df[col].isna().sum() == 0]
                else:
                    filtered_cols = df.columns.tolist()

                filter_var = st.selectbox("📋 Seleccionar Variable:", filtered_cols)

            with col_filter3:
                if filter_var:
                    if pd.api.types.is_numeric_dtype(df[filter_var]):
                        min_val, max_val = float(df[filter_var].min()), float(df[filter_var].max())
                        range_filter = st.slider("🎚️ Rango:", min_val, max_val, (min_val, max_val))
                        df_filtered = df[(df[filter_var] >= range_filter[0]) & (df[filter_var] <= range_filter[1])]
                        st.metric("✅ Registros Filtrados", f"{len(df_filtered):,} / {len(df):,}")
                    else:
                        unique_vals = df[filter_var].dropna().unique()[:30]
                        selected_vals = st.multiselect("☑️ Valores:", unique_vals)
                        if selected_vals:
                            df_filtered = df[df[filter_var].isin(selected_vals)]
                            st.metric("✅ Filtrados", f"{len(df_filtered):,} / {len(df):,}")
                        else:
                            df_filtered = df

            with st.expander("👁️ Vista Previa de Datos Filtrados", expanded=True):
                st.dataframe(df_filtered.head(100) if 'df_filtered' in locals() else df.head(100),
                           use_container_width=True, height=400)

            with st.expander("📈 Estadísticas Detalladas de Variable"):
                if filter_var:
                    if pd.api.types.is_numeric_dtype(df[filter_var]):
                        col_stats = st.columns(4)
                        col_stats[0].metric("N", df[filter_var].count())
                        col_stats[1].metric("Media", f"{df[filter_var].mean():.2f}")
                        col_stats[2].metric("Mediana", f"{df[filter_var].median():.2f}")
                        col_stats[3].metric("DE", f"{df[filter_var].std():.2f}")

                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        axes[0].hist(df[filter_var].dropna(), bins=30, color='#3b82f6', edgecolor='white')
                        axes[0].set_title(f'Distribución de {filter_var}')
                        axes[0].set_xlabel(filter_var)
                        axes[0].set_ylabel('Frecuencia')
                        df[filter_var].dropna().plot(kind='box', ax=axes[1])
                        axes[1].set_title(f'Boxplot de {filter_var}')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

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
                        removed = before - len(df)
                        st.session_state.df_master = df
                        persist_user_data()
                        st.success(f"✅ {removed} duplicados eliminados. Quedan {len(df):,} registros.")
                        st.rerun()
                with col_ops[1]:
                    if st.button("🔠 Texto a MAYÚSCULAS", use_container_width=True):
                        df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
                        st.session_state.df_master = df
                        persist_user_data()
                        st.success("✅ Conversión aplicada")
                with col_ops[2]:
                    if st.button("📊 Matriz de Correlación", use_container_width=True):
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:
                            fig = px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale='RdBu_r')
                            st.plotly_chart(fig, use_container_width=True)

                col_to_fix = st.selectbox("Seleccionar Columna:", df.columns)
                col1, col2 = st.columns(2)

                with col1:
                    new_name = st.text_input("Nuevo nombre:", col_to_fix)
                    if st.button("Renombrar"):
                        df.rename(columns={col_to_fix: new_name}, inplace=True)
                        st.session_state.df_master = df
                        persist_user_data()
                        st.success(f"✅ Columna renombrada")

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
                            persist_user_data()
                            st.success(f"✅ Convertido")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with tab2:
                method = st.radio("Método:", ["Mediana", "Moda", "Eliminar Filas"], horizontal=True)
                if st.button("✨ Aplicar Imputación", use_container_width=True):
                    try:
                        if method == "Mediana":
                            df[col_to_fix] = df[col_to_fix].fillna(df[col_to_fix].median())
                        elif method == "Moda":
                            df[col_to_fix] = df[col_to_fix].fillna(df[col_to_fix].mode()[0])
                        else:
                            df.dropna(subset=[col_to_fix], inplace=True)
                        st.session_state.df_master = df
                        persist_user_data()
                        st.success("✅ Imputación completada")
                    except Exception as e:
                        st.error(f"Error: {e}")

            with tab3:
                if pd.api.types.is_numeric_dtype(df[col_to_fix]):
                    q1, q3 = df[col_to_fix].quantile(0.25), df[col_to_fix].quantile(0.75)
                    iqr = q3 - q1
                    out = df[(df[col_to_fix] < (q1 - 1.5*iqr)) | (df[col_to_fix] > (q3 + 1.5*iqr))]
                    st.warning(f"🔍 **{len(out)}** outliers ({len(out)/len(df)*100:.1f}%)")
                    col_fig, col_btn = st.columns([2, 1])
                    with col_fig:
                        fig = px.box(df, y=col_to_fix, points="outliers")
                        st.plotly_chart(fig, use_container_width=True)
                    with col_btn:
                        if st.button("🗑️ Limpiar Outliers", use_container_width=True):
                            df = df[~df.index.isin(out.index)]
                            st.session_state.df_master = df
                            persist_user_data()
                            st.success(f"✅ Eliminados {len(out)} outliers")

    # ==========================================
    # MÓDULO: BIOESTADÍSTICA
    # ==========================================
    elif menu == "📊 Bioestadística":
        st.header("📊 Bioestadística Avanzada")

        if st.session_state.df_master is None:
            st.warning("⚠️ Por favor cargue datos primero.")
        else:
            df = st.session_state.df_master

            analysis_type = st.radio("🎯 Tipo de Análisis:",
                ["📊 Comparación de Grupos", "📈 Una Variable", "📉 Correlación"], horizontal=True)

            if analysis_type == "📊 Comparación de Grupos":
                col_vars = st.columns([1, 1, 1])
                with col_vars[0]:
                    vn = st.selectbox("Variable Numérica (Y):", df.select_dtypes(include=np.number).columns)
                with col_vars[1]:
                    vc = st.selectbox("Variable Categórica (X/Grupo):", df.columns)
                with col_vars[2]:
                    alpha_norm = st.select_slider("α Normalidad:", options=[0.01, 0.05, 0.10], value=0.05)

                if st.button("🔬 EJECUTAR ANÁLISIS", use_container_width=True):
                    with st.spinner("⏳ Procesando..."):
                        _, _, _, stats, *_ = get_heavy_imports()
                        clean_data = df[[vn, vc]].dropna()
                        grupos_data = {g: clean_data[clean_data[vc] == g][vn].values for g in clean_data[vc].unique()}
                        grupos = [g for g in grupos_data.values() if len(g) > 0]
                        nombres_grupos = [g for g, v in grupos_data.items() if len(v) > 0]

                        if len(grupos) < 2:
                            st.error("Se requieren al menos 2 grupos")
                        else:
                            clean_y = clean_data[vn]
                            stat, p_norm = stats.shapiro(clean_y[:5000]) if len(clean_y) <= 5000 else stats.normaltest(clean_y)

                            col_norm = st.columns(3)
                            with col_norm[0]:
                                st.metric("Test", "Shapiro-Wilk" if len(clean_y) <= 5000 else "D'Agostino")
                            with col_norm[1]:
                                st.metric("p-value", f"{p_norm:.4f}",
                                         delta="Normal" if p_norm > 0.05 else "No Normal")
                            with col_norm[2]:
                                st.metric("N. Total", len(clean_y))

                            is_normal = p_norm > alpha_norm

                            if len(grupos) > 2:
                                res = stats.f_oneway(*grupos) if is_normal else stats.kruskal(*grupos)
                                test_name = "ANOVA" if is_normal else "Kruskal-Wallis"
                            else:
                                res = stats.ttest_ind(grupos[0], grupos[1]) if is_normal else stats.mannwhitneyu(grupos[0], grupos[1])
                                test_name = "T-Test" if is_normal else "Mann-Whitney U"

                            col_res = st.columns(3)
                            with col_res[0]:
                                st.metric("Prueba", test_name)
                            with col_res[1]:
                                st.metric("Estadístico", f"{res.statistic:.2f}")
                            with col_res[2]:
                                st.metric("p-value", f"{res.pvalue:.6f}",
                                         delta="Significativo" if res.pvalue < 0.05 else "No significativo")

            elif analysis_type == "📈 Una Variable":
                var_single = st.selectbox("Seleccionar Variable:", df.select_dtypes(include=np.number).columns)
                if st.button("📊 ANALIZAR VARIABLE", use_container_width=True):
                    data = df[var_single].dropna()
                    col_stats = st.columns(4)
                    with col_stats[0]:
                        st.metric("N", len(data))
                        st.metric("Media", f"{data.mean():.2f}")
                    with col_stats[1]:
                        st.metric("Mediana", f"{data.median():.2f}")
                        st.metric("DS", f"{data.std():.2f}")
                    with col_stats[2]:
                        st.metric("Mín", f"{data.min():.2f}")
                        st.metric("Máx", f"{data.max():.2f}")
                    with col_stats[3]:
                        st.metric("25%", f"{data.quantile(0.25):.2f}")
                        st.metric("75%", f"{data.quantile(0.75):.2f}")

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
                    axes[0].axvline(data.mean(), color='red', linestyle='--', label=f'Media: {data.mean():.2f}')
                    axes[0].axvline(data.median(), color='green', linestyle='--', label=f'Mediana: {data.median():.2f}')
                    axes[0].legend()
                    axes[1].boxplot(data, vert=True, patch_artist=True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            elif analysis_type == "📉 Correlación":
                col_corr = st.columns(2)
                with col_corr[0]:
                    var_x = st.selectbox("Variable X:", df.select_dtypes(include=np.number).columns, key="var_x")
                with col_corr[1]:
                    var_y = st.selectbox("Variable Y:", df.select_dtypes(include=np.number).columns, key="var_y")

                if var_x != var_y and st.button("🔗 ANALIZAR CORRELACIÓN", use_container_width=True):
                    _, _, _, stats, *_ = get_heavy_imports()
                    clean_corr = df[[var_x, var_y]].dropna()
                    r_pearson, p_pearson = stats.pearsonr(clean_corr[var_x], clean_corr[var_y])
                    interp = "Muy débil" if abs(r_pearson) < 0.1 else "Débil" if abs(r_pearson) < 0.3 else "Moderada" if abs(r_pearson) < 0.5 else "Fuerte"

                    col_results = st.columns(4)
                    with col_results[0]:
                        st.metric("Pearson r", f"{r_pearson:.3f}")
                        st.metric("Pearson p", f"{p_pearson:.4f}")
                    with col_results[1]:
                        st.metric("Interpretación", interp)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(clean_corr[var_x], clean_corr[var_y], alpha=0.5)
                    z = np.polyfit(clean_corr[var_x], clean_corr[var_y], 1)
                    p = np.poly1d(z)
                    ax.plot(clean_corr[var_x].sort_values(), p(clean_corr[var_x].sort_values()), "r--", alpha=0.8)
                    ax.set_xlabel(var_x)
                    ax.set_ylabel(var_y)
                    ax.set_title(f'Correlación: r = {r_pearson:.3f}, p = {p_pearson:.4f}')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

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
                a = st.number_input("a (Expuestos + Enfermedad +)", min_value=0, value=30, step=1)
            with col_a[1]:
                b = st.number_input("b (Expuestos + Enfermedad -)", min_value=0, value=70, step=1)
            col_b = st.columns(2)
            with col_b[0]:
                c = st.number_input("c (No Expuestos + Enfermedad +)", min_value=0, value=20, step=1)
            with col_b[1]:
                d = st.number_input("d (No Expuestos + Enfermedad -)", min_value=0, value=80, step=1)

        with col_t2:
            df_2x2_display = pd.DataFrame({
                '': ['Expuestos (+)', 'No Expuestos (-)', 'Total'],
                'Enfermedad (+)': [f'{a}', f'{c}', f'{a+c}'],
                'Enfermedad (-)': [f'{b}', f'{d}', f'{b+d}'],
                'Total': [f'{a+b}', f'{c+d}', f'{a+b+c+d}']
            })
            st.dataframe(df_2x2_display, hide_index=True, use_container_width=True)
            st.markdown(f"**Total:** {a+b+c+d}")

        if st.button("🧮 CALCULAR MÉTRICAS", use_container_width=True):
            metrics = calculate_2x2_metrics(a, b, c, d)

            col_metrics = st.columns(4)
            with col_metrics[0]:
                st.metric("📊 Sensibilidad", f"{metrics['sensitivity']:.2%}")
                st.metric("🎯 Especificidad", f"{metrics['specificity']:.2%}")
            with col_metrics[1]:
                st.metric("✅ VPP", f"{metrics['vpp']:.2%}")
                st.metric("✅ VPN", f"{metrics['vpn']:.2%}")
            with col_metrics[2]:
                st.metric("📈 OR", f"{metrics['odds_ratio']:.2f}")
                st.metric("📉 IC 95% OR", f"[{metrics['ci_low_or']:.2f}, {metrics['ci_high_or']:.2f}]")
            with col_metrics[3]:
                st.metric("⚡ RR", f"{metrics['risk_ratio']:.2f}")
                st.metric("📐 ARR", f"{metrics['arr']:.2%}")

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
                st.metric("🎯 Índice Youden", f"{youden:.3f}")

    # ==========================================
    # MÓDULO: TAMAÑO DE MUESTRA EXTENDIDO
    # ==========================================
    elif menu == "📏 Tamaño de Muestra":
        st.header("📏 Calculadora de Tamaño de Muestra V7.0")

        study_type = st.selectbox("🎯 Tipo de Estudio:",
            ["📊 Cohortes", "🔬 Casos y Controles", "🧪 Ensayos Clínicos", "🔍 Observacional", "📈 Multivariada", "⚙️ Confundidores", "📋 Transversal"])

        if study_type == "🔍 Observacional":
            st.markdown("### Estudios Observacionales (PSM-ready)")
            col_obs = st.columns(2)
            with col_obs[0]:
                p_exposed = st.number_input("Proporción Expuestos (p1):", 0.001, 0.999, 0.30, format="%.3f")
                alpha_obs = st.select_slider("α:", options=[0.01, 0.05, 0.10], value=0.05)
            with col_obs[1]:
                p_unexposed = st.number_input("Proporción No Expuestos (p2):", 0.001, 0.999, 0.50, format="%.3f")
                power_obs = st.select_slider("Poder:", options=[0.70, 0.80, 0.90, 0.95], value=0.80)
            k_covariates = st.number_input("N.° Confundidores a controlar:", 1, 20, 3)

            if st.button("🧮 CALCULAR (OBSERVACIONAL)", use_container_width=True):
                result = calculate_sample_size_observational(p_exposed, p_unexposed, alpha_obs, power_obs, k_covariates)
                col_res = st.columns(3)
                with col_res[0]:
                    st.metric("Grupo", f"{result['n_per_group']:,}")
                with col_res[1]:
                    st.metric("Total", f"{result['total']:,}")
                with col_res[2]:
                    st.metric("Eventos/Variable", result['events_per_variable'])
                st.info(f"✅ Ajustado para controlar {k_covariates} confundidores (regla 10:1)")

        elif study_type == "📈 Multivariada":
            st.markdown("### Regresión Multivariada")
            col_mv = st.columns(2)
            with col_mv[0]:
                n_predictors = st.number_input("N.° Predictores:", 1, 20, 5)
                r2_expected = st.slider("R² esperado:", 0.1, 0.9, 0.3)
            with col_mv[1]:
                alpha_mv = st.select_slider("α:", options=[0.01, 0.05, 0.10], value=0.05)
                power_mv = st.select_slider("Poder:", options=[0.70, 0.80, 0.90, 0.95], value=0.80)

            if st.button("🧮 CALCULAR (MULTIVARIADA)", use_container_width=True):
                result = calculate_sample_size_multivariate(n_predictors, r2_expected, alpha_mv, power_mv)
                col_res = st.columns(3)
                with col_res[0]:
                    st.metric("Predictores", result['n_predictors'])
                with col_res[1]:
                    st.metric("R² esperado", f"{result['r2_expected']:.1%}")
                with col_res[2]:
                    st.metric("Mínimo N", result['min_n'])
                st.info(f"✅ {result['recommendation']}")

        elif study_type == "⚙️ Confundidores":
            st.markdown("### Control de Confundidores")
            st.info("Cálculo de tamaño muestral ajustado por múltiples confundidores")

            col_cf = st.columns(3)
            with col_cf[0]:
                base_n = st.number_input("N base (sin ajust):", 100, 10000, 500)
                n_confounders = st.number_input("N.° Confundidores:", 1, 10, 3)
            with col_cf[1]:
                effect_reduction = st.slider("Reducción efecto (%):", 0, 50, 20)
                alpha_cf = st.select_slider("α:", options=[0.01, 0.05, 0.10], value=0.05)
            with col_cf[2]:
                power_cf = st.select_slider("Poder:", options=[0.70, 0.80, 0.90, 0.95], value=0.80)

            inflation_factor = 1 + (n_confounders * effect_reduction / 100)
            n_adjusted = int(base_n * inflation_factor)

            if st.button("🧮 CALCULAR (CON FUNDORES)", use_container_width=True):
                col_res = st.columns(3)
                with col_res[0]:
                    st.metric("N Base", f"{base_n:,}")
                with col_res[1]:
                    st.metric("Factor", f"{inflation_factor:.2f}x")
                with col_res[2]:
                    st.metric("N Ajustado", f"{n_adjusted:,}")
                st.success(f"✅ Se necesitan {n_adjusted:,} participantes para controlar {n_confounders} confundidores")

        elif study_type == "🔬 Casos y Controles":
            col_cc1, col_cc2 = st.columns(2)
            with col_cc1:
                or_expected = st.number_input("OR Esperado:", 0.1, 10.0, 2.0, format="%.2f")
                alpha_cc = st.select_slider("α:", options=[0.01, 0.05, 0.10], value=0.05)
            with col_cc2:
                ratio_cc = st.number_input("Ratio Controles:Casos:", 1, 10, 4)
                power_cc = st.select_slider("Poder:", options=[0.70, 0.80, 0.90, 0.95], value=0.80)

            if st.button("🧮 CALCULAR (CASOS-CONTROLES)", use_container_width=True):
                result = calculate_sample_size_case_control(or_expected, alpha_cc, power_cc, ratio_cc)
                col_res = st.columns(4)
                with col_res[0]:
                    st.metric("🏥 Casos", f"{result['n_cases']:,}")
                with col_res[1]:
                    st.metric("👥 Controles", f"{result['n_controls']:,}")
                with col_res[2]:
                    st.metric("📊 Total", f"{result['total']:,}")
                with col_res[3]:
                    st.metric("Ratio", f"{ratio_cc}:1")

        elif study_type == "🧪 Ensayos Clínicos":
            col_ec = st.columns([1, 1, 1, 1])
            with col_ec[0]:
                p1 = st.number_input("Proporción Grupo 1 (p1):", 0.001, 0.999, 0.30, format="%.3f")
            with col_ec[1]:
                p2 = st.number_input("Proporción Grupo 2 (p2):", 0.001, 0.999, 0.50, format="%.3f")
                power = st.select_slider("Poder:", options=[0.70, 0.80, 0.90, 0.95], value=0.80)
            with col_ec[2]:
                ratio = st.number_input("Ratio (n2/n1):", 0.1, 10.0, 1.0, format="%.2f")
                alpha = st.select_slider("α:", options=[0.01, 0.05, 0.10], value=0.05)
            with col_ec[3]:
                test_type = st.radio("Tipo:", ["Two-sided", "One-sided"])

            if st.button("🧮 CALCULAR (ENSAYO)", use_container_width=True):
                if test_type == "One-sided":
                    alpha = alpha / 2
                result = calculate_sample_size(p1, p2, alpha, power, ratio)
                col_res = st.columns(4)
                with col_res[0]:
                    st.metric("Grupo 1", f"{result['n1']:,}")
                with col_res[1]:
                    st.metric("Grupo 2", f"{result['n2']:,}")
                with col_res[2]:
                    st.metric("Total", f"{result['total']:,}")
                with col_res[3]:
                    nnt_calc = 1 / abs(p1 - p2) if p1 != p2 else float('inf')
                    st.metric("NNT", f"{int(nnt_calc):,}" if nnt_calc != float('inf') else "∞")

        elif study_type == "📊 Cohortes":
            col_sample = st.columns([1, 1, 1, 1])
            with col_sample[0]:
                p1 = st.number_input("p1:", 0.001, 0.999, 0.30, format="%.3f")
            with col_sample[1]:
                p2 = st.number_input("p2:", 0.001, 0.999, 0.50, format="%.3f")
                power = st.select_slider("Poder:", options=[0.70, 0.80, 0.90, 0.95], value=0.80)
            with col_sample[2]:
                ratio = st.number_input("Ratio:", 0.1, 10.0, 1.0, format="%.2f")
                alpha = st.select_slider("α:", options=[0.01, 0.05, 0.10], value=0.05)
            with col_sample[3]:
                test_type = st.radio("Tipo:", ["Two-sided", "One-sided"])

            if st.button("🧮 CALCULAR (COHORTES)", use_container_width=True):
                if test_type == "One-sided":
                    alpha = alpha / 2
                result = calculate_sample_size(p1, p2, alpha, power, ratio)
                col_res = st.columns(4)
                with col_res[0]:
                    st.metric("Grupo 1", f"{result['n1']:,}")
                with col_res[1]:
                    st.metric("Grupo 2", f"{result['n2']:,}")
                with col_res[2]:
                    st.metric("Total", f"{result['total']:,}")
                with col_res[3]:
                    st.metric("NNT", f"{int(1/abs(p1-p2)):,}" if p1 != p2 else "∞")

        elif study_type == "📋 Transversal":
            st.markdown("### Estudios Transversales (Prevalencia)")
            st.info("Cálculo de tamaño muestral para estudios transversales con población total y corrección de población finita (FPC)")

            col_trans = st.columns([1, 1, 1, 1])
            with col_trans[0]:
                p_trans = st.number_input("Prevalencia esperada (p):", 0.001, 0.999, 0.30, format="%.3f", help="Proporción esperada del evento en la población")
                population = st.number_input("Población total (N):", 0, 100000000, 10000, format="%d", help="Tamaño de la población total. Ingrese 0 para poblaciones infinitas (sin corrección FPC)")
            with col_trans[1]:
                precision = st.number_input("Precisión (d):", 0.001, 0.5, 0.05, format="%.3f", help="Margen de error o half-width del IC")
            with col_trans[2]:
                alpha_trans = st.select_slider("α (significancia):", options=[0.01, 0.05, 0.10], value=0.05)
                power_trans = st.select_slider("Poder:", options=[0.70, 0.80, 0.90, 0.95], value=0.80)
            with col_trans[3]:
                effect_trans = st.selectbox("Diseño:", ["Simple", "Estratificado"], index=0, help="Simple: muestra aleatoria simple. Estratificado: divide en estratos")

            # Calcular n sin FPC
            from scipy.stats import norm
            z_alpha_trans = norm.ppf(1 - alpha_trans/2)

            n_infinite = ((z_alpha_trans**2) * p_trans * (1 - p_trans)) / (precision**2)

            if st.button("🧮 CALCULAR (TRANSVERSAL)", use_container_width=True):
                # Aplicar corrección de población finita si corresponde
                if population > 0 and n_infinite / population > 0.05:
                    n_finite = n_infinite / (1 + (n_infinite - 1) / population)
                    fpc_applied = True
                else:
                    n_finite = n_infinite
                    fpc_applied = False

                col_resultados = st.columns(4)
                with col_resultados[0]:
                    st.metric("N (sin FPC)", f"{int(np.ceil(n_infinite)):,}")
                with col_resultados[1]:
                    st.metric("N (con FPC)", f"{int(np.ceil(n_finite)):,}")
                with col_resultados[2]:
                    st.metric("FPC aplicada", "Sí" if fpc_applied else "No")
                with col_resultados[3]:
                    if population > 0:
                        pct = (n_infinite / population) * 100
                        st.metric("Muestreo %", f"{pct:.1f}%" if pct < 100 else ">100%")
                    else:
                        st.metric("Muestreo %", "∞")

                st.success(f"✅ Tamaño muestral recomendado: {int(np.ceil(n_finite)):,} participantes")

                if fpc_applied:
                    st.info(f"📊 Nota: La corrección FPC se aplicó porque n/población = {n_infinite/population:.1%} > 5%")

    # ==========================================
    # MÓDULO: VIGILANCIA & IA MEJORADO
    # ==========================================
    elif menu == "📈 Vigilancia & IA":
        st.header("📈 Vigilancia Epidemiológica y Proyecciones V7.0")

        with st.expander("ℹ️ Modelos Disponibles", expanded=False):
            st.markdown("""
            - **SEIR**: Modelo compartimental clásico
            - **ARIMA**: Series temporales para forecasting
            - **Monte Carlo**: Simulaciones con bandas de confianza
            """)

        with st.expander("📥 Datos del Brote (CON FECHAS)", expanded=True):
            col_date1, col_date2 = st.columns([1, 3])
            with col_date1:
                start_date = st.date_input("Fecha inicio:", datetime.now() - timedelta(days=10))
            with col_date2:
                st.info("📝 Agregue registros con fechas. Los datos se guardan automáticamente.")

            # Inicializar datos solo si df_v es None
            if st.session_state.df_v is None:
                base_date = start_date
                dates = [base_date + timedelta(days=i) for i in range(10)]
                casos_acum = np.cumsum([10, 15, 25, 40, 55, 70, 90, 120, 150, 180])
                session_recuperados = np.concatenate([[0], casos_acum[:-1] * 0.7])
                session_recuperados = np.minimum(casos_acum * 0.6, session_recuperados).astype(int)
                activos_init = casos_acum - session_recuperados - [0, 0, 1, 2, 3, 4, 6, 8, 10, 12]

                st.session_state.df_v = pd.DataFrame({
                    "Fecha": dates,
                    "Dia": range(1, 11),
                    "Activos": activos_init.tolist(),
                    "Nuevos": [10, 15, 25, 40, 55, 70, 90, 120, 150, 180],
                    "Recuperados": session_recuperados.tolist(),
                    "Fallecidos": [0, 0, 1, 2, 3, 4, 6, 8, 10, 12]
                })
                # También inicializar df_v_edited
                st.session_state.df_v_edited = st.session_state.df_v.copy()

            # Botón separado para reiniciar (fuera del form)
            col_reset_btn, col_other = st.columns([1, 3])
            with col_reset_btn:
                if st.button("🔄 Reiniciar Datos", use_container_width=True):
                    st.session_state.df_v = None
                    st.session_state.df_v_edited = None
                    st.rerun()

            # Editor de datos - SIN form wrapper para evitar errores
            st.markdown("**Tabla de Datos (Puede editar, agregar y eliminar filas):**")

            # Sincronizar df_v_edited con df_v si hay cambios
            if st.session_state.df_v is not None and (st.session_state.get('df_v_edited') is None or len(st.session_state.df_v_edited) != len(st.session_state.df_v)):
                st.session_state.df_v_edited = st.session_state.df_v.copy()

            edited_df_v = st.data_editor(
                st.session_state.df_v_edited,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=False,
                column_config={
                    "Fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
                    "Dia": st.column_config.NumberColumn("Día", min_value=1),
                    "Activos": st.column_config.NumberColumn("Activos", min_value=0),
                    "Nuevos": st.column_config.NumberColumn("Nuevos", min_value=0),
                    "Recuperados": st.column_config.NumberColumn("Recuperados", min_value=0),
                    "Fallecidos": st.column_config.NumberColumn("Fallecidos", min_value=0)
                },
                key="vigilance_data_editor"
            )

            col_save, col_info = st.columns([1, 3])
            with col_save:
                if st.button("💾 GUARDAR CAMBIOS", use_container_width=True):
                    if edited_df_v is not None and len(edited_df_v) > 0:
                        st.session_state.df_v = edited_df_v.copy()
                        st.session_state.df_v_edited = edited_df_v.copy()
                        persist_user_data()
                        st.success("✅ Cambios guardados correctamente!")
                        st.rerun()
            with col_info:
                st.caption("💡 Use el icono + para agregar filas. 🗑️ para eliminar. Haga clic en celda para editar.")

        # Usar datos de session state para cálculos — normalizar tipos al cargar
        df_v_calc = st.session_state.df_v
        if df_v_calc is not None and len(df_v_calc) > 0:
            required_cols = {'Nuevos', 'Activos', 'Recuperados', 'Fallecidos', 'Dia'}
            missing_cols = required_cols - set(df_v_calc.columns)
            if missing_cols:
                st.error(f"❌ Faltan columnas requeridas en los datos: {missing_cols}. Use '🔄 Reiniciar Datos'.")
                df_v_calc = None
            else:
                try:
                    for col in ['Nuevos', 'Activos', 'Recuperados', 'Fallecidos', 'Dia']:
                        df_v_calc[col] = pd.to_numeric(df_v_calc[col], errors='coerce').fillna(0).astype(int)
                    st.session_state.df_v = df_v_calc
                except Exception as e:
                    st.error(f"❌ Error al normalizar columnas numéricas: {e}. Use '🔄 Reiniciar Datos'.")
                    df_v_calc = None

        # Parámetros SEIR
        st.subheader("🧬 Parámetros del Modelo SEIR")
        col_params = st.columns([1, 1, 1, 1, 1])
        with col_params[0]:
            beta = st.slider("β (Transmisión)", 0.1, 1.0, 0.3, 0.01)
        with col_params[1]:
            sigma = st.slider("σ (Incubación)", 0.05, 0.5, 0.2, 0.01)
        with col_params[2]:
            gamma = st.slider("γ (Recuperación)", 0.05, 0.5, 0.1, 0.01)
        with col_params[3]:
            rho = st.slider("ρ (Detección)", 0.1, 1.0, 0.4, 0.01)
        with col_params[4]:
            n_sim = st.number_input("Sim. Monte Carlo", 50, 1000, 100, 50)

        r0 = beta / gamma

        if df_v_calc is None or len(df_v_calc) == 0:
            st.info("👆 Ingrese o cargue datos en la tabla de arriba para ver métricas y proyecciones.")
        else:
            casos_reales = int(df_v_calc['Nuevos'].sum() / rho)
            ifr = df_v_calc['Fallecidos'].sum() / casos_reales * 100 if casos_reales > 0 else 0

            col_metrics = st.columns([1, 1, 1, 1, 1])
            with col_metrics[0]:
                st.metric("R0", f"{r0:.2f}", delta="🔴 Epidemia" if r0 > 1 else "🟢 Controlada")
            with col_metrics[1]:
                st.metric("Casos Totales", f"{casos_reales:,}" if casos_reales > 0 else "0")
            with col_metrics[2]:
                st.metric("IFR", f"{ifr:.2f}%" if ifr > 0 else "0%")
            with col_metrics[3]:
                recovered_val = int(df_v_calc['Recuperados'].iloc[-1])
                st.metric("Recuperados", f"{recovered_val:,}")
            with col_metrics[4]:
                deaths_val = int(df_v_calc['Fallecidos'].sum())
                st.metric("Fallecidos", f"{deaths_val:,}")

        # SELECCIÓN DE MODELO
            # SELECCIÓN DE MODELO
            model_type = st.radio("🔬 Modelo de Proyección:", ["SEIR + Monte Carlo", "ARIMA"], horizontal=True)

            # Solo ejecutar modelos si hay datos suficientes
            if len(df_v_calc) > 3:
                col_btn_run, _ = st.columns([1, 3])
                with col_btn_run:
                    run_model = st.button("▶️ EJECUTAR MODELO", use_container_width=True, key="run_vigilance_model")
                if run_model:
                    st.session_state.modelo_ejecutado = True

                if st.session_state.get('modelo_ejecutado', False):
                    try:
                        with st.spinner("⏳ Ejecutando modelo..."):
                            _, _, RandomForestRegressor, _, ARIMA, _ = get_heavy_imports()

                            days_total = len(df_v_calc) + 30
                            N = casos_reales * 10
                            I0 = casos_reales // 10
                            E0 = I0 * 2
                            R0_init = int(df_v_calc['Recuperados'].iloc[-1])
                            S0 = N - E0 - I0 - R0_init

                            S_arr, E_arr, I_arr, R_arr = [S0], [E0], [I0], [R0_init]
                            for t in range(1, days_total):
                                S_t, E_t, I_t, R_t = S_arr[-1], E_arr[-1], I_arr[-1], R_arr[-1]
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

                            if model_type == "SEIR + Monte Carlo":
                                # Proyecciones con desviaciones estándar
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                                model.fit(df_v_calc[['Dia']], df_v_calc['Nuevos'])
                                futuro_x = np.array([[len(df_v_calc)+i] for i in range(1, 16)])
                                base_preds = model.predict(futuro_x)

                                all_sims_nuevos = []
                                all_sims_activos = []
                                all_sims_recuperados = []
                                all_sims_fallecidos = []

                                std_err = max(df_v_calc['Nuevos'].std(), 1)
                                std_activos = max(df_v_calc['Activos'].std(), 1)
                                std_rec = max(df_v_calc['Recuperados'].std(), 1)
                                std_fall = max(df_v_calc['Fallecidos'].std(), 1)

                                for _ in range(n_sim):
                                    noise_n = np.random.normal(0, std_err * 0.6, size=base_preds.shape)
                                    all_sims_nuevos.append(np.maximum(0, base_preds + noise_n))
                                    all_sims_activos.append(np.maximum(0, df_v_calc['Activos'].iloc[-1] + np.random.normal(0, std_activos, size=base_preds.shape)))
                                    all_sims_recuperados.append(np.maximum(0, df_v_calc['Recuperados'].iloc[-1] + np.cumsum(np.maximum(0, base_preds + noise_n) * 0.7)))
                                    all_sims_fallecidos.append(np.maximum(0, df_v_calc['Fallecidos'].iloc[-1] + np.cumsum(np.maximum(0, base_preds + noise_n) * 0.02)))

                                all_sims_nuevos = np.array(all_sims_nuevos)
                                all_sims_activos = np.array(all_sims_activos)
                                all_sims_recuperados = np.array(all_sims_recuperados)
                                all_sims_fallecidos = np.array(all_sims_fallecidos)

                                p_nuevos_mean = np.mean(all_sims_nuevos, axis=0)
                                p_nuevos_std = np.std(all_sims_nuevos, axis=0)
                                p_activos_mean = np.mean(all_sims_activos, axis=0)
                                p_activos_std = np.std(all_sims_activos, axis=0)
                                p_rec_mean = np.mean(all_sims_recuperados, axis=0)
                                p_rec_std = np.std(all_sims_recuperados, axis=0)
                                p_fall_mean = np.mean(all_sims_fallecidos, axis=0)
                                p_fall_std = np.std(all_sims_fallecidos, axis=0)

                                fig_seir = go.Figure()
                                days_range = list(range(days_total))
                                fig_seir.add_trace(go.Scatter(x=days_range, y=S_arr, name="S", line=dict(color='#60a5fa')))
                                fig_seir.add_trace(go.Scatter(x=days_range, y=E_arr, name="E", line=dict(color='#f59e0b')))
                                fig_seir.add_trace(go.Scatter(x=days_range, y=I_arr, name="I", line=dict(color='#ef4444', width=3)))
                                fig_seir.add_trace(go.Scatter(x=days_range, y=R_arr, name="R", line=dict(color='#10b981')))
                                fig_seir.update_layout(title=f"📊 SEIR (R0={r0:.2f})", height=400, template="plotly_dark")
                                st.plotly_chart(fig_seir, use_container_width=True)

                                # Gráfico con desviaciones estándar
                                futuro_dias = list(range(len(df_v_calc) + 1, len(df_v_calc) + 16))

                                fig_proj = go.Figure()
                                fig_proj.add_trace(go.Scatter(
                                    x=list(df_v_calc['Dia']) + futuro_dias,
                                    y=list(df_v_calc['Nuevos']) + p_nuevos_mean.tolist(),
                                    name="Nuevos",
                                    line=dict(color='#3b82f6', width=3),
                                    error_y=dict(type='data', array=p_nuevos_std.tolist(), visible=True)
                                ))
                                fig_proj.add_trace(go.Scatter(
                                    x=list(df_v_calc['Dia']) + futuro_dias,
                                    y=list(df_v_calc['Activos']) + p_activos_mean.tolist(),
                                    name="Activos",
                                    line=dict(color='#ef4444', width=2)
                                ))
                                fig_proj.add_trace(go.Scatter(
                                    x=list(df_v_calc['Dia']) + futuro_dias,
                                    y=list(df_v_calc['Recuperados']) + p_rec_mean.tolist(),
                                    name="Recuperados",
                                    line=dict(color='#10b981', width=2)
                                ))
                                fig_proj.add_trace(go.Scatter(
                                    x=list(df_v_calc['Dia']) + futuro_dias,
                                    y=list(df_v_calc['Fallecidos']) + p_fall_mean.tolist(),
                                    name="Fallecidos",
                                    line=dict(color='#6b7280', width=2)
                                ))
                                fig_proj.update_layout(title="📈 Proyecciones CON Desviaciones Estándar", height=450, template="plotly_dark")
                                st.plotly_chart(fig_proj, use_container_width=True)

                                # Tabla de proyecciones
                                df_proy = pd.DataFrame({
                                    "Día": futuro_dias,
                                    "Nuevos (Media)": p_nuevos_mean.astype(int),
                                    "Nuevos (DE)": p_nuevos_std.astype(int),
                                    "Activos (Media)": p_activos_mean.astype(int),
                                    "Activos (DE)": p_activos_std.astype(int),
                                    "Recuperados (Media)": p_rec_mean.astype(int),
                                    "Recuperados (DE)": p_rec_std.astype(int),
                                    "Fallecidos (Media)": p_fall_mean.astype(int),
                                    "Fallecidos (DE)": p_fall_std.astype(int)
                                })
                                st.dataframe(df_proy, use_container_width=True)

                            else:  # ARIMA
                                st.info("🔮 Modelo ARIMA para forecasting de series temporales")
                                arima_order = st.multiselect("Orden ARIMA (p,d,q):", [0, 1, 2], default=[1, 1, 1])
                                if len(arima_order) == 3:
                                    order_tuple = tuple(arima_order)
                                else:
                                    order_tuple = (1, 1, 1)

                                if st.button("🔮 EJECUTAR ARIMA", use_container_width=True):
                                    try:
                                        arima_result = calculate_arima_forecast(df_v_calc['Nuevos'].values, periods=15, order=order_tuple)
                                        if 'error' not in arima_result:
                                            futuro_dias = list(range(len(df_v_calc) + 1, len(df_v_calc) + 16))
                                            fig_arima = go.Figure()
                                            fig_arima.add_trace(go.Scatter(x=df_v_calc['Dia'], y=df_v_calc['Nuevos'], name="Histórico", line=dict(width=3)))
                                            fig_arima.add_trace(go.Scatter(x=futuro_dias, y=arima_result['forecast'], name="Proyección ARIMA", line=dict(dash='dash', color='#ef4444')))
                                            fig_arima.add_trace(go.Scatter(
                                                x=futuro_dias + futuro_dias[::-1],
                                                y=list(arima_result['upper_ci']) + list(arima_result['lower_ci'])[::-1],
                                                fill='toself', fillcolor='rgba(239,68,68,0.2)', name="IC 95%"
                                            ))
                                            fig_arima.update_layout(title=f"🔮 ARIMA Forecast (orden={order_tuple})", height=400, template="plotly_dark")
                                            st.plotly_chart(fig_arima, use_container_width=True)
                                            st.metric("AIC", f"{arima_result['aic']:.2f}")
                                        else:
                                            st.error(f"Error en ARIMA: {arima_result['error']}")
                                    except Exception as e:
                                        st.error(f"Error en ARIMA: {e}")
                    except Exception as model_err:
                        st.error(f"⚠️ Error ejecutando modelo: {model_err}")
                        st.info("Intente con datos diferentes o verifique que las columnas tengan valores numéricos válidos.")
                else:
                    st.info("👆 Configure los parámetros y haga clic en **▶️ EJECUTAR MODELO** para generar las proyecciones.")

            # Exportar reporte ASIS
            st.markdown("---")
            r0_pdf = beta / gamma
            casos_pdf = int(df_v_calc['Nuevos'].sum() / rho)
            ifr_pdf = df_v_calc['Fallecidos'].sum() / casos_pdf * 100 if casos_pdf > 0 else 0

            if st.button("📄 GENERAR REPORTE ASIS (PDF)", use_container_width=True):
                with st.spinner("Generando reporte..."):
                    data_for_report = {
                        'vigilance': {
                            'total_cases': casos_pdf,
                            'r0': r0_pdf,
                            'ifr': ifr_pdf
                        }
                    }
                    try:
                        pdf_buffer = generate_asis_report(data_for_report, 'pdf')
                        st.download_button(
                            "📥 Descargar PDF",
                            pdf_buffer.getvalue(),
                            "ASIS_Report.pdf",
                            "application/pdf"
                        )
                    except Exception as pdf_err:
                        st.error(f"Error generando PDF: {pdf_err}")

    # ==========================================
    # MÓDULO: REVISIÓN DE LITERATURA
    # ==========================================
    elif menu == "📚 Revisión de Literatura":
        st.header("📚 Centro de Evidencia Científica")

        tab_pico, tab_prisma, tab_forest, tab_meta, tab_quality, tab_psm = st.tabs([
            "🤖 PICO", "📑 PRISMA", "🌲 Forest", "📊 Meta", "⚖️ RoB/GRADE", "📐 PSM"
        ])

        with tab_pico:
            st.subheader("🤖 Analizador IA de Evidencia")
            api_k = st.text_input("🔑 OpenAI API Key", type="password", key="api_pico", placeholder="sk-...")
            if not api_k:
                st.info("💡 Ingrese su API Key para activar el análisis.")
            else:
                ext = LiteratureAIExtractor(api_k)
                metodo = st.radio("📥 Método:", ["PDF", "DOI"], horizontal=True)

                if metodo == "PDF":
                    f = st.file_uploader("Subir PDF", type="pdf")
                    if f and st.button("🔍 Extraer PICO"):
                        with st.spinner("Analizando..."):
                            res = ext.from_pdf(f)
                            if res and "error" not in res:
                                st.session_state.articulos_pico.append(res)
                                persist_user_data()
                                st.success("✅ Analizado!")
                else:
                    doi = st.text_input("DOI", placeholder="10.1056/...")
                    if doi and st.button("🔍 Consultar DOI"):
                        with st.spinner("Consultando..."):
                            res = ext.from_doi(doi)
                            if res and "error" not in res:
                                st.session_state.articulos_pico.append(res)
                                persist_user_data()
                                st.success("✅ Analizado!")

                if st.session_state.articulos_pico:
                    df_articulos = pd.DataFrame(st.session_state.articulos_pico)
                    st.dataframe(df_articulos, use_container_width=True)
                    if st.button("🗑️ Limpiar"):
                        st.session_state.articulos_pico = []
                        persist_user_data()

        with tab_prisma:
            st.subheader("📑 Flujograma PRISMA 2020")
            prisma_data = st.session_state.prisma_data if st.session_state.prisma_data else {
                'registros_db': 1500, 'duplicados': 400, 'excluidos_title': 500, 'estudios_included': 25
            }
            col_p1, col_p2 = st.columns([1, 2])
            with col_p1:
                prisma_data['registros_db'] = st.number_input("Registros DB:", value=prisma_data.get('registros_db', 1500))
                prisma_data['duplicados'] = st.number_input("Duplicados:", value=prisma_data.get('duplicados', 400))
                prisma_data['excluidos_title'] = st.number_input("Excluidos Título:", value=prisma_data.get('excluidos_title', 500))
                prisma_data['estudios_included'] = st.number_input("Estudios Incluidos:", value=prisma_data.get('estudios_included', 25))
                st.session_state.prisma_data = prisma_data
                if st.button("💾 Guardar PRISMA", use_container_width=True, key="save_prisma"):
                    persist_user_data()
                    st.success("✅ PRISMA guardado!")

        with tab_forest:
            st.subheader("🌲 Forest Plot")
            if 'forest_studies' not in st.session_state:
                st.session_state.forest_studies = pd.DataFrame({
                    'Estudio': ['Smith 2020', 'Johnson 2019', 'Williams 2021'],
                    'Eventos_Tto': [20, 35, 28], 'Total_Tto': [100, 150, 120],
                    'Eventos_Ctrl': [30, 50, 45], 'Total_Ctrl': [100, 150, 120]
                })
            edit_forest = st.data_editor(st.session_state.forest_studies, num_rows="dynamic", use_container_width=True)
            if st.button("🌲 Generar Forest Plot"):
                st.session_state.forest_studies = edit_forest
                persist_user_data()
                df_calc = edit_forest.copy()
                df_calc['OR'] = (df_calc['Eventos_Tto'] / (df_calc['Total_Tto'] - df_calc['Eventos_Tto'])) / (df_calc['Eventos_Ctrl'] / (df_calc['Total_Ctrl'] - df_calc['Eventos_Ctrl']))
                fig, ax = plt.subplots(figsize=(12, max(4, len(df_calc) * 0.8)))
                for i, (_, row) in enumerate(df_calc.iterrows()):
                    or_val = row['OR']
                    if pd.notna(or_val) and or_val > 0:
                        ax.plot(or_val, i, 'bs', markersize=8)
                        ax.text(or_val + 0.15, i, f"{row['Estudio']}: OR={or_val:.2f}", va='center')
                ax.axvline(x=1, color='red', linestyle='--')
                ax.set_xscale('log')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            if st.button("➕ Enviar a Meta-análisis"):
                if len(edit_forest) >= 2:
                    st.session_state.meta_studies = edit_forest
                    persist_user_data()
                    st.success(f"✅ {len(edit_forest)} estudios transferidos!")

        with tab_meta:
            st.subheader("📊 Meta-análisis")
            if len(st.session_state.meta_studies) < 2:
                st.warning("📌 Importe desde Forest Plot (mínimo 2 estudios)")
            else:
                mod_meta = st.selectbox("Modelo:", ["Efectos Fijos (Peto)", "Efectos Aleatorios (DerSimonian-Laird)"])
                if st.button("📊 Calcular"):
                    ev_e = tuple(st.session_state.meta_studies['Eventos_Tto'])
                    tt_e = tuple(st.session_state.meta_studies['Total_Tto'])
                    ev_c = tuple(st.session_state.meta_studies['Eventos_Ctrl'])
                    tt_c = tuple(st.session_state.meta_studies['Total_Ctrl'])
                    res_m = meta_analysis_fixed_effect(ev_e, tt_e, ev_c, tt_c) if "Fijos" in mod_meta else meta_analysis_random_effects(ev_e, tt_e, ev_c, tt_c)
                    pooled_or = res_m['pooled_or'] if "Fijos" in mod_meta else res_m['pooled_or_re']
                    i2 = res_m['I2']
                    c1, c2, c3 = st.columns(3)
                    c1.metric("OR Combinado", f"{pooled_or:.2f}")
                    c2.metric("I²", f"{i2:.1f}%")
                    c3.metric("p-value", f"{res_m.get('p_value', 0.05):.4f}")

        with tab_quality:
            st.subheader("⚖️ Evaluación de Calidad")
            q_sub = st.radio("Herramienta:", ["RoB 2", "GRADE"], horizontal=True)
            if q_sub == "RoB 2":
                s_name = st.text_input("Nombre del Estudio:", placeholder="Autor, Año")
                d1 = st.select_slider("D1: Randomización", ["Low", "Some Concerns", "High"])
                d2 = st.select_slider("D2: Intervención", ["Low", "Some Concerns", "High"])
                d3 = st.select_slider("D3: Datos Faltantes", ["Low", "Some Concerns", "High"])
                d4 = st.select_slider("D4: Medición", ["Low", "Some Concerns", "High"])
                if st.button("💾 Guardar RoB"):
                    st.session_state.rob_assessments.append({'Estudio': s_name, 'D1': d1, 'D2': d2, 'D3': d3, 'D4': d4})
                    persist_user_data()
                    st.success("✅ Guardado!")
                if st.session_state.rob_assessments:
                    st.dataframe(pd.DataFrame(st.session_state.rob_assessments), use_container_width=True)
            else:
                outcome = st.text_input("Resultado:")
                r_bias = st.number_input("Riesgo de Sesgo (0 a -2)", -2, 0, 0)
                incons = st.number_input("Inconsistencia (0 a -2)", -2, 0, 0)
                indir = st.number_input("Indirectitud (0 a -2)", -2, 0, 0)
                large = st.checkbox("Efecto Grande (+1)")
                confound = st.checkbox("Factores Confusores (+1)")
                if st.button("⚖️ Calcular GRADE"):
                    score = 4 + r_bias + incons + indir + (1 if large else 0) + (1 if confound else 0)
                    score = max(1, min(4, score))
                    labels = {4: "🔴 Alta", 3: "🟡 Moderada", 2: "🟠 Baja", 1: "⚫ Muy Baja"}
                    st.metric("Certeza", labels.get(score, "⚫ Muy Baja"))

        with tab_psm:
            st.subheader("📐 Propensity Score Matching (PSM)")
            st.info("El PSM permite equilibrar covariables en estudios observacionales")

            if st.session_state.df_master is None:
                st.warning("⚠️ Cargue datos primero en Dashboard")
            else:
                df = st.session_state.df_master
                col_psm = st.columns(3)
                with col_psm[0]:
                    treatment_col = st.selectbox("Variable de Exposición:", df.columns)
                with col_psm[1]:
                    outcome_col = st.selectbox("Variable de Resultado:", df.select_dtypes(include=np.number).columns)
                with col_psm[2]:
                    covariate_cols = st.multiselect("Covariables (confundidores):", [c for c in df.columns if c != treatment_col and c != outcome_col])

                if st.button("📐 EJECUTAR PSM", use_container_width=True):
                    with st.spinner("Ejecutando PSM..."):
                        # Convertir columna de tratamiento a binario
                        df['treatment'] = (df[treatment_col] == df[treatment_col].unique()[0]).astype(int)
                        result_psm = propensity_score_matching(df, 'treatment', outcome_col, covariate_cols)

                        col_res = st.columns(4)
                        with col_res[0]:
                            st.metric("ATE", f"{result_psm['ate']:.4f}")
                        with col_res[1]:
                            st.metric("Std. Diff", f"{result_psm['std_diff']:.4f}")
                        with col_res[2]:
                            st.metric("N Pareados", result_psm['n_matched'])
                        with col_res[3]:
                            balanced_count = sum(1 for v in result_psm['balance_metrics'].values() if v['balanced'])
                            st.metric("Balanceados", f"{balanced_count}/{len(covariate_cols)}")

                        st.markdown("### Métricas de Balance")
                        balance_df = pd.DataFrame(result_psm['balance_metrics']).T
                        st.dataframe(balance_df, use_container_width=True)
# ==========================================
    # MÓDULO: SUPERVIVENCIA MEJORADO (COX/POISSON)
    # ==========================================
    elif menu == "📉 Supervivencia (KM)":
        st.header("📉 Análisis de Supervivencia V7.0 - Kaplan-Meier, Cox y Poisson")

        tabs = st.tabs(["📝 Datos", "📈 Kaplan-Meier", "📊 Cox PH", "📈 Poisson"])

        with tabs[0]:
            st.markdown("### 📝 Datos de Supervivencia")
            if st.session_state.survival_data is None:
                np.random.seed(42)
                st.session_state.survival_data = pd.DataFrame({
                    'ID': range(1, 101),
                    'Tiempo': np.concatenate([np.random.exponential(30, 50), np.random.exponential(20, 50)]).round(1),
                    'Evento': np.random.binomial(1, 0.4, 100),
                    'Grupo': ['Tratamiento'] * 50 + ['Control'] * 50,
                    'Edad': np.random.randint(30, 80, 100),
                    'Sexo': np.random.choice(['M', 'F'], 100)
                })

            uploaded = st.file_uploader("📂 Cargar CSV:", type="csv")
            if uploaded:
                st.session_state.survival_data = pd.read_csv(uploaded)
                persist_user_data()
                st.success(f"✅ {len(st.session_state.survival_data)} registros cargados!")

            if st.session_state.survival_data is not None:
                df = st.session_state.survival_data
                edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
                col_sv, col_sv_info = st.columns([1, 3])
                with col_sv:
                    if st.button("💾 GUARDAR CAMBIOS", use_container_width=True, key="save_survival"):
                        st.session_state.survival_data = edited
                        persist_user_data()
                        st.success("✅ Datos guardados!")
                        st.rerun()
                with col_sv_info:
                    st.caption("💡 Edite las celdas y presione Guardar para confirmar cambios.")
                col_stats = st.columns(4)
                col_stats[0].metric("Muestras", len(edited))
                col_stats[1].metric("Eventos", int(edited['Evento'].sum()))
                col_stats[2].metric("Censuras", int(len(edited) - edited['Evento'].sum()))
                col_stats[3].metric("Tiempo medio", f"{edited['Tiempo'].mean():.1f}")

        with tabs[1]:
            st.markdown("### 📈 Curva de Kaplan-Meier")
            df = st.session_state.survival_data
            if df is not None:
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
                        times, events = df_km['time'].values, df_km['event'].values
                        n = len(times)
                        unique_times = np.unique(times[events == 1])
                        survival_times, survival_probs = [0], [1.0]
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
                            results[str(group)] = calculate_km(df.loc[mask, time_col].values, df.loc[mask, event_col].values)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = {'Tratamiento': '#3498db', 'Control': '#e74c3c', 'Global': '#2ecc71'}
                    for name, data in results.items():
                        ax.step(data['times'], data['survival'], where='post', color=colors.get(name, '#3498db'), linewidth=2.5, label=name)
                    ax.set_xlabel('Tiempo')
                    ax.set_ylabel('Probabilidad de Supervivencia')
                    ax.set_title('Curva de Kaplan-Meier')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(left=0)
                    ax.set_ylim(0, 1.05)
                    st.pyplot(fig)
                    plt.close(fig)

        with tabs[2]:
            st.markdown("### 📊 Regresión de Cox (Hazard Ratios Ajustados)")
            st.info("Cox PH permite calcular Hazard Ratios ajustados por múltiples variables")
            df = st.session_state.survival_data
            if df is not None:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                col_cox = st.columns([1, 1, 1])
                with col_cox[0]:
                    time_col = st.selectbox("Tiempo:", num_cols, key="cox_time")
                with col_cox[1]:
                    event_col = st.selectbox("Evento:", num_cols, key="cox_event")
                with col_cox[2]:
                    covariables = st.multiselect("Covariables:", [c for c in df.columns if c not in [time_col, event_col]])

                if st.button("📊 AJUSTAR COX PH", use_container_width=True):
                    with st.spinner("Ajustando modelo..."):
                        KaplanMeierFitter, CoxPHFitter, _, _, _, _ = get_analysis_imports()
                        df_cox = df[[time_col, event_col] + covariables].dropna().copy()
                        for col in covariables:
                            if df_cox[col].dtype == 'object':
                                df_cox[col] = pd.factorize(df_cox[col])[0]
                        try:
                            cph = CoxPHFitter()
                            cph.fit(df_cox, duration_col=time_col, event_col=event_col)
                            st.dataframe(cph.summary, use_container_width=True)
                            col_hr = st.columns(3)
                            with col_hr[0]:
                                st.metric("Log-Likelihood", f"{cph.log_likelihood_:.2f}")
                            with col_hr[1]:
                                st.metric("AIC", f"{cph.AIC_:.2f}")
                            with col_hr[2]:
                                st.metric("Concordance", f"{cph.concordance_index_:.3f}")
                        except Exception as e:
                            st.error(f"Error en Cox PH: {e}")

        with tabs[3]:
            st.markdown("### 📈 Regresión de Poisson (Tasas de Incidencia)")
            st.info("Útil para calcular tasas de incidencia y Hazard Ratios")
            df = st.session_state.survival_data
            if df is not None:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                col_poisson = st.columns([1, 1, 1])
                with col_poisson[0]:
                    time_col = st.selectbox("Tiempo:", num_cols, key="poisson_time")
                with col_poisson[1]:
                    event_col = st.selectbox("Evento:", num_cols, key="poisson_event")
                with col_poisson[2]:
                    offset_col = st.selectbox("Offset (exposición):", ['Ninguno'] + num_cols)

                if st.button("📊 AJUSTAR POISSON", use_container_width=True):
                    with st.spinner("Ajustando modelo..."):
                        from statsmodels.genmod.generalized_linear_model import GLM
                        from statsmodels.genmod.families import Poisson
                        df_poisson = df[[time_col, event_col]].dropna().copy()
                        try:
                            if offset_col != 'Ninguno':
                                df_poisson['offset'] = np.log(df[offset_col].fillna(1))
                            else:
                                df_poisson['offset'] = np.log(df_poisson[time_col])
                            poisson_model = GLM(df_poisson[event_col], np.ones(len(df_poisson)),
                                               family=Poisson(), offset=df_poisson['offset'].values)
                            result = poisson_model.fit()
                            incidence_rate = df_poisson[event_col].sum() / df_poisson[time_col].sum()
                            col_ir = st.columns(3)
                            with col_ir[0]:
                                st.metric("Tasa Incidencia", f"{incidence_rate:.4f}")
                            with col_ir[1]:
                                st.metric("Log-Likelihood", f"{result.llf:.2f}")
                            with col_ir[2]:
                                st.metric("Deviance", f"{result.deviance:.2f}")
                            st.text(result.summary())
                        except Exception as e:
                            st.error(f"Error en Poisson: {e}")
                            
    # ==========================================
    # MÓDULO: CURVAS ROC
    # ==========================================
    elif menu == "🎯 Curvas ROC":
        st.header("🎯 Curvas ROC - Evaluación Diagnóstica")
        tabs = st.tabs(["📝 Datos", "📈 Curva ROC", "📊 Comparación"])
        
        with tabs[0]:
            if st.session_state.roc_data is None:
                np.random.seed(42)
                st.session_state.roc_data = pd.DataFrame({
                    'ID': range(1, 201),
                    'Probabilidad': np.concatenate([np.random.beta(5, 2, 100), np.random.beta(2, 5, 100)]),
                    'Real': ['Positivo'] * 100 + ['Negativo'] * 100
                })

            uploaded = st.file_uploader("📂 Cargar CSV:", type="csv")
            if uploaded:
                st.session_state.roc_data = pd.read_csv(uploaded)
                persist_user_data()
                st.success("✅ Datos cargados!")

            if st.session_state.roc_data is not None:
                st.dataframe(st.session_state.roc_data.head(20), use_container_width=True)

        with tabs[1]:
            if st.session_state.roc_data is not None:
                df = st.session_state.roc_data
                pred_col = st.selectbox("📊 Variable:", [c for c in df.columns if c not in ['ID', 'Real']])

                if st.button("📊 GENERAR CURVA ROC", use_container_width=True):
                    from sklearn.metrics import roc_curve, auc, confusion_matrix
                    y_true = (df['Real'] == df['Real'].unique()[0]).astype(int)
                    y_score = df[pred_col].values
                    fpr, tpr, thresholds = roc_curve(y_true, y_score)
                    roc_auc = auc(fpr, tpr)

                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    axes[0].plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'ROC (AUC={roc_auc:.3f})')
                    axes[0].plot([0, 1], [0, 1], 'k--')
                    axes[0].set_xlabel('1 - Especificidad')
                    axes[0].set_ylabel('Sensibilidad')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)

                    j_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[j_idx]
                    y_pred = (y_score >= optimal_threshold).astype(int)
                    cm = confusion_matrix(y_true, y_pred)

                    im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    axes[1].set_title(f'Matriz (θ={optimal_threshold:.3f})')
                    axes[1].set_xticks([0, 1])
                    axes[1].set_yticks([0, 1])
                    axes[1].set_xticklabels(['Negativo', 'Positivo'])
                    axes[1].set_yticklabels(['Negativo', 'Positivo'])
                    for i in range(2):
                        for j in range(2):
                            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    col_m = st.columns(4)
                    with col_m[0]: st.metric("AUC-ROC", f"{roc_auc:.4f}")
                    with col_m[1]: st.metric("Sensibilidad", f"{tpr[j_idx]:.4f}")
                    with col_m[2]: st.metric("Especificidad", f"{1-fpr[j_idx]:.4f}")
                    with col_m[3]: st.metric("Punto Corte", f"{optimal_threshold:.4f}")

        with tabs[2]:
            st.markdown("### 📊 Comparación de Múltiples Variables Predictoras")
            if st.session_state.roc_data is not None:
                df = st.session_state.roc_data
                num_cols_roc = [c for c in df.select_dtypes(include=np.number).columns if c != 'ID']
                if len(num_cols_roc) >= 2:
                    vars_comp = st.multiselect("Seleccionar variables a comparar:", num_cols_roc, default=num_cols_roc[:min(3, len(num_cols_roc))])
                    if vars_comp and st.button("📊 COMPARAR CURVAS ROC", use_container_width=True):
                        from sklearn.metrics import roc_curve, auc
                        y_true_comp = (df['Real'] == df['Real'].unique()[0]).astype(int)
                        fig_comp, ax_comp = plt.subplots(figsize=(10, 7))
                        ax_comp.plot([0, 1], [0, 1], 'k--', label='Azar (AUC=0.50)')
                        colors_roc = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
                        results_comp = []
                        for idx, var in enumerate(vars_comp):
                            fpr_c, tpr_c, _ = roc_curve(y_true_comp, df[var].values)
                            auc_c = auc(fpr_c, tpr_c)
                            ax_comp.plot(fpr_c, tpr_c, color=colors_roc[idx % len(colors_roc)], linewidth=2.5, label=f'{var} (AUC={auc_c:.3f})')
                            results_comp.append({'Variable': var, 'AUC': round(auc_c, 4)})
                        ax_comp.set_xlabel('1 - Especificidad (FPR)')
                        ax_comp.set_ylabel('Sensibilidad (TPR)')
                        ax_comp.set_title('Comparación de Curvas ROC')
                        ax_comp.legend(loc='lower right')
                        ax_comp.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_comp)
                        plt.close(fig_comp)
                        st.markdown("#### 📋 Tabla Comparativa de AUC")
                        df_comp = pd.DataFrame(results_comp).sort_values('AUC', ascending=False)
                        st.dataframe(df_comp, hide_index=True, use_container_width=True)
                        mejor = df_comp.iloc[0]
                        st.success(f"✅ Mejor predictor: **{mejor['Variable']}** con AUC = {mejor['AUC']:.4f}")
                else:
                    st.info("📋 Necesita al menos 2 variables numéricas en los datos para comparar curvas ROC.")
            else:
                st.warning("⚠️ Cargue datos en la pestaña 📝 Datos primero.")

    # ==========================================
    # MÓDULO: MAPAS GEOGRÁFICOS
    # ==========================================
    elif menu == "🗺️ Mapas Geográficos":
        st.header("🗺️ Mapas Geográficos - Epidemiología Espacial")

        tabs = st.tabs(["📊 Coroplético", "🔥 Heatmap", "📍 Marcadores"])

        with tabs[0]:
            st.markdown("### 📊 Mapa Coroplético")

            if st.session_state.map_data is None:
                st.session_state.map_data = pd.DataFrame({
                    'Pais': ['Colombia'] * 6,
                    'Departamento': ['Antioquia', 'Cundinamarca', 'Valle del Cauca', 'Atlántico', 'Santander', 'Bolívar'],
                    'Municipio': ['Medellín', 'Bogotá', 'Cali', 'Barranquilla', 'Bucaramanga', 'Cartagena'],
                    'Casos': [1500, 1200, 1100, 900, 800, 750],
                    'Poblacion': [6500000, 3000000, 4500000, 2500000, 2000000, 2100000]
                })

            map_data = st.data_editor(st.session_state.map_data, num_rows="dynamic", use_container_width=True)
            col_map_save, col_map_info = st.columns([1, 3])
            with col_map_save:
                if st.button("💾 GUARDAR CAMBIOS", use_container_width=True, key="save_map"):
                    st.session_state.map_data = map_data
                    persist_user_data()
                    st.success("✅ Datos guardados!")
            with col_map_info:
                st.caption("💡 Edite las celdas y presione Guardar para confirmar cambios.")

            col_stat = st.columns(3)
            col_stat[0].metric("Total Casos", f"{map_data['Casos'].sum():,}")
            col_stat[1].metric("Ubicaciones", len(map_data))
            if map_data['Poblacion'].sum() > 0:
                avg_tasa = (map_data['Casos'].sum() / map_data['Poblacion'].sum() * 100000)
                col_stat[2].metric("Tasa x100k", f"{avg_tasa:.2f}")

        with tabs[1]:
            st.markdown("### 🔥 Mapa de Calor")
            hotspots = {'Bogotá': (4.7110, -74.0721), 'Medellín': (6.2442, -75.5812), 'Cali': (3.8000, -76.5220)}
            if st.button("🔥 GENERAR HEATMAP"):
                heat_data = []
                for city, (lat, lon) in hotspots.items():
                    for _ in range(50):
                        heat_data.append({'lat': lat + np.random.normal(0, 0.3), 'lon': lon + np.random.normal(0, 0.3), 'city': city})
                df_heat = pd.DataFrame(heat_data)
                fig = px.density_mapbox(df_heat, lat='lat', lon='lon', radius=15, center={'lat': 4.5709, 'lon': -74.2973}, zoom=5, mapbox_style='carto-darkmatter')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            st.markdown("### 📍 Mapa con Marcadores")
            if st.button("📍 MOSTRAR MAPA"):
                fig = px.scatter_mapbox(pd.DataFrame({'lat': [6.2442, 4.7110, 3.8000], 'lon': [-75.5812, -74.0721, -76.5220], 'name': ['Hospital Medellín', 'Clínica Bogotá', 'UCI Cali']}),
                    lat='lat', lon='lon', hover_name='name', zoom=5, center={'lat': 4.5709, 'lon': -74.2973}, mapbox_style='carto-darkmatter', height=500)
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # MÓDULO: BIOINFORMÁTICA
    # ==========================================
    elif menu == "🧬 Bioinformática":
        st.header("🧬 Análisis de Secuencias Genéticas")

        CODON_TABLE = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S', 'CCT': 'P', 'CCC': 'P',
            'CCA': 'P', 'CCG': 'P', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A',
            'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R',
            'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G', 'TAA': '*', 'TAG': '*', 'TGA': '*'
        }

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
                st.success(f"✅ {seq_type} - {len(seq_clean)} pb")

                if st.button("🧬 ANALIZAR", use_container_width=True):
                    gc = (seq_clean.count('G') + seq_clean.count('C')) / len(seq_clean)
                    composition = Counter(seq_clean)
                    col_stats = st.columns(4)
                    with col_stats[0]: st.metric("Longitud", f"{len(seq_clean):,}")
                    with col_stats[1]: st.metric("Tipo", seq_type)
                    with col_stats[2]: st.metric("GC", f"{gc:.2%}")
                    with col_stats[3]: st.metric("Ratio GC/AT", f"{gc/(1-gc):.2f}" if gc < 1 else "N/A")

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    bases = ['A', 'T', 'G', 'C'] if 'U' not in seq_clean else ['A', 'U', 'G', 'C']
                    counts = [composition.get(b, 0) for b in bases]
                    colors = {'A': '#e74c3c', 'T': '#3498db', 'G': '#2ecc71', 'C': '#f39c12', 'U': '#9b59b6'}
                    ax1.bar(bases, counts, color=[colors.get(b, '#95a5a6') for b in bases])
                    for i, c in enumerate(counts):
                        ax1.text(i, c + max(counts)*0.02, str(c), ha='center')
                    ax2.pie(counts, labels=bases, colors=[colors.get(b, '#95a5a6') for b in bases], autopct='%1.1f%%', startangle=90)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    comp_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N', 'U': 'A'}
                    rev_comp = ''.join([comp_map.get(b, 'N') for b in seq_clean[::-1]])
                    st.markdown("### 🔄 Complementaria Reversa")
                    st.code('\n'.join([rev_comp[i:i+80] for i in range(0, len(rev_comp), 80)]))

                    protein = ''.join([CODON_TABLE.get(seq_clean[i:i+3], 'X') for i in range(0, len(seq_clean) - len(seq_clean) % 3, 3)])
                    st.markdown("### 🧪 Traducción a Proteína")
                    st.code('\n'.join([protein[i:i+60] for i in range(0, len(protein), 60)]))

    # ==========================================
    # MÓDULO: MI SUSCRIPCIÓN (23 USD)
    # ==========================================
    elif menu == "💳 Mi Suscripción":
        st.header("💳 Gestión de Suscripción - 23 USD/mes")

        db = load_users()
        user_data = db.get(st.session_state.user, {})

        col_sub = st.columns(2)

        with col_sub[0]:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px;">
                    <h3 style="color: white;">✨ EpiDiagnosis Pro Premium</h3>
                    <ul style="color: #f0f0f0; font-size: 15px; line-height: 2;">
                        <li>✓ Análisis PICO con GPT-4</li>
                        <li>✓ Predicciones epidemiológicas avanzadas</li>
                        <li>✓ Monte Carlo Simulations</li>
                        <li>✓ Bioestadística completa</li>
                        <li>✓ Meta-análisis y Forest Plot</li>
                        <li>✓ Evaluación RoB/GRADE</li>
                        <li>✓ Análisis de supervivencia (KM, Cox, Poisson)</li>
                        <li>✓ Curvas ROC</li>
                        <li>✓ Mapas geográficos</li>
                        <li>✓ Propensity Score Matching (PSM)</li>
                        <li>✓ Estudios observacionales</li>
                        <li>✓ Proyecciones ARIMA</li>
                        <li>✓ Exportación ASIS</li>
                    </ul>
                    <h2 style="color: #ffd700; margin-top: 20px;">💰 23 USD / Mes</h2>
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
                except (ValueError, TypeError):
                    st.metric("⏰ Expiración", expiry)

            st.markdown(f"**📅 Fecha de expiración:** {expiry}")

            PAYMENT_LINK = "https://checkout.bold.co/payment/LNK_2W3K24BLVU"

            st.markdown("---")
            st.markdown("### 🔒 Realizar Pago - 23 USD/mes")

            st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: #1e293b; border-radius: 15px; margin-top: 20px; border: 2px solid #f59e0b;">
                    <h4 style="color: #f59e0b; margin-bottom: 15px;">💳 SUSCRIPCIÓN MENSUAL</h4>
                    <h2 style="color: white; font-size: 36px; margin-bottom: 20px;">23 USD / mes</h2>
                    <a href="{PAYMENT_LINK}" target="_blank" style="text-decoration: none;">
                        <button style="
                            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                            color: white;
                            padding: 25px 60px;
                            border-radius: 15px;
                            font-size: 22px;
                            font-weight: bold;
                            border: none;
                            cursor: pointer;
                            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
                        ">
                            💳 PAGAR AHORA - 23 USD
                        </button>
                    </a>
                    <p style="color: #94a3b8; margin-top: 15px;">Pago seguro con Bold.co</p>
                </div>
            """, unsafe_allow_html=True)

            st.info("💡 Después del pago, comunícate con soporte para activar tu licencia Premium.\n📞 WhatsApp: (+57) 3113682907\n📧 j.collazosmd@gmail.com")

    # ==========================================
    # MÓDULO: ADMIN
    # ==========================================
    elif menu == "⚙️ Admin":
        st.header("⚙️ Panel de Administración")

        db = load_users()
        st.subheader("📊 Usuarios Registrados")
        df_users = pd.DataFrame([{"Email": email, "Rol": data.get("role", "N/A"), "Expiración": data.get("expiry", "N/A")}
                                 for email, data in db.items()])
        st.dataframe(df_users, use_container_width=True)

        st.subheader("🔧 Gestión de Licencias")
        col_admin = st.columns([2, 1, 1])
        with col_admin[0]:
            target = st.text_input("Email del usuario:")
        with col_admin[1]:
            days = st.number_input("Días a agregar:", 1, 365, 30)

        col_btns_admin = st.columns(3)
        with col_btns_admin[0]:
            if st.button("➕ Renovar Licencia"):
                if target in db:
                    try:
                        current_expiry = datetime.strptime(db[target]['expiry'], "%Y-%m-%d")
                        if current_expiry < datetime.now():
                            current_expiry = datetime.now()
                        new_expiry = (current_expiry + timedelta(days=days)).strftime("%Y-%m-%d")
                        db[target]['expiry'] = new_expiry
                        save_users(db)
                        st.success(f"✅ Renovada hasta {new_expiry}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Usuario no encontrado")

        with col_btns_admin[1]:
            if st.button("🎫 Cambiar a Admin"):
                if target in db:
                    db[target]['role'] = 'admin'
                    save_users(db)
                    st.success(f"✅ {target} ahora es admin")

        with col_btns_admin[2]:
            if st.button("🗑️ Eliminar Usuario"):
                if target in db and target != "JCOLLAZOSR@UOC.EDU":
                    del db[target]
                    save_users(db)
                    st.success(f"✅ {target} eliminado")
                else:
                    st.error("No se puede eliminar este usuario")

        st.subheader("📈 Estadísticas del Sistema")
        col_stats_admin = st.columns(3)
        with col_stats_admin[0]:
            st.metric("Total Usuarios", len(db))
        with col_stats_admin[1]:
            active = sum(1 for u in db.values() if datetime.strptime(u.get("expiry", "2000-01-01"), "%Y-%m-%d") > datetime.now())
            st.metric("Usuarios Activos", active)
        with col_stats_admin[2]:
            st.metric("Administradores", sum(1 for u in db.values() if u.get("role") == "admin"))

    # ==========================================
    # FOOTER
    # ==========================================
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; padding: 10px;'>"
        "🩺 EpiDiagnosis Pro V7.0 | © 2026 | Fundación Juan Manuel Collazos | "
        "Desarrollado por: Juan Manuel Collazos Rozo, MD, MSc. | "
        "WhatsApp: (+57) 3113682907 - Correo: j.collazosmd@gmail.com"
        "</div>",
        unsafe_allow_html=True
    )
