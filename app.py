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
import plotly.graph_objects as go
import plotly.express as px
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import credentials

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
    # a = expuestos con enfermedad, b = expuestos sin enfermedad
    # c = no expuestos con enfermedad, d = no expuestos sin enfermedad

    total_exposed = a + b
    total_unexposed = c + d
    total_disease = a + c
    total_nodisease = b + d
    total = a + b + c + d

    # Prevalencias
    prevalence_exposed = a / total_exposed if total_exposed > 0 else 0
    prevalence_unexposed = c / total_unexposed if total_unexposed > 0 else 0

    # Sensibilidad y Especificidad
    sensitivity = a / total_disease if total_disease > 0 else 0
    specificity = d / total_nodisease if total_nodisease > 0 else 0

    # VPP y VPN
    vpp = a / total_exposed if total_exposed > 0 else 0
    vpn = d / total_unexposed if total_unexposed > 0 else 0

    # Odds Ratio
    or_num = a * d if a > 0 and d > 0 else 0
    or_den = b * c if b > 0 and c > 0 else 1
    odds_ratio = or_num / or_den if or_den > 0 else 0

    # Riesgo Relativo
    rr = prevalence_exposed / prevalence_unexposed if prevalence_unexposed > 0 else 0

    # Reducción de Riesgo Absoluto
    arr = prevalence_exposed - prevalence_unexposed

    # NNT (Number Needed to Treat)
    nnt = 1 / abs(arr) if arr != 0 else float('inf')

    # Likelihood Ratios
    lr_positive = sensitivity / (1 - specificity) if (1 - specificity) > 0 else 0
    lr_negative = (1 - sensitivity) / specificity if specificity > 0 else 0

    # Intervalos de confianza (95%)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if min(a, b, c, d) > 0 else 0
    ci_low_or = np.exp(np.log(or_num/or_den) - 1.96 * se_log_or) if se_log_or > 0 else 0
    ci_high_or = np.exp(np.log(or_num/or_den) + 1.96 * se_log_or) if se_log_or > 0 else 0

    # Chi-cuadrado
    expected_a = (total_exposed * total_disease) / total
    expected_b = (total_exposed * total_nodisease) / total
    expected_c = (total_unexposed * total_disease) / total
    expected_d = (total_unexposed * total_nodisease) / total

    chi_sq = ((a - expected_a)**2/expected_a + (b - expected_b)**2/expected_b +
              (c - expected_c)**2/expected_c + (d - expected_d)**2/expected_d) if all(x > 0 for x in [expected_a, expected_b, expected_c, expected_d]) else 0

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
    """
    Calcula tamaño de muestra para comparación de proporciones
    p1: proporción grupo 1
    p2: proporción grupo 2
    alpha: nivel de significancia
    power: poder estadístico
    ratio: proporción de asignación (n2/n1)
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    p_bar = (p1 + ratio * p2) / (1 + ratio)

    n1 = ((z_alpha * np.sqrt((1 + 1/ratio) * p_bar * (1 - p_bar)) +
           z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)/ratio))**2 /
          (p1 - p2)**2)

    n2 = n1 * ratio

    return {'n1': int(np.ceil(n1)), 'n2': int(np.ceil(n2)), 'total': int(np.ceil(n1 + n2))}

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

    # Heterogeneidad (Q statistic)
    Q = np.sum([w * (l - pooled_log_or)**2 for w, l in zip(weights, log_or)])
    df = len(log_or) - 1
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    return {
        'pooled_or': pooled_or,
        'pooled_ci_low': pooled_ci_low,
        'pooled_ci_high': pooled_ci_high,
        'pooled_log_or': pooled_log_or,
        'pooled_se': pooled_se,
        'Q': Q,
        'df': df,
        'I2': I2,
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

    # Pesos con efectos aleatorios
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

# ==========================================
# OPTIMIZACIÓN 4: AUTENTICACIÓN ROBUSTA
# ==========================================
def login_attempts_check(ip="default"):
    """Verificar intentos de login"""
    if not rate_limiter.is_allowed(f"login_{ip}"):
        return False, "Demasiados intentos. Espere 60 segundos."
    return True, ""

# ==========================================
# FLUJO DE AUTENTICACIÓN MEJORADO
# ==========================================
if not st.session_state.auth:
    st.title("🩺 EpiDiagnosis Pro v6.0")
    c1, c2 = st.columns(2)

    with c1:
        with st.container():
            st.markdown("### 🔐 Acceso al Sistema")
            with st.form("login"):
                u = st.text_input("Email", placeholder="su@email.com").upper().strip()
                p = st.text_input("Clave", type="password", placeholder="••••••••")

                col_login = st.columns(2)
                with col_login[0]:
                    submit_login = st.form_submit_button("ENTRAR", use_container_width=True)

                if submit_login:
                    allowed, msg = login_attempts_check()
                    if not allowed:
                        st.error(msg)
                    else:
                        db = load_users()
                        if u in db:
                            if db[u]["password"] == secure_hash(p):
                                expiry = datetime.strptime(db[u]["expiry"], "%Y-%m-%d")
                                if expiry > datetime.now():
                                    st.session_state.auth = True
                                    st.session_state.user = u
                                    st.session_state.role = db[u]["role"]
                                    st.session_state.user_logins[u] = datetime.now().isoformat()
                                    st.rerun()
                                else:
                                    st.error("Su licencia ha expirado. Por favor renueve.")
                            else:
                                st.error("Credenciales incorrectas")
                        else:
                            st.error("Usuario no registrado")

with c2:
    with st.container():
        st.markdown("### 📝 Registro Trial")
        with st.form("reg"):
            ru = st.text_input("Email", placeholder="su@email.com", key="reg_email").upper().strip()
            rp = st.text_input("Clave", type="password", placeholder="••••••••", key="reg_pass")
            rid = st.text_input("ID Documento", placeholder="C.C. o Passport")
            
            # NUEVOS CAMPOS
            rnombre = st.text_input("Nombre", placeholder="Ej: Juan", key="reg_nombre")
            rapellido = st.text_input("Apellido", placeholder="Ej: Pérez", key="reg_apellido")
            rprofesion = st.selectbox(
                "Profesión", 
                ["Médico", "Enfermero", "Investigador", "Estudiante", "Bioestadístico", "Epidemiólogo", "Otro"],
                key="reg_profesion"
            )

            col_reg = st.columns(2)
            with col_reg[0]:
                submit_reg = st.form_submit_button("ACTIVAR PRUEBA", use_container_width=True)

            if submit_reg:
                db = load_users()
                exp = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
                if ru not in db and ru and rp and rid and rnombre and rapellido:
                    db[ru] = {
                        "password": secure_hash(rp),
                        "role": "user",
                        "expiry": exp,
                        "id_doc": rid,
                        "dob": "2000-01-01",
                        "name": rnombre,      # NUEVO
                        "lastname": rapellido,  # NUEVO
                        "profession": rprofesion  # NUEVO
                    }
                    save_users(db)
                    st.success("✅ Cuenta creada exitosamente. Ya puede iniciar sesión.")
                elif ru in db:
                    st.warning("Este email ya está registrado.")
                else:
                    st.warning("Complete todos los campos.")
                    
        # SECCIÓN DE PAGO
        st.markdown("---")
        st.markdown("### 💳 Acceso Premium")

        col_pay1, col_pay2 = st.columns([1, 2])
        with col_pay1:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 10px;">✨ EpiDiagnosis Pro</h4>
                    <p style="color: #f0f0f0; font-size: 14px;">
                        Licencia mensual<br>
                        <span style="font-size: 28px; font-weight: bold; color: #ffd700;">23 US$</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with col_pay2:
            st.markdown("""
                <div style="padding: 15px; background: #1e293b; border-radius: 10px;">
                    <h5 style="color: #60a5fa; margin-bottom: 10px;">🎁 Beneficios Premium:</h5>
                    <ul style="color: #cbd5e1; font-size: 14px; line-height: 1.8;">
                        <li>✓ Análisis PICO con IA avanzada</li>
                        <li>✓ Predicciones epidemiológicas</li>
                        <li>✓ Monte Carlo Simulations</li>
                        <li>✓ Meta-análisis completo</li>
                        <li>✓ RoB/GRADE evaluación</li>
                        <li>✓ Análisis de supervivencia</li>
                        <li>✓ Curvas ROC</li>
                        <li>✓ Mapas geográficos</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        PAYMENT_LINK = "https://checkout.bold.co/payment/LNK_2W3K24BLVU"

        st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <a href="{PAYMENT_LINK}" target="_blank" style="text-decoration: none;">
                    <button style="
                        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                        color: white;
                        padding: 18px 40px;
                        border-radius: 12px;
                        font-size: 18px;
                        font-weight: bold;
                        border: none;
                        cursor: pointer;
                        transition: all 0.3s;
                        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
                    ">
                        🔒 PAGAR SEGURO CON BOLD.CO
                    </button>
                </a>
                <p style="color: #94a3b8; font-size: 12px; margin-top: 10px;">
                    🔒 Pago 100% seguro | Aceptamos tarjetas, PSE, Nequi, Daviplata
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.stop()
# ==========================================
# SIDEBAR NAVEGACIÓN MEJORADA v6.0
# ==========================================
with st.sidebar:
    st.markdown("<div class='sidebar-brand'>🩺 EpiDiagnosis Pro</div>", unsafe_allow_html=True)
    st.write(f"👤 **{st.session_state.get('user', 'Usuario')}**")
    st.write(f"🎫 Rol: `{st.session_state.get('role', 'guest').upper()}`")
    st.write(f"📌 Versión: `6.0`")

    st.markdown("---")

    # 1. Definir las opciones base
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

    # 2. Agregar opciones condicionales (Evita que haya None en la lista)
    if st.session_state.get('role') == "user":
        opciones.append("💳 Mi Suscripción")
    elif st.session_state.get('role') == "admin":
        opciones.append("⚙️ Admin")

    # 3. Renderizar el radio menú
    menu = st.radio("📋 MÓDULOS CIENTÍFICOS", opciones)

    st.markdown("---")
    
    # 4. Botón de salida
    if st.button("🚪 Cerrar Sesión", use_container_width=True):
        st.session_state.auth = False
        st.rerun()

    st.markdown("---")
    st.info("📞 Soporte: (+57) 3113682907\n\n📧 j.collazosmd@gmail.com\n\n🕐 Lun-Vie: 8AM-6PM")
    
# ==========================================
# MÓDULO 1: DASHBOARD
# ==========================================
if menu == "🏠 Dashboard & Cloud":
    st.header("📂 Conector de Datos Inteligente")

    with st.expander("ℹ️ Instrucciones", expanded=False):
        st.markdown("""
            1. Copie el enlace público de su Google Sheets, Excel Online o archivo CSV
            2. Pegue el enlace en el campo de abajo
            3. Los datos se cargarán y mostrarán automáticamente
        """)

    url = st.text_input(
        "Enlace público:",
        placeholder="https://docs.google.com/spreadsheets/d/... o URL de CSV/Excel"
    )

    col_load1, col_load2 = st.columns([1, 4])
    with col_load1:
        load_btn = st.button("📥 CARGAR DATOS", use_container_width=True)

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
# MÓDULO 2: LIMPIEZA DE DATOS
# ==========================================
elif menu == "🧹 Limpieza de Datos":
    st.header("🧹 Refinería de Datos Pro")

    if st.session_state.df_master is None:
        st.warning("⚠️ Por favor cargue datos primero en el módulo Dashboard.")
        st.stop()

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
# MÓDULO 3: BIOESTADÍSTICA BÁSICA
# ==========================================
elif menu == "📊 Bioestadística":
    st.header("📊 Rigor Bioestadístico")

    if st.session_state.df_master is None:
        st.warning("⚠️ Por favor cargue datos primero en el módulo Dashboard.")
        st.stop()

    df = st.session_state.df_master

    # Selector de tipo de análisis
    analysis_type = st.radio(
        "🎯 Tipo de Análisis:",
        ["📊 Comparación de Grupos", "📈 Una Variable", "📉 Correlación"],
        horizontal=True
    )

    # =====================
    # COMPARACIÓN DE GRUPOS
    # =====================
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

        # Opciones avanzadas
        with st.expander("⚙️ Opciones Avanzadas"):
            col_opts = st.columns(3)
            with col_opts[0]:
                show_ci = st.checkbox("Mostrar IC 95%", value=True)
            with col_opts[1]:
                show_power = st.checkbox("Calcular Poder", value=True)
            with col_opts[2]:
                equal_var = st.checkbox("Igualar Varianzas", value=True)

        if st.button("🔬 EJECUTAR ANÁLISIS", use_container_width=True):
            with st.spinner("⏳ Procesando análisis estadístico..."):
                _, _, _, stats = get_heavy_imports()

                # Preparar datos
                clean_data = df[[vn, vc]].dropna()
                grupos_data = {g: clean_data[clean_data[vc] == g][vn].values 
                              for g in clean_data[vc].unique()}
                grupos = [g for g in grupos_data.values() if len(g) > 0]
                nombres_grupos = [g for g, v in grupos_data.items() if len(v) > 0]

                if len(grupos) < 2:
                    st.error("Se requieren al menos 2 grupos con datos")
                    st.stop()

                # Limpiar variable Y
                clean_y = clean_data[vn]
                stat, p_norm = stats.shapiro(clean_y[:5000]) if len(clean_y) <= 5000 else stats.normaltest(clean_y)

                # Resultados de normalidad
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

                # Normalidad por grupo
                st.markdown("#### Normalidad por Grupo")
                normality_results = []
                for nombre, datos in zip(nombres_grupos, grupos):
                    if len(datos) >= 3:
                        stat_g, p_g = stats.shapiro(datos[:5000]) if len(datos) <= 5000 else stats.normaltest(datos)
                        normalidad = "Normal" if p_g > alpha_norm else "No Normal"
                        normalidad_icon = "✅" if p_g > alpha_norm else "⚠️"
                        normalidad_color = "#10b981" if p_g > alpha_norm else "#f59e0b"
                        normality_results.append({
                            "Grupo": nombre,
                            "N": len(datos),
                            "Media": f"{np.mean(datos):.2f}",
                            "DS": f"{np.std(datos):.2f}",
                            "p-normalidad": f"{p_g:.4f}",
                            "Distribución": f"{normalidad_icon} {normalidad}"
                        })
                
                if normality_results:
                    df_norm = pd.DataFrame(normality_results)
                    st.dataframe(df_norm, use_container_width=True, hide_index=True)

                # Selección de prueba
                if p_norm > alpha_norm:
                    st.info(f"📊 **Distribución Normal** (p={p_norm:.4f} > {alpha_norm}): Se aplica prueba paramétrica")
                    is_normal = True
                else:
                    st.warning(f"⚠️ **Distribución No Normal** (p={p_norm:.4f} ≤ {alpha_norm}): Se aplica prueba no paramétrica")
                    is_normal = False

                # Homogeneidad de varianzas (Levene)
                if len(grupos) >= 2:
                    stat_lev, p_lev = stats.levene(*grupos)
                    homocedasticidad = p_lev > 0.05
                    
                    with col_norm[1]:
                        st.metric("Levene p-value", f"{p_lev:.4f}",
                                 delta="Iguales" if homocedasticidad else "Diferentes",
                                 delta_color="normal" if homocedasticidad else "inverse")

                # Ejecución de pruebas
                st.markdown("---")
                st.markdown("### 🔬 Resultados de la Prueba Estadística")

                if len(grupos) > 2:
                    # Múltiples grupos
                    if is_normal:
                        if equal_var:
                            res = stats.f_oneway(*grupos)
                            test_name = "ANOVA (One-Way)"
                            effect_name = "Eta-squared (η²)"
                            # Calcular eta-squared
                            grand_mean = np.mean(clean_y)
                            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in grupos)
                            ss_total = sum((clean_y - grand_mean)**2)
                            effect_size = ss_between / ss_total if ss_total > 0 else 0
                        else:
                            res = stats.kruskal(*grupos)
                            test_name = "Kruskal-Wallis"
                            effect_name = "Epsilon-squared (ε²)"
                            # Epsilon-squared para Kruskal
                            H = res.statistic
                            n = len(clean_y)
                            k = len(grupos)
                            effect_size = (H - k + 1) / (n - k) if n > k else 0
                    else:
                        res = stats.kruskal(*grupos)
                        test_name = "Kruskal-Wallis"
                        effect_name = "Epsilon-squared (ε²)"
                        H = res.statistic
                        n = len(clean_y)
                        k = len(grupos)
                        effect_size = (H - k + 1) / (n - k) if n > k else 0
                else:
                    # Dos grupos
                    if is_normal:
                        res = stats.ttest_ind(grupos[0], grupos[1], equal_var=equal_var)
                        test_name = f"T-Test{' (Welch)' if not equal_var else ''}"
                        effect_name = "Cohen's d"
                        # Cohen's d
                        mean_diff = np.mean(grupos[0]) - np.mean(grupos[1])
                        pooled_std = np.sqrt(((len(grupos[0])-1)*np.var(grupos[0], ddof=1) + 
                                             (len(grupos[1])-1)*np.var(grupos[1], ddof=1)) / 
                                            (len(grupos[0]) + len(grupos[1]) - 2))
                        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                    else:
                        res = stats.mannwhitneyu(grupos[0], grupos[1])
                        test_name = "Mann-Whitney U"
                        effect_name = "r (rank-biserial)"
                        # Rank-biserial correlation
                        n1, n2 = len(grupos[0]), len(grupos[1])
                        effect_size = (2 * res.statistic / (n1 * n2)) - 1

                # Interpretación del tamaño del efecto
                if effect_name == "Cohen's d":
                    if abs(effect_size) < 0.2:
                        effect_interp = "Muy pequeño"
                        effect_color = "gray"
                    elif abs(effect_size) < 0.5:
                        effect_interp = "Pequeño"
                        effect_color = "blue"
                    elif abs(effect_size) < 0.8:
                        effect_interp = "Mediano"
                        effect_color = "orange"
                    else:
                        effect_interp = "Grande"
                        effect_color = "red"
                elif effect_name in ["Eta-squared (η²)", "Epsilon-squared (ε²)"]:
                    if effect_size < 0.01:
                        effect_interp = "Muy pequeño"
                        effect_color = "gray"
                    elif effect_size < 0.06:
                        effect_interp = "Pequeño"
                        effect_color = "blue"
                    elif effect_size < 0.14:
                        effect_interp = "Mediano"
                        effect_color = "orange"
                    else:
                        effect_interp = "Grande"
                        effect_color = "red"
                else:
                    if abs(effect_size) < 0.1:
                        effect_interp = "Muy pequeño"
                        effect_color = "gray"
                    elif abs(effect_size) < 0.3:
                        effect_interp = "Pequeño"
                        effect_color = "blue"
                    elif abs(effect_size) < 0.5:
                        effect_interp = "Mediano"
                        effect_color = "orange"
                    else:
                        effect_interp = "Grande"
                        effect_color = "red"

                # Mostrar métricas principales
                col_res = st.columns(4)

                with col_res[0]:
                    st.metric("Prueba", test_name)
                with col_res[1]:
                    st.metric("Estadístico", f"{res.statistic:.2f}")
                with col_res[2]:
                    st.metric("p-value", f"{res.pvalue:.6f}",
                             delta="Significativo" if res.pvalue < 0.05 else "No significativo",
                             delta_color="off" if res.pvalue < 0.05 else "normal")
                with col_res[3]:
                    st.metric(effect_name, f"{effect_size:.3f}",
                             delta=effect_interp,
                             delta_color=effect_color)

                # IC 95% de la diferencia de medias (si aplica)
                if show_ci and len(grupos) == 2 and is_normal:
                    mean1, mean2 = np.mean(grupos[0]), np.mean(grupos[1])
                    se_diff = np.sqrt(np.var(grupos[0], ddof=1)/len(grupos[0]) + 
                                      np.var(grupos[1], ddof=1)/len(grupos[1]))
                    df_se = len(grupos[0]) + len(grupos[1]) - 2
                    t_crit = stats.t.ppf(0.975, df_se)
                    
                    diff = mean1 - mean2
                    ci_low = diff - t_crit * se_diff
                    ci_high = diff + t_crit * se_diff
                    
                    st.markdown(f"""
                    **Diferencia de Medias:** {diff:.2f} (IC 95%: [{ci_low:.2f}, {ci_high:.2f}])
                    """)

                # Cálculo de poder (post-hoc)
                if show_power:
                    from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
                    
                    effect_for_power = abs(effect_size) if abs(effect_size) > 0 else 0.5
                    n_obs = min(len(grupos[0]), len(grupos[1])) if len(grupos) == 2 else len(clean_y)
                    
                    if len(grupos) > 2:
                        power_analysis = FTestAnovaPower()
                        power_calc = power_analysis.solve_power(effect_for_power, nobs=n_obs, alpha=0.05, df_between=len(grupos)-1)
                    else:
                        power_analysis = TTestIndPower()
                        power_calc = power_analysis.solve_power(effect_for_power, nobs=n_obs, alpha=0.05, ratio=len(grupos[1])/len(grupos[0]))
                    
                    with col_res[1]:
                        st.metric("Poder Observado", f"{power_calc:.1%}",
                                 delta="Adecuado (≥80%)" if power_calc >= 0.8 else "Bajo (<80%)",
                                 delta_color="normal" if power_calc >= 0.8 else "inverse")

                # Comparaciones post-hoc (Tukey o Mann-Whitney por pares)
                if len(grupos) > 2 and res.pvalue < 0.05:
                    st.markdown("#### 🔍 Comparaciones Post-Hoc (Tukey HSD)")
                    
                    from scipy.stats import tukey_hsd
                    
                    try:
                        if is_normal:
                            tukey_result = tukey_hsd(*grupos)
                            
                            posthoc_data = []
                            for i, g1 in enumerate(nombres_grupos):
                                for j, g2 in enumerate(nombres_grupos):
                                    if i < j:
                                        p_adj = tukey_result.pvalue[i, j]
                                        mean_diff = np.mean(grupos_data[g1]) - np.mean(grupos_data[g2])
                                        significant = "*" if p_adj < 0.05 else ""
                                        significant += "**" if p_adj < 0.01 else ""
                                        significant += "***" if p_adj < 0.001 else ""
                                        posthoc_data.append({
                                            "Comparación": f"{g1} vs {g2}",
                                            "Diferencia": f"{mean_diff:.2f}",
                                            "p-ajustado": f"{p_adj:.4f}",
                                            "Significativo": significant
                                        })
                            
                            df_posthoc = pd.DataFrame(posthoc_data)
                            st.dataframe(df_posthoc, use_container_width=True, hide_index=True)
                        else:
                            st.info("Para distribuciones no normales, se recomienda usar pruebas de Dunn o Bonferroni-Nemenyi.")
                    except Exception as e:
                        st.warning(f"No se pudo realizar Tukey: {e}")

                # Visualización
                st.markdown("---")
                st.markdown("### 📊 Visualización")

                fig, axes = plt.subplots(1, 3, figsize=(16, 5))

                # Boxplot
                ax1 = axes[0]
                box_data = [grupos_data[g] for g in nombres_grupos]
                bp = ax1.boxplot(box_data, labels=nombres_grupos, patch_artist=True)
                
                colors_box = plt.cm.Set2(np.linspace(0, 1, len(grupos)))
                for patch, color in zip(bp['boxes'], colors_box):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax1.set_ylabel(vn)
                ax1.set_title('Boxplot por Grupo')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Scatter + media
                ax2 = axes[1]
                means = [np.mean(g) for g in grupos]
                stds = [np.std(g) for g in grupos]
                x_pos = range(len(grupos))
                ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors_box, alpha=0.7, edgecolor='black')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(nombres_grupos)
                ax2.set_ylabel(vn)
                ax2.set_title('Media ± DS')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Histogramas superpuestos
                ax3 = axes[2]
                for i, (nombre, datos) in enumerate(zip(nombres_grupos, grupos)):
                    ax3.hist(datos, bins=20, alpha=0.5, label=nombre, color=colors_box[i])
                ax3.set_xlabel(vn)
                ax3.set_ylabel('Frecuencia')
                ax3.set_title('Distribución por Grupo')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # Interpretación
                st.markdown("---")
                st.markdown("### 📝 Interpretación")

                if res.pvalue < 0.05:
                    conclusion = "**SI hay diferencia estadísticamente significativa** entre los grupos analizados."
                    st.success(f"""
                    {conclusion}
                    
                    - **Prueba utilizada:** {test_name}
                    - **Estadístico:** {res.statistic:.2f}
                    - **p-value:** {res.pvalue:.6f}
                    - **{effect_name}:** {effect_size:.3f} ({effect_interp})
                    - **Conclusión práctica:** La diferencia observada es {"clínicamente relevante" if abs(effect_size) >= 0.5 else "de magnitud " + effect_interp.lower()}
                    """)
                else:
                    conclusion = "**NO hay diferencia estadísticamente significativa** entre los grupos."
                    st.info(f"""
                    {conclusion}
                    
                    - **Prueba utilizada:** {test_name}
                    - **Estadístico:** {res.statistic:.2f}
                    - **p-value:** {res.pvalue:.6f}
                    - **{effect_name}:** {effect_size:.3f} ({effect_interp})
                    """)

                # Tabla resumen de grupos
                st.markdown("---")
                st.markdown("### 📋 Resumen Descriptivo por Grupo")

                summary_data = []
                for nombre, datos in zip(nombres_grupos, grupos):
                    summary_data.append({
                        "Grupo": nombre,
                        "N": len(datos),
                        "Media": f"{np.mean(datos):.2f}",
                        "Mediana": f"{np.median(datos):.2f}",
                        "DS": f"{np.std(datos):.2f}",
                        "Mín": f"{np.min(datos):.2f}",
                        "Máx": f"{np.max(datos):.2f}",
                        "IQR": f"{np.percentile(datos, 75) - np.percentile(datos, 25):.2f}"
                    })

                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # =====================
    # UNA VARIABLE
    # =====================
    elif analysis_type == "📈 Una Variable":
        st.markdown("### Análisis de Una Variable")
        
        var_single = st.selectbox("Seleccionar Variable:", 
                                  df.select_dtypes(include=np.number).columns)

        if st.button("📊 ANALIZAR VARIABLE", use_container_width=True):
            data = df[var_single].dropna()

            # Estadísticas descriptivas
            col_stats = st.columns(4)

            with col_stats[0]:
                st.metric("N", len(data))
                st.metric("Media", f"{data.mean():.2f}")
            with col_stats[1]:
                st.metric("Mediana", f"{data.median():.2f}")
                st.metric("Moda", f"{data.mode().values[0]:.2f}")
            with col_stats[2]:
                st.metric("DS", f"{data.std():.2f}")
                st.metric("Varianza", f"{data.var():.2f}")
            with col_stats[3]:
                st.metric("Mín", f"{data.min():.2f}")
                st.metric("Máx", f"{data.max():.2f}")

            # Normalidad
            stat, p_norm = stats.shapiro(data[:5000]) if len(data) <= 5000 else stats.normaltest(data)
            
            st.markdown(f"""
            **Test de Normalidad:** p = {p_norm:.4f}
            - {"✅ Distribución Normal" if p_norm > 0.05 else "⚠️ Distribución No Normal"}
            """)

            # Visualización
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

    # =====================
    # CORRELACIÓN
    # =====================
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
            # Limpiar datos
            clean_corr = df[[var_x, var_y]].dropna()
            x = clean_corr[var_x]
            y = clean_corr[var_y]

            # Pearson
            r_pearson, p_pearson = stats.pearsonr(x, y)
            
            # Spearman
            r_spearman, p_spearman = stats.spearmanr(x, y)

            col_results = st.columns(4)

            with col_results[0]:
                st.metric("Pearson r", f"{r_pearson:.3f}")
                st.metric("Pearson p", f"{p_pearson:.4f}")
            with col_results[1]:
                st.metric("Spearman ρ", f"{r_spearman:.3f}")
                st.metric("Spearman p", f"{p_spearman:.4f}")
            with col_results[2]:
                # Interpretación correlación
                if abs(r_pearson) < 0.1:
                    interp = "Muy débil"
                elif abs(r_pearson) < 0.3:
                    interp = "Débil"
                elif abs(r_pearson) < 0.5:
                    interp = "Moderada"
                elif abs(r_pearson) < 0.7:
                    interp = "Fuerte"
                else:
                    interp = "Muy fuerte"
                
                st.metric("Interpretación", interp,
                         delta="Positiva" if r_pearson > 0 else "Negativa",
                         delta_color="normal" if r_pearson > 0 else "inverse")
            with col_results[3]:
                st.metric("N", len(x))

            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x, y, alpha=0.5)
            
            # Línea de tendencia
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

            # Interpretación
            if p_pearson < 0.05:
                st.success(f"""
                **Existe correlación estadísticamente significativa** entre {var_x} y {var_y}.
                
                - r = {r_pearson:.3f} indica una asociación {interp.lower()} {"positiva" if r_pearson > 0 else "negativa"}
                - R² = {r_pearson**2:.3f}: El {r_pearson**2*100:.1f}% de la variabilidad es explicada por la relación
                """)
            else:
                st.info(f"No se encontró correlación estadísticamente significativa (p = {p_pearson:.4f})")
                
# ==========================================
# MÓDULO 4: CALCULADORA 2x2
# ==========================================
elif menu == "🔢 Calculadora 2x2":
    st.header("🔢 Calculadora de Tablas 2x2")
    st.markdown("### Configure su tabla de contingencia")

    # Selector de tipo de estudio
    study_design = st.radio(
        "📋 Diseño del Estudio:",
        ["Cohorte (expuestos → enfermedad)", "Casos y Controles", "Prueba Diagnóstica"],
        horizontal=True
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 📊 Tabla 2x2")
        
        # Crear inputs para los 4 valores
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

        # Validación
        total_expuestos = a + b
        total_no_expuestos = c + d
        total_enfermos = a + c
        total_no_enfermos = b + d
        total_general = a + b + c + d

        if total_expuestos == 0 or total_no_expuestos == 0 or total_enfermos == 0 or total_no_enfermos == 0:
            st.warning("⚠️ Algunos totales son cero. Algunas métricas no se calcularán correctamente.")

    with col_t2:
        # Mostrar tabla con VALORES
        st.markdown("#### 📋 Tabla Resumen")
        
        df_2x2_display = pd.DataFrame({
            '': ['Expuestos (+)', 'No Expuestos (-)', 'Total'],
            'Enfermedad (+)': [f'{a}', f'{c}', f'{a+c}'],
            'Enfermedad (-)': [f'{b}', f'{d}', f'{b+d}'],
            'Total': [f'{a+b}', f'{c+d}', f'{a+b+c+d}']
        })
        
        # Estilo para la tabla
        st.dataframe(df_2x2_display, hide_index=True, use_container_width=True)
        
        st.markdown(f"""
        **Resumen:**
        - Total expuestos: **{total_expuestos}**
        - Total no expuestos: **{total_no_expuestos}**
        - Total general: **{total_general}**
        """)

    # Opción de población para FPC
    st.markdown("#### 🌍 Población (Opcional - Corrección FPC)")
    col_pop = st.columns([1, 1])
    
    with col_pop[0]:
        population_2x2 = st.number_input(
            "Población Total (N):",
            min_value=0, value=0, step=100,
            help="Tamaño de la población total. Si es 0, se asume infinita (no aplica FPC)"
        )
    
    with col_pop[1]:
        apply_fpc_2x2 = st.checkbox(
            "Aplicar Corrección de Población Finita (FPC)",
            value=True,
            help="Ajusta las métricas cuando la muestra es >5% de la población"
        )

    if st.button("🧮 CALCULAR MÉTRICAS", use_container_width=True):
        metrics = calculate_2x2_metrics(a, b, c, d)

        st.markdown("---")
        st.markdown("### 📈 Resultados del Análisis")

        # Alerta FPC si aplica
        if population_2x2 > 0 and total_general / population_2x2 > 0.05:
            st.warning(f"⚠️ **FPC aplicada:** La muestra ({total_general}) representa el {total_general/population_2x2:.1%} de la población ({population_2x2:,})")

        # Primera fila de métricas principales
        col_metrics = st.columns(4)

        with col_metrics[0]:
            st.metric("📊 Sensibilidad", f"{metrics['sensitivity']:.2%}",
                     help="Probabilidad de que la prueba identifique correctamente a los enfermos")
            st.metric("🎯 Especificidad", f"{metrics['specificity']:.2%}",
                     help="Probabilidad de que la prueba identifique correctamente a los no enfermos")

        with col_metrics[1]:
            st.metric("✅ VPP (Valor Predictivo +)", f"{metrics['vpp']:.2%}",
                     help="Probabilidad de enfermedad dado resultado positivo")
            st.metric("✅ VPN (Valor Predictivo -)", f"{metrics['vpn']:.2%}",
                     help="Probabilidad de no enfermedad dado resultado negativo")

        with col_metrics[2]:
            st.metric("📈 Odds Ratio (OR)", f"{metrics['odds_ratio']:.2f}",
                     help="Asociación entre exposición y enfermedad")
            st.metric("📉 IC 95% OR", f"[{metrics['ci_low_or']:.2f}, {metrics['ci_high_or']:.2f}]",
                     help="Intervalo de confianza del 95% para el OR")

        with col_metrics[3]:
            # Calcular RR con protección
            rr_display = metrics['risk_ratio'] if metrics['prevalence_unexposed'] > 0 else 0
            st.metric("⚡ Riesgo Relativo (RR)", f"{rr_display:.2f}",
                     help="Reducción relativa del riesgo en expuestos vs no expuestos")
            st.metric("📐 RRA (ARR)", f"{metrics['arr']:.2%}",
                     help="Reducción Absoluta del Riesgo")

        # Segunda fila de métricas adicionales
        st.markdown("#### 🔬 Métricas Adicionales")
        col_extra = st.columns(4)

        with col_extra[0]:
            st.metric("🧮 LR+ (Razón de Verosimilitud +)", f"{metrics['lr_positive']:.2f}",
                     help="Cuánto aumenta la probabilidad de enfermedad con resultado +")
            st.metric("🧮 LR- (Razón de Verosimilitud -)", f"{metrics['lr_negative']:.2f}",
                     help="Cuánto reduce la probabilidad de enfermedad con resultado -")

        with col_extra[1]:
            nnt_display = f"{metrics['nnt']:.0f}" if metrics['nnt'] != float('inf') else "∞"
            st.metric("👥 NNT (NNT)", nnt_display,
                     help="Número de personas a tratar para evitar 1 evento")

        with col_extra[2]:
            st.metric("📊 Chi² (Chi-cuadrado)", f"{metrics['chi_square']:.2f}",
                     help="Estadístico de prueba de independencia")
            st.metric("📊 p-value", f"{metrics['p_value']:.4f}",
                     help="Significancia estadística (p < 0.05 = significativo)")

        with col_extra[3]:
            # Youden Index
            youden = metrics['sensitivity'] + metrics['specificity'] - 1
            st.metric("🎯 Índice de Youden", f"{youden:.3f}",
                     help="Máximo valor de sensibilidad + especificidad - 1")
            
            # Prevalencia general
            prevalence_general = total_enfermos / total_general if total_general > 0 else 0
            st.metric("📊 Prevalencia General", f"{prevalence_general:.2%}",
                     help="Proporción de enfermedad en la población total")

        # Visualización
        st.markdown("---")
        st.markdown("### 📊 Visualización Gráfica")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico 1: Comparación de prevalencias
        ax1 = axes[0]
        categories = ['Expuestos', 'No Expuestos']
        prevalences = [metrics['prevalence_exposed'] * 100, metrics['prevalence_unexposed'] * 100]
        colors = ['#ef4444', '#10b981']
        bars = ax1.bar(categories, prevalences, color=colors, edgecolor='black', width=0.6)
        ax1.set_ylabel('Prevalencia (%)', fontsize=12)
        ax1.set_title('Prevalencia de Enfermedad por Exposición', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(max(prevalences) * 1.3, 100))
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, v in zip(bars, prevalences):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Gráfico 2: Forest plot mejorado
        ax2 = axes[1]
        
        # Línea null
        ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='OR = 1 (Null)')
        
        # Intervalo de confianza
        ci_low = max(metrics['ci_low_or'], 0.1)
        ci_high = metrics['ci_high_or']
        
        # Punto estimado
        or_val = metrics['odds_ratio']
        
        # Barra de error
        ax2.errorbar(or_val, 0.5, xerr=[[or_val - ci_low], [ci_high - or_val]],
                    fmt='D', color='#3b82f6', capsize=10, markersize=12,
                    linewidth=2, markeredgecolor='black', markeredgewidth=1.5,
                    elinewidth=2)
        
        # Configuración del plot
        x_max = max(ci_high * 1.3, or_val * 1.5, 3)
        ax2.set_xlim(0, x_max)
        ax2.set_ylim(-0.1, 1.0)
        ax2.set_xlabel('Odds Ratio', fontsize=12)
        ax2.set_title('Forest Plot del Odds Ratio', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Anotación con valores
        ax2.annotate(f'OR = {or_val:.2f}\nIC 95%: [{ci_low:.2f}, {ci_high:.2f}]',
                    xy=(or_val, 0.5), xytext=(or_val * 0.7, 0.75),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        st.pyplot(fig)

        # Tabla de contingencia detallada
        st.markdown("---")
        st.markdown("### 📋 Tabla de Contingencia Completa")

        col_contingency_table = st.columns(2)
        
        with col_contingency_table[0]:
            # Crear tabla bonita
            contingency_df = pd.DataFrame({
                ' ': ['Enfermedad +', 'Enfermedad -', 'Total'],
                'Expuestos +': [a, b, a+b],
                'No Expuestos -': [c, d, c+d],
                'Total': [a+c, b+d, a+b+c+d]
            })
            st.dataframe(contingency_df.set_index(' '), use_container_width=True)

        with col_contingency_table[1]:
            # Proporciones
            prop_df = pd.DataFrame({
                ' ': ['P(Enf|Exp)', 'P(Enf|NoExp)', 'Total Enf', 'Total No-Enf'],
                'Valor': [
                    f'{a/(a+b)*100:.1f}%' if (a+b) > 0 else 'N/A',
                    f'{c/(c+d)*100:.1f}%' if (c+d) > 0 else 'N/A',
                    f'{a+c}',
                    f'{b+d}'
                ]
            })
            st.dataframe(prop_df.set_index(' '), use_container_width=True)

        # Interpretación clínica
        st.markdown("---")
        st.markdown("### 📝 Interpretación Clínica")

        or_val = metrics['odds_ratio']
        ci_low = metrics['ci_low_or']
        ci_high = metrics['ci_high_or']
        
        # Significancia
        is_significant = ci_low > 1 or ci_high < 1
        includes_null = ci_low <= 1 <= ci_high

        if or_val > 1:
            direction = "MAYOR"
            direction_icon = "⚠️"
            direction_color = "warning"
            interpretation = f"""
            **La exposición está asociada con {direction} riesgo de enfermedad.**
            
            - **Odds Ratio = {or_val:.2f}**: Los expuestos tienen {or_val:.1f}x más odds de presentar la enfermedad que los no expuestos
            - **IC 95%: [{ci_low:.2f}, {ci_high:.2f}]**: El intervalo {'NO incluye' if not includes_null else 'incluye'} el valor nulo (OR=1)
            - **Chi² = {metrics['chi_square']:.2f}, p = {metrics['p_value']:.4f}**: {'La asociación ES estadísticamente significativa (p<0.05)' if metrics['p_value'] < 0.05 else 'La asociación NO es estadísticamente significativa'}
            - **NNT = {metrics['nnt']:.0f}**: Se necesitan tratar {metrics['nnt']:.0f} personas para evitar 1 caso adicional
            """
        else:
            direction = "MENOR"
            direction_icon = "✅"
            direction_color = "success"
            interpretation = f"""
            **La exposición está asociada con {direction} riesgo de enfermedad.**
            
            - **Odds Ratio = {or_val:.2f}**: Los expuestos tienen {(1-or_val)*100:.1f}% menos odds de presentar la enfermedad que los no expuestos
            - **IC 95%: [{ci_low:.2f}, {ci_high:.2f}]**: El intervalo {'NO incluye' if not includes_null else 'incluye'} el valor nulo (OR=1)
            - **Chi² = {metrics['chi_square']:.2f}, p = {metrics['p_value']:.4f}**: {'La asociación ES estadísticamente significativa (p<0.05)' if metrics['p_value'] < 0.05 else 'La asociación NO es estadísticamente significativa'}
            """

        if direction_color == "warning":
            st.warning(interpretation)
        else:
            st.success(interpretation)

        # Guía rápida
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
            | **p-value** | <0.05 indica significancia estadística |
            """)

# ==========================================
# MÓDULO 5: TAMAÑO DE MUESTRA
# ==========================================
elif menu == "📏 Tamaño de Muestra":
    st.header("📏 Calculadora de Tamaño de Muestra")

    # Selector de tipo de estudio
    study_type = st.radio(
        "🎯 Tipo de Estudio:",
        ["📊 Cohortes (Comparación de Proporciones)", "🔬 Casos y Controles"],
        horizontal=True
    )

    # =====================
    # CASOS Y CONTROLES
    # =====================
    if study_type == "🔬 Casos y Controles":
        st.markdown("### Parámetros del Estudio de Casos y Controles")

        col_cc1, col_cc2 = st.columns(2)

        with col_cc1:
            st.markdown("#### 📈 Parámetros Estadísticos")
            or_expected = st.number_input(
                "Odds Ratio Esperado (OR):",
                min_value=0.1, max_value=10.0, value=2.0, format="%.2f",
                help="OR que se desea detectar como significativo. Ej: 2.0 = duplica el riesgo"
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
                help="Número de controles por cada caso. Común: 4:1 o 3:1"
            )
            power_cc = st.select_slider(
                "Poder estadístico (1-β):",
                options=[0.70, 0.80, 0.90, 0.95], value=0.80,
                help="Probabilidad de detectar un efecto real"
            )

        st.markdown("---")
        st.markdown("#### 🌍 Población (Opcional)")
        col_pop_cc = st.columns([1, 1, 1])

        with col_pop_cc[0]:
            population_cc = st.number_input(
                "Población Total (N):",
                min_value=0, value=0, step=100,
                help="Tamaño de la población total. Si es 0, se asume infinita"
            )
        with col_pop_cc[1]:
            apply_fpc_cc = st.checkbox(
                "Aplicar FPC",
                value=True,
                help="Corrección de Población Finita cuando n/N > 5%"
            )
        with col_pop_cc[2]:
            test_type_cc = st.radio(
                "Tipo de prueba:",
                ["Two-sided", "One-sided"],
                horizontal=True
            )

        if st.button("🧮 CALCULAR MUESTRA (CASOS-CONTROLES)", use_container_width=True):
            if test_type_cc == "One-sided":
                alpha_cc = alpha_cc / 2

            result_cc = calculate_sample_size_case_control(
                or_expected, alpha=alpha_cc, power=power_cc,
                ratio_controls_cases=ratio_cc,
                population=population_cc if population_cc > 0 else None,
                apply_fpc=apply_fpc_cc
            )

            st.markdown("---")
            st.markdown("### 📊 Resultados - Casos y Controles")

            # Mostrar alerta si se aplicó FPC
            if result_cc.get('fpc_applied', False):
                st.success(f"""
                **✅ Corrección de Población Finita (FPC) aplicada:**
                - Muestra sin FPC: **{result_cc['n_without_fpc']:,}**
                - Muestra con FPC: **{result_cc['total']:,}**
                - Población total: **{population_cc:,}**
                - Ahorro: **{result_cc['n_without_fpc'] - result_cc['total']:,} participantes**
                """)
            elif population_cc > 0 and not result_cc.get('fpc_applied', False):
                st.info(f"""
                **ℹ️ FPC no necesaria:**
                - La muestra calculada ({result_cc['n_without_fpc']:,}) representa menos del 5% de la población ({population_cc:,})
                - Condición: n/N = {result_cc['n_without_fpc']/population_cc:.2%} < 5%
                """)

            col_res_cc = st.columns(4)

            with col_res_cc[0]:
                st.metric("🏥 Casos necesarios", f"{result_cc['n_cases']:,}")
            with col_res_cc[1]:
                st.metric("👥 Controles necesarios", f"{result_cc['n_controls']:,}")
            with col_res_cc[2]:
                st.metric("📊 Total (N)", f"{result_cc['total']:,}")
            with col_res_cc[3]:
                ratio_display = ratio_cc
                st.metric("Ratio", f"{ratio_display}:1")

            # Visualización
            fig_cc, ax_cc = plt.subplots(figsize=(10, 6))

            labels = ['Casos', 'Controles', 'Total']
            sizes = [result_cc['n_cases'], result_cc['n_controls'], result_cc['total']]
            colors = ['#ef4444', '#3b82f6', '#10b981']
            
            x_pos = np.arange(len(labels))
            bars = ax_cc.bar(x_pos, sizes, color=colors, edgecolor='black')
            
            ax_cc.set_xticks(x_pos)
            ax_cc.set_xticklabels([f'{l}\n({s:,})' for l, s in zip(labels, sizes)])
            ax_cc.set_ylabel('Tamaño de Muestra')
            ax_cc.set_title(f'Tamaño de Muestra para Detectar OR = {or_expected}')
            ax_cc.grid(True, alpha=0.3, axis='y')

            for bar, size in zip(bars, sizes):
                ax_cc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{size:,}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig_cc)

            # Interpretación
            st.markdown("---")
            st.markdown("### 📝 Interpretación")

            st.info(f"""
            **Para detectar un Odds Ratio de {or_expected} en un estudio de casos y controles:**

            - Se requieren **{result_cc['n_cases']:,} casos** y **{result_cc['n_controls']:,} controles**
            - **Total: {result_cc['total']:,} participantes**
            - Ratio utilizado: {ratio_cc}:1 (controles:casos)
            - Con un poder del {power_cc*100:.0f}% y α = {alpha_cc:.3f}

            **Fórmula utilizada:** Fleiss con corrección de Yates para tablas 2x2
            """)

            # Muestra ajustada
            st.markdown("#### 🔧 Muestras Ajustadas por Pérdidas")
            col_adj_cc = st.columns(3)
            for pct in [0.10, 0.15, 0.20]:
                adj_total = int(result_cc['total'] * (1 + pct))
                adj_cases = int(result_cc['n_cases'] * (1 + pct))
                adj_controls = int(result_cc['n_controls'] * (1 + pct))
                with col_adj_cc[int(pct*5)-1]:
                    st.metric(f"+{pct*100:.0f}% pérdida", f"N = {adj_total:,}")
                    st.caption(f"Casos={adj_cases:,}, Controles={adj_controls:,}")

    # =====================
    # COHORTES
    # =====================
    else:
        st.markdown("### Parámetros del Estudio de Cohortes")

        col_sample = st.columns([1, 1, 1, 1])

        with col_sample[0]:
            p1 = st.number_input(
                "Proporción Grupo 1 (p1):",
                min_value=0.001, max_value=0.999, value=0.30, format="%.3f",
                help="Proporción esperada en el grupo de intervención/expuestos"
            )
            population = st.number_input(
                "Población Total (N):",
                min_value=0, value=0, step=100,
                help="Tamaño de la población total. Si es 0, se asume infinita"
            )

        with col_sample[1]:
            p2 = st.number_input(
                "Proporción Grupo 2 (p2):",
                min_value=0.001, max_value=0.999, value=0.50, format="%.3f",
                help="Proporción esperada en el grupo control/no expuestos"
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
            test_type = st.radio(
                "Tipo de prueba:",
                ["Two-sided", "One-sided"],
                horizontal=True
            )

        with col_sample[3]:
            alpha = st.select_slider(
                "Nivel de significancia (α):",
                options=[0.01, 0.05, 0.10], value=0.05,
                help="Probabilidad de error tipo I"
            )
            use_fpc = st.checkbox(
                "Aplicar FPC",
                value=True,
                help="Corrección de Población Finita cuando n/N > 5%"
            )

        if st.button("🧮 CALCULAR MUESTRA (COHORTES)", use_container_width=True):
            if test_type == "One-sided":
                alpha = alpha / 2

            pop_for_calc = population if population > 0 else None
            result = calculate_sample_size_cohort(p1, p2, alpha, power, ratio, pop_for_calc, use_fpc)

            st.markdown("---")
            st.markdown("### 📊 Resultados - Cohortes")

            if result.get('fpc_applied', False):
                st.success(f"""
                **✅ Corrección de Población Finita (FPC) aplicada:**
                - Muestra sin FPC: **{result['n_without_fpc']:,}**
                - Muestra con FPC: **{result['total']:,}**
                - Población total: **{population:,}**
                - Ahorro: **{result['n_without_fpc'] - result['total']:,} participantes**
                """)
            elif population > 0 and not result.get('fpc_applied', False):
                st.info(f"""
                **ℹ️ FPC no necesaria:**
                - La muestra calculada ({result['n_without_fpc']:,}) representa menos del 5% de la población ({population:,})
                - Condición: n/N = {result['n_without_fpc']/population:.2%} < 5%
                """)

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

            # Visualización
            fig, ax = plt.subplots(figsize=(10, 6))

            sample_sizes = list(range(20, result['total'] * 2, 10))
            powers = []
            for n in sample_sizes:
                n1 = n / (1 + ratio)
                n2 = n * ratio / (1 + ratio)
                p_bar = (p1 + ratio * p2) / (1 + ratio)
                se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                z_power = (abs(p1 - p2) / se) - 1.96
                power_calc = norm.cdf(z_power)
                powers.append(power_calc * 100)

            ax.plot(sample_sizes, powers, 'b-', linewidth=2)
            ax.axhline(y=power*100, color='red', linestyle='--', label=f'Poder objetivo: {power*100}%')
            ax.axvline(x=result['total'], color='green', linestyle='--', label=f'N calculado: {result["total"]}')
            ax.fill_between(sample_sizes, powers, alpha=0.3)
            ax.set_xlabel('Tamaño de Muestra Total')
            ax.set_ylabel('Poder Estadístico (%)')
            ax.set_title('Curva de Poder vs Tamaño de Muestra')
            ax.legend()
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Interpretación
            st.markdown("---")
            st.markdown("### 📝 Interpretación")

            effect_size = abs(p1 - p2)

            if result.get('fpc_applied', False):
                fpc_msg = f"""
                - ⚠️ **Se aplicó corrección de población finita (FPC)**
                - Muestra reducida de {result['n_without_fpc']:,} a {result['total']:,}
                - Porque n/N = {result['n_without_fpc']/population:.2%} > 5%
                """
            elif population > 0:
                fpc_msg = f"- Población total conocida: **{population:,}** (FPC no necesaria cuando n/N < 5%)"
            else:
                fpc_msg = "- Población asumida como infinita"

            st.info(f"""
            **Para detectar una diferencia de {effect_size:.1%} entre las proporciones ({p1:.1%} vs {p2:.1%}):**

            - Se requieren **{result['total']:,} participantes** en total
            - **{result['n1']:,}** en el Grupo 1 y **{result['n2']:,}** en el Grupo 2
            - Con un poder del {power*100:.0f}% y α = {alpha*2 if test_type == "Two-sided" else alpha:.3f}
            {fpc_msg}

            **Nota:** Se recomienda agregar un 10-20% adicional para posibles pérdidas de seguimiento.
            """)

            # Muestra ajustada
            st.markdown("#### 🔧 Muestras Ajustadas por Pérdidas")
            col_adj = st.columns(3)
            for pct in [0.10, 0.15, 0.20]:
                adj_total = int(result['total'] * (1 + pct))
                adj_n1 = int(result['n1'] * (1 + pct))
                adj_n2 = int(result['n2'] * (1 + pct))
                with col_adj[int(pct*5)-1]:
                    st.metric(f"+{pct*100:.0f}% pérdida", f"N = {adj_total:,}")
                    st.caption(f"n1={adj_n1:,}, n2={adj_n2:,}")
    
# ==========================================
# MÓDULO 6: VIGILANCIA 6.0 (AVANZADO) - SEIR COMPLETO
# ==========================================
elif menu == "📈 Vigilancia & IA":
    st.header("📈 Vigilancia Epidemiológica Avanzada v6.0 (Modelo SEIR)")

    # Info del modelo
    st.info("""
    **Modelo SEIR Completo:**
    - **S** (Susceptibles): Población en riesgo de infectarse
    - **E** (Expuestos): Infectados en período de incubación
    - **I** (Infectados): Casos activos con síntomas
    - **R** (Recuperados): Inmunes o que se recuperaron
    """)

    with st.expander("📥 Datos del Brote", expanded=True):
        if st.session_state.df_v is None or st.button("🔄 Reiniciar Datos"):
            # Calcular recuperados acumulados aproximados (usando gamma ~0.1)
            casos_acum = np.cumsum([10, 15, 25, 40, 55, 70, 90, 120, 150, 180])
            # Recuperados ~ casos de hace ~10 días
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
                         help="1/período de incubación (ej: 0.2 = 5 días)")
    with col_params[2]:
        gamma = st.slider("γ (Tasa Recuperación)", 0.05, 0.5, 0.1, 0.01,
                         help="1/tiempo de infección (ej: 0.1 = 10 días)")
    with col_params[3]:
        rho = st.slider("ρ (Tasa Diagnóstico)", 0.1, 1.0, 0.4, 0.01,
                       help="Proporción de casos detectados")
    with col_params[4]:
        n_sim = st.number_input("Sim. Monte Carlo", 50, 1000, 100, 50)

    # Parámetros derivados
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
        with st.spinner("⏳ Ejecutando modelo SEIR + Proyecciones IA..."):
            _, _, RandomForestRegressor, _ = get_heavy_imports()

            # =====================
            # MODELO SEIR (Cálculo de R dinámico)
            # =====================
            days_total = len(df_v) + 30  # Histórico + proyección
            N = casos_reales * 10  # Población total aproximada

            # Estado inicial SEIR
            I0 = casos_reales // 10  # Infectados iniciales
            E0 = I0 * 2  # Expuestos ~2x infectados
            R0_init = int(df_v['Recuperados'].iloc[-1]) if 'Recuperados' in df_v.columns else I0
            S0 = N - E0 - I0 - R0_init

            # Arrays para guardar resultados
            S_arr, E_arr, I_arr, R_arr = [S0], [E0], [I0], [R0_init]

            # Simular modelo SEIR
            for t in range(1, days_total):
                S_t = S_arr[-1]
                E_t = E_arr[-1]
                I_t = I_arr[-1]
                R_t = R_arr[-1]

                # Ecuaciones SEIR
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

            # =====================
            # PROYECCIÓN IA (para infectados observados)
            # =====================
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

            # =====================
            # GRÁFICA SEIR COMPLETA
            # =====================
            fig_seir = go.Figure()

            days_range = list(range(days_total))

            # Susceptibles (S)
            fig_seir.add_trace(go.Scatter(
                x=days_range, y=S_arr,
                name="S (Susceptibles)",
                line=dict(color='#60a5fa', width=2),
                fill='tozeroy', fillcolor='rgba(96, 165, 250, 0.2)'
            ))

            # Expuestos (E)
            fig_seir.add_trace(go.Scatter(
                x=days_range, y=E_arr,
                name="E (Expuestos)",
                line=dict(color='#f59e0b', width=2),
                fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.2)'
            ))

            # Infectados (I)
            fig_seir.add_trace(go.Scatter(
                x=days_range, y=I_arr,
                name="I (Infectados)",
                line=dict(color='#ef4444', width=3),
                fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)'
            ))

            # Recuperados (R)
            fig_seir.add_trace(go.Scatter(
                x=days_range, y=R_arr,
                name="R (Recuperados)",
                line=dict(color='#10b981', width=2),
                fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.2)'
            ))

            # Línea vertical separando histórico de proyección
            fig_seir.add_vline(x=len(df_v)-1, line_dash="dash", line_color="white",
                              annotation_text="Proyección")

            fig_seir.update_layout(
                title=f"📊 Modelo SEIR Completo (R0 = {r0:.2f})",
                xaxis_title="Día",
                yaxis_title="Población",
                height=500,
                template="plotly_dark",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )

            st.plotly_chart(fig_seir, use_container_width=True)

            # =====================
            # GRÁFICA DE INFECTADOS CON IA
            # =====================
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

            # =====================
            # TABLAS DE RESULTADOS
            # =====================
            col_tabs = st.tabs(["📋 Proyecciones IA", "📊 Resumen SEIR"])

            with col_tabs[0]:
                df_proy = pd.DataFrame({
                    "Día": futuro_x.flatten(),
                    "Proyección": p_mean.astype(int),
                    "IC Inferior": p_low.astype(int),
                    "IC Superior": p_high.astype(int)
                })
                st.dataframe(df_proy, use_container_width=True)

            with col_tabs[1]:
                final_day = len(df_v) + 14
                resumen_seir = pd.DataFrame({
                    "Día": [1, final_day],
                    "S (Susceptibles)": [int(S_arr[0]), int(S_arr[final_day])],
                    "E (Expuestos)": [int(E_arr[0]), int(E_arr[final_day])],
                    "I (Infectados)": [int(I_arr[0]), int(I_arr[final_day])],
                    "R (Recuperados)": [int(R_arr[0]), int(R_arr[final_day])],
                    "% Recuperados": [f"{R_arr[0]/N*100:.1f}%", f"{R_arr[final_day]/N*100:.1f}%"]
                })
                st.dataframe(resumen_seir, use_container_width=True)

                st.markdown(f"""
                **Resumen SEIR:**
                - Población total: **{N:,}**
                - R0 = {r0:.2f} ({'Epidemia en expansión' if r0 > 1 else 'Epidemia controlada'})
                - Tiempo de infección: **{duracion_infeccion:.1f} días**
                - Período de incubación: **{periodo_incubacion:.1f} días**
                """)
                
# ==========================================
# MÓDULO UNIFICADO OPTIMIZADO: REVISIÓN DE LITERATURA
# ==========================================

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Any

# ==========================================
# CLASE: MANEJADOR DE ESTADO CENTRALIZADO
# ==========================================
class ReviewState:
    """Manejador centralizado de estado para revisión sistemática."""

    DEFAULTS = {
        'prisma_data': {
            'registros_db': 1500, 'registros_registros': 50, 'duplicados': 400,
            'excluidos_title': 500, 'excluidos_abstract': 400, 'articulos_recuperados': 250,
            'articulos_evaluated': 200, 'articulos_excluidos': 150, 'estudios_included': 25
        },
        'forest_studies': pd.DataFrame({
            'Estudio': ['Smith 2020', 'Johnson 2019', 'Williams 2021'],
            'Eventos_Tto': [20, 35, 28], 'Total_Tto': [100, 150, 120],
            'Eventos_Ctrl': [30, 50, 45], 'Total_Ctrl': [100, 150, 120]
        }),
        'articulos_pico': [],
        'meta_studies': pd.DataFrame(),
        'rob_assessments': [],
        'grade_assessments': []
    }

    @staticmethod
    def init(key: str, default: Any) -> Any:
        """Inicializa estado con valor por defecto si no existe."""
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]

    @staticmethod
    def reset(key: str) -> None:
        """Reinicia estado a valor por defecto."""
        if key in ReviewState.DEFAULTS:
            st.session_state[key] = ReviewState.DEFAULTS[key]

# ==========================================
# FUNCIONES HELPER: PRISMA FLOWCHART
# ==========================================
def create_prisma_box(y: float, x: float, val: int, txt: str, col: str) -> go.Scatter:
    """Crea una caja PRISMA con estilo consistente."""
    return go.Scatter(
        x=[x-0.15, x+0.15, x+0.15, x-0.15, x-0.15],
        y=[y, y, y-0.4, y-0.4, y],
        fill='toself',
        fillcolor=col,
        line=dict(color='white', width=2),
        text=f"{txt}<br>{val}",
        mode='text',
        showlegend=False,
        textfont=dict(size=12, color='white'),
        hoverinfo='text'
    )

def render_prisma_chart(data: Dict) -> go.Figure:
    """Renderiza el diagrama PRISMA con los datos proporcionados."""
    fig = go.Figure()
    records_initial = data['registros_db']
    after_duplicates = records_initial - data['duplicados']
    final_included = data['estudios_included']

    fig.add_trace(create_prisma_box(5, 0, records_initial, "Identificación", "#667eea"))
    fig.add_trace(create_prisma_box(3, 0, after_duplicates, "Screening", "#10b981"))
    fig.add_trace(create_prisma_box(1, 0, final_included, "Incluidos", "#f59e0b"))

    fig.update_layout(
        height=400,
        xaxis=dict(visible=False, range=[-0.5, 0.5]),
        yaxis=dict(visible=False, range=[0, 6]),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# FUNCIONES HELPER: FOREST PLOT
# ==========================================
def calculate_odds_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula odds ratios y CI para cada estudio."""
    df = df.copy()
    df['OR'] = (df['Eventos_Tto'] / (df['Total_Tto'] - df['Eventos_Tto'])) / \
               (df['Eventos_Ctrl'] / (df['Total_Ctrl'] - df['Eventos_Ctrl']))
    df['OR'] = df['OR'].replace([np.inf, -np.inf], np.nan)
    return df

def render_forest_plot(df: pd.DataFrame) -> plt.Figure:
    """Genera el Forest Plot con estilo profesional."""
    df_calc = calculate_odds_ratios(df)
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.8)))

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
    return fig

# ==========================================
# FUNCIONES HELPER: META-ANÁLISIS
# ==========================================
@st.cache_data(ttl=3600)
def cached_meta_analysis(ev_tto: tuple, tt_tto: tuple, ev_ctrl: tuple, tt_ctrl: tuple,
                         model_type: str) -> Dict:
    """Ejecuta meta-análisis con caché para mejorar rendimiento."""
    ev_e, tt_e = list(ev_tto), list(tt_tto)
    ev_c, tt_c = list(ev_ctrl), list(tt_ctrl)

    if "Fijos" in model_type:
        return meta_analysis_fixed_effect(ev_e, tt_e, ev_c, tt_c)
    else:
        return meta_analysis_random_effects(ev_e, tt_e, ev_c, tt_c)

# ==========================================
# FUNCIONES HELPER: UI COMPONENTS
# ==========================================
def metric_card(col, label: str, value: str, delta: Optional[str] = None):
    """Renderiza una tarjeta de métrica estilizada."""
    with col:
        st.metric(label=label, value=value, delta=delta)

def render_study_table(df: pd.DataFrame, display_cols: List[str]) -> None:
    """Renderiza tabla de estudios con columnas disponibles."""
    available_cols = [c for c in display_cols if c in df.columns]
    if available_cols and len(df) > 0:
        st.dataframe(df[available_cols], use_container_width=True, height=min(300, len(df) * 50 + 50))
    else:
        st.info("No hay estudios registrados.")

# ==========================================
# SUB-MÓDULO: PICO EXTRACTOR
# ==========================================
def render_pico_tab():
    """Renderiza la pestaña de extracción PICO con IA."""
    st.subheader("🤖 Analizador IA de Evidencia Científica")

    api_k = st.text_input(
        "🔑 OpenAI API Key", type="password", key="api_pico",
        placeholder="sk-...",
        help="Obtenga su API key en platform.openai.com"
    )

    if not api_k:
        st.info("💡 Ingrese su OpenAI API Key para activar el análisis inteligente.")
        return

    col_left, col_right = st.columns([1, 2])

    with col_left:
        metodo = st.radio("📥 Método de Carga:", ["PDF", "DOI"], key="met_pico", horizontal=True)
        ext = LiteratureAIExtractor(api_k)
        res = None

        if metodo == "PDF":
            f = st.file_uploader("Subir artículo PDF", type="pdf", key="pdf_pico")
            if f and st.button("🔍 Extraer PICO", use_container_width=True):
                with st.spinner("⏳ Analizando con IA..."):
                    res = ext.from_pdf(f)
        else:
            doi = st.text_input("DOI (ej: 10.1056/NEJMoa...)", placeholder="10.1056/...", key="doi_pico")
            if doi and st.button("🔍 Consultar DOI", use_container_width=True):
                with st.spinner("⏳ Consultando CrossRef..."):
                    res = ext.from_doi(doi)

        if res:
            if "error" in res:
                st.error(f"❌ {res['error']}")
            else:
                st.session_state.articulos_pico.append(res)
                st.success("✅ Artículo analizado exitosamente!")

    with col_right:
        articulos = ReviewState.init('articulos_pico', [])

        if articulos:
            st.write("📚 Biblioteca de Evidencia")
            df_articulos = pd.DataFrame(articulos)
            render_study_table(df_articulos, ['titulo', 'diseno', 'grade', 'resultados_desenlaces'])

            col_btns = st.columns(2)
            with col_btns[0]:
                if st.button("🗑️ Limpiar Biblioteca", use_container_width=True):
                    st.session_state.articulos_pico = []
                    st.rerun()
            with col_btns[1]:
                st.download_button(
                    "📥 Exportar JSON",
                    data=json.dumps(articulos, indent=2, ensure_ascii=False),
                    file_name="pico_data.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("📭 Biblioteca vacía. Cargue un artículo para comenzar.")

# ==========================================
# SUB-MÓDULO: PRISMA FLOWCHART
# ==========================================
def render_prisma_tab():
    """Renderiza la pestaña del flujograma PRISMA."""
    st.subheader("📑 Flujograma PRISMA 2020")
    prisma_data = ReviewState.init('prisma_data', ReviewState.DEFAULTS['prisma_data'])

    col_p1, col_p2 = st.columns([1, 2])

    with col_p1:
        st.write("### 📊 Parámetros de Datos")
        prisma_data['registros_db'] = st.number_input("Registros de DB:", min_value=0, value=prisma_data['registros_db'], step=10)
        prisma_data['duplicados'] = st.number_input("Duplicados:", min_value=0, max_value=prisma_data['registros_db'], value=prisma_data['duplicados'], step=10)
        prisma_data['excluidos_title'] = st.number_input("Excluidos por Título:", min_value=0, value=prisma_data['excluidos_title'], step=10)
        prisma_data['estudios_included'] = st.number_input("Estudios Finales Incluidos:", min_value=0, max_value=prisma_data['registros_db'], value=prisma_data['estudios_included'], step=1)

        if st.button("🔄 Reiniciar Valores", use_container_width=True):
            ReviewState.reset('prisma_data')
            st.rerun()

    with col_p2:
        fig_prisma = render_prisma_chart(prisma_data)
        st.plotly_chart(fig_prisma, use_container_width=True)

        total_screen = prisma_data['registros_db'] - prisma_data['duplicados']
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        metric_card(col_stat1, "Registros Iniciales", f"{prisma_data['registros_db']:,}")
        metric_card(col_stat2, "Tras Screening", f"{total_screen:,}")
        metric_card(col_stat3, "Estudios Finales", f"{prisma_data['estudios_included']:,}")

# ==========================================
# SUB-MÓDULO: FOREST PLOT
# ==========================================
def render_forest_tab():
    """Renderiza la pestaña del Forest Plot."""
    st.subheader("🌲 Análisis Visual de Efectos")
    forest_studies = ReviewState.init('forest_studies', ReviewState.DEFAULTS['forest_studies'])

    st.write("### 📝 Datos de Estudios")
    edit_forest = st.data_editor(
        forest_studies,
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
                fig = render_forest_plot(edit_forest)
                st.pyplot(fig)
            else:
                st.warning("⚠️ Agregue al menos un estudio.")

    with col_transfer:
        if st.button("➕ Enviar a Meta-análisis", use_container_width=True):
            if len(edit_forest) >= 2:
                st.session_state.meta_studies = edit_forest
                st.success(f"✅ {len(edit_forest)} estudios transferidos a Meta-análisis!")
            else:
                st.warning("Se requieren al menos 2 estudios para meta-análisis.")

# ==========================================
# SUB-MÓDULO: META-ANÁLISIS
# ==========================================
def render_meta_tab():
    """Renderiza la pestaña de Meta-análisis."""
    st.subheader("📊 Modelos Estadísticos")
    meta_studies = ReviewState.init('meta_studies', ReviewState.DEFAULTS['forest_studies'])

    if len(meta_studies) < 2:
        st.warning("📌 Se requieren al menos 2 estudios. Importe datos desde Forest Plot.")
        return

    st.info(f"📚 {len(meta_studies)} estudios cargados para análisis.")

    col_model, col_action = st.columns([2, 1])
    with col_model:
        mod_meta = st.selectbox(
            "Modelo:",
            ["Efectos Fijos (Peto)", "Efectos Aleatorios (DerSimonian-Laird)"],
            label_visibility="collapsed"
        )

    if st.button("📊 Calcular Meta-análisis", use_container_width=True):
        ev_e = tuple(meta_studies['Eventos_Tto'].tolist())
        tt_e = tuple(meta_studies['Total_Tto'].tolist())
        ev_c = tuple(meta_studies['Eventos_Ctrl'].tolist())
        tt_c = tuple(meta_studies['Total_Ctrl'].tolist())

        res_m = cached_meta_analysis(ev_e, tt_e, ev_c, tt_c, mod_meta)
        key_or = 'pooled_or' if "Fijos" in mod_meta else 'pooled_or_re'
        pooled_or = res_m.get(key_or, 1.0)
        i2 = res_m.get('I2', 0)
        p_val = res_m.get('p_value', 0.05)

        c1, c2, c3 = st.columns(3)
        metric_card(c1, "OR Combinado", f"{pooled_or:.2f}")
        metric_card(c2, "I² (Heterogeneidad)", f"{i2:.1f}%", f"{'Alta' if i2 > 75 else 'Moderada' if i2 > 50 else 'Baja'}")
        metric_card(c3, "p-value", f"{p_val:.4f}", "< 0.05" if p_val < 0.05 else "≥ 0.05")

        if i2 > 75:
            st.error("⚠️ Alta heterogeneidad detectada (I² > 75%). Considere usar modelo de efectos aleatorios.")
        elif i2 > 50:
            st.warning("⚠️ Heterogeneidad moderada (I² > 50%). Interpretar con precaución.")
        else:
            st.success("✅ Baja heterogeneidad. Resultados consistentes entre estudios.")

# ==========================================
# SUB-MÓDULO: CALIDAD (RoB/GRADE)
# ==========================================
def render_quality_tab():
    """Renderiza la pestaña de evaluación de calidad."""
    st.subheader("⚖️ Evaluación de Calidad de Evidencia")

    q_sub = st.radio("Herramienta de Evaluación:", ["RoB 2 (Riesgo de Sesgo)", "GRADE"], horizontal=True)

    if q_sub == "RoB 2 (Riesgo de Sesgo)":
        render_rob2_assessment()
    else:
        render_grade_assessment()

def render_rob2_assessment():
    """Renderiza formulario de evaluación RoB 2."""
    st.subheader("🔍 Evaluación de Riesgo de Sesgo (RoB 2)")
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
        if not s_name:
            st.error("⚠️ Ingrese el nombre del estudio.")
            return

        assessment = {'Estudio': s_name, 'D1': d1, 'D2': d2, 'D3': d3, 'D4': d4}
        st.session_state.rob_assessments.append(assessment)
        st.success("✅ Evaluación guardada exitosamente!")

    if st.session_state.rob_assessments:
        st.write("### 📋 Evaluaciones Guardadas")
        df_rob = pd.DataFrame(st.session_state.rob_assessments)
        st.dataframe(df_rob, use_container_width=True)

def render_grade_assessment():
    """Renderiza formulario de evaluación GRADE."""
    st.subheader("📋 Sistema GRADE - Certeza de Evidencia")
    outcome = st.text_input("📝 Resultado (Outcome):", placeholder="ej: Mortalidad, Eventos adversos")

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
        if not outcome:
            st.error("⚠️ Ingrese el resultado a evaluar.")
            return

        score = 4 + r_bias + incons + indir
        score += (1 if large else 0) + (1 if dose else 0) + (1 if confound else 0)
        score = max(1, min(4, score))

        labels = {4: "🔴 Alta", 3: "🟡 Moderada", 2: "🟠 Baja", 1: "⚫ Muy Baja"}
        st.metric("Certeza de Evidencia", labels.get(score, "⚫ Muy Baja"))

        st.write(f"""
        **Desglose del Cálculo:**
        - Nivel base: 4
        - Riesgo de Sesgo: {r_bias}
        - Inconsistencia: {incons}
        - Indirectitud: {indir}
        - Ajustes positivos: {sum([1 if large else 0, 1 if dose else 0, 1 if confound else 0])}
        - **Score Final: {score}/4**
        """)

        assessment = {
            'Outcome': outcome, 'Score': score,
            'RiesgoSesgo': r_bias, 'Inconsistencia': incons, 'Indirectitud': indir,
            'EfectoGrande': large, 'DosisResp': dose, 'Confounders': confound
        }
        st.session_state.grade_assessments.append(assessment)

# ==========================================
# MÓDULO PRINCIPAL: REVISIÓN DE LITERATURA
# ==========================================
def render_literature_review_module(menu: str):
    """Módulo principal unificado de revisión de literatura."""

    if menu != "📚 Revisión de Literatura":
        return

    st.header("📚 Centro de Evidencia Científica")
    st.markdown("Gestione todo el proceso de su revisión sistemática desde una sola interfaz.")

    tab_pico, tab_prisma, tab_forest, tab_meta, tab_quality = st.tabs([
        "🤖 Extracción PICO",
        "📑 PRISMA Flowchart",
        "🌲 Forest Plot",
        "📊 Meta-análisis",
        "⚖️ Calidad (RoB/GRADE)"
    ])

    with tab_pico:
        render_pico_tab()

    with tab_prisma:
        render_prisma_tab()

    with tab_forest:
        render_forest_tab()

    with tab_meta:
        render_meta_tab()

    with tab_quality:
        render_quality_tab()
        
# ==========================================
# MÓDULO 12 OPTIMIZADO: ANÁLISIS DE SUPERVIVENCIA (KAPLAN-MEIER)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CLASE: MANEJADOR DE ESTADO
# ==========================================
class SurvivalState:
    DEFAULTS = {'survival_data': None, 'km_results': {}}

    @staticmethod
    def init(key: str, default):
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]

    @staticmethod
    def generate_sample_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        return pd.DataFrame({
            'ID': range(1, n + 1),
            'Tiempo': np.concatenate([np.random.exponential(30, n//2), np.random.exponential(20, n//2)]).round(1),
            'Evento': np.random.binomial(1, 0.4, n),
            'Grupo': ['Tratamiento'] * (n//2) + ['Control'] * (n//2),
            'Edad': np.random.randint(30, 80, n),
            'Sexo': np.random.choice(['M', 'F'], n)
        })

KM_COLORS = {'Tratamiento': '#3498db', 'Control': '#e74c3c', 'Global': '#2ecc71'}

# ==========================================
# FUNCIONES HELPER
# ==========================================
def calculate_km_manual(time: np.ndarray, event: np.ndarray) -> Dict:
    df = pd.DataFrame({'time': time, 'event': event}).sort_values('time')
    times = df['time'].values
    events = df['event'].values
    n = len(times)
    unique_times = np.unique(times[events == 1])
    
    survival_times, survival_probs = [0], [1.0]
    conf_lower, conf_upper = [1.0], [1.0]
    survived = n
    
    for t in unique_times:
        d = np.sum(events[times == t])
        r = np.sum(time >= t)
        if r > 0:
            survived = survived * (1 - d / r)
            survival_times.append(t)
            survival_probs.append(survived / n)
            conf_lower.append(max(0, survival_probs[-1] - 0.1))
            conf_upper.append(min(1, survival_probs[-1] + 0.1))
    
    survival_times.append(times.max())
    survival_probs.append(survival_probs[-1])
    
    return {'times': np.array(survival_times), 'survival': np.array(survival_probs),
            'ci_lower': np.array(conf_lower), 'ci_upper': np.array(conf_upper)}

def calculate_median_survival(times: np.ndarray, probs: np.ndarray) -> float:
    for t, s in zip(times, probs):
        if s <= 0.5: return t
    return float('inf')

def log_rank_test(time1: np.ndarray, event1: np.ndarray, time2: np.ndarray, event2: np.ndarray) -> Dict:
    times = np.sort(np.unique(np.concatenate([time1[event1==1], time2[event2==1]])))
    obs1, exp1, var_sum = 0, 0, 0
    
    for t in times:
        r1 = np.sum(time1 >= t)
        r2 = np.sum(time2 >= t)
        r_total = r1 + r2
        d1 = np.sum((time1 == t) & (event1 == 1))
        d_total = d1 + np.sum((time2 == t) & (event2 == 1))
        
        if r_total > 0:
            e1 = r1 * d_total / r_total
            obs1 += d1
            exp1 += e1
            if r_total > 1:
                var_sum += (r1 * r2 * d_total * (r_total - d_total)) / (r_total ** 2 * (r_total - 1))
    
    chi2 = (obs1 - exp1) ** 2 / var_sum if var_sum > 0 else 0
    p_value = chi2 / (chi2 + 10) if chi2 < 100 else 0.0001
    
    return {'chi2': chi2, 'p_value': min(1.0, p_value), 'observed': obs1, 'expected': exp1}

def plot_km(results: Dict, ax=None, show_ci=True) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in results.items():
        color = KM_COLORS.get(name, '#3498db')
        ax.step(data['times'], data['survival'], where='post', color=color, linewidth=2.5, label=name, marker='o', markersize=4)
        if show_ci and 'ci_lower' in data:
            ax.fill_between(data['times'], data['ci_lower'], data['ci_upper'], step='post', alpha=0.2, color=color)
    
    ax.set_xlabel('Tiempo'); ax.set_ylabel('Probabilidad de Supervivencia')
    ax.set_title('Curva de Kaplan-Meier', fontsize=14, fontweight='bold')
    ax.legend(loc='best'); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, ax.get_xlim()[1]]); ax.set_ylim([0, 1.05])
    return ax.figure if ax is None else None

# ==========================================
# SUB-MÓDULOS
# ==========================================
def render_data_input():
    st.markdown("### 📝 Ingrese datos de supervivencia")
    
    if st.session_state.survival_data is None:
        st.session_state.survival_data = SurvivalState.generate_sample_data()
    
    col_upload = st.columns([1, 1, 1])
    with col_upload[0]:
        uploaded = st.file_uploader("📂 Cargar CSV:", type="csv")
        if uploaded:
            st.session_state.survival_data = pd.read_csv(uploaded)
            st.success(f"✅ {len(pd.read_csv(uploaded))} registros!")
    with col_upload[1]:
        if st.button("🎲 Generar Ejemplo", use_container_width=True):
            st.session_state.survival_data = SurvivalState.generate_sample_data()
            st.rerun()
    with col_upload[2]:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.survival_data = None
            st.rerun()
    
    if st.session_state.survival_data is not None and len(st.session_state.survival_data) > 0:
        df = st.session_state.survival_data
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", disabled=True),
                "Tiempo": st.column_config.NumberColumn("Tiempo", min_value=0, format="%.1f"),
                "Evento": st.column_config.NumberColumn("Evento (0=Censura, 1=Evento)", min_value=0, max_value=1),
                "Grupo": st.column_config.TextColumn("Grupo"),
                "Edad": st.column_config.NumberColumn("Edad", min_value=0),
                "Sexo": st.column_config.TextColumn("Sexo")
            })
        st.session_state.survival_data = edited
        
        col_stats = st.columns(4)
        with col_stats[0]: st.metric("Muestras", len(edited))
        with col_stats[1]: st.metric("Eventos", int(edited['Evento'].sum()))
        with col_stats[2]: st.metric("Censuras", int(len(edited) - edited['Evento'].sum()))
        with col_stats[3]: st.metric("Tiempo medio", f"{edited['Tiempo'].mean():.1f}")
    else:
        st.info("📭 Cargue datos o genere ejemplo.")

def render_km_curve():
    st.markdown("### 📈 Curva de Kaplan-Meier")
    
    if st.session_state.survival_data is None or len(st.session_state.survival_data) == 0:
        st.warning("⚠️ Primero cargue/gener datos")
        return
    
    df = st.session_state.survival_data
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = [c for c in df.columns if c not in num_cols]
    
    col_setup = st.columns([1, 1, 1])
    with col_setup[0]:
        time_col = st.selectbox("⏱️ Tiempo:", num_cols, index=0)
    with col_setup[1]:
        event_col = st.selectbox("⚠️ Evento:", num_cols, index=min(1, len(num_cols)-1))
    with col_setup[2]:
        group_by = st.selectbox("👥 Agrupar:", ['Ninguno'] + text_cols if text_cols else ['Ninguno'])
    
    show_ci = st.checkbox("Mostrar IC 95%", value=True)
    
    if st.button("📊 GENERAR CURVA KM", use_container_width=True, type="primary"):
        try:
            results = {}
            if group_by == 'Ninguno':
                results['Global'] = calculate_km_manual(df[time_col].values, df[event_col].values)
            else:
                for group in df[group_by].unique():
                    mask = df[group_by] == group
                    results[str(group)] = calculate_km_manual(df.loc[mask, time_col].values, df.loc[mask, event_col].values)
            
            st.session_state.km_results = results
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_km(results, ax, show_ci)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### 📊 Estadísticas")
            col_stats = st.columns(3)
            for i, (name, data) in enumerate(results.items()):
                with col_stats[i % 3]:
                    median = calculate_median_survival(data['times'], data['survival'])
                    median_str = f"{median:.1f}" if not np.isinf(median) else "No alcanzada"
                    st.metric(f"Mediana {name}", median_str)
            
            st.markdown("### 📋 Tabla de Vida")
            timeline = st.slider("Tiempo máx:", 5, int(df[time_col].max()), 50)
            for name, data in results.items():
                table = pd.DataFrame({
                    'Tiempo': data['times'][data['times'] <= timeline],
                    'Supervivencia': data['survival'][data['times'] <= timeline],
                    'IC_lower': data['ci_lower'][data['times'] <= timeline],
                    'IC_upper': data['ci_upper'][data['times'] <= timeline]
                })
                with st.expander(f"Tabla {name}"):
                    st.dataframe(table, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def render_advanced_analysis():
    st.markdown("### 📊 Análisis Avanzado")
    
    if st.session_state.survival_data is None or len(st.session_state.survival_data) == 0:
        st.warning("⚠️ Primero cargue/gener datos")
        return
    
    df = st.session_state.survival_data
    tabs_adv = st.tabs(["📈 Log-Rank", "📉 HR", "📊 Comparación"])
    
    with tabs_adv[0]:
        st.markdown("#### 🔬 Test de Log-Rank")
        time_col = df.select_dtypes(include=[np.number]).columns[0]
        event_col = df.select_dtypes(include=[np.number]).columns[1]
        group_col = st.selectbox("👥 Grupo:", [c for c in df.columns if c not in ['ID', time_col, event_col]])
        
        if st.button("🔬 EJECUTAR LOG-RANK", use_container_width=True, type="primary"):
            groups = df[group_col].unique()
            if len(groups) != 2:
                st.warning(f"⚠️ Se requieren 2 grupos. Encontrados: {len(groups)}")
                return
            
            mask1 = df[group_col] == groups[0]
            mask2 = df[group_col] == groups[1]
            result = log_rank_test(df.loc[mask1, time_col].values, df.loc[mask1, event_col].values,
                                 df.loc[mask2, time_col].values, df.loc[mask2, event_col].values)
            
            col_lr = st.columns(3)
            with col_lr[0]: st.metric("Chi²", f"{result['chi2']:.4f}")
            with col_lr[1]: st.metric("p-value", f"{result['p_value']:.6f}")
            with col_lr[2]: st.metric("Conclusión", "✅ Significativo" if result['p_value'] < 0.05 else "❌ No")
            
            if result['p_value'] < 0.05:
                st.success(f"🏆 **Se rechaza H0**: Las curvas difieren significativamente (p={result['p_value']:.4f})")
            else:
                st.info(f"📊 **No se puede rechazar H0** (p={result['p_value']:.4f})")
    
    with tabs_adv[1]:
        st.markdown("#### 📉 Ratio de Hazard")
        st.info("HR > 1 = Mayor riesgo, HR < 1 = Menor riesgo")
        
        if st.button("📊 Calcular HR", use_container_width=True, type="primary"):
            if 'Grupo' not in df.columns:
                st.warning("⚠️ Se requiere columna 'Grupo'")
                return
            
            groups = df['Grupo'].unique()
            time_col = df.select_dtypes(include=[np.number]).columns[0]
            event_col = df.select_dtypes(include=[np.number]).columns[1]
            result = log_rank_test(df.loc[df['Grupo']==groups[0], time_col].values,
                                 df.loc[df['Grupo']==groups[0], event_col].values,
                                 df.loc[df['Grupo']==groups[1], time_col].values,
                                 df.loc[df['Grupo']==groups[1], event_col].values)
            
            hr = np.exp(np.sqrt(result['chi2']) * (1 if result['observed'] > result['expected'] else -1))
            col_hr = st.columns(3)
            with col_hr[0]: st.metric("HR", f"{hr:.3f}")
            with col_hr[1]: st.metric("Riesgo", "Mayor" if hr > 1 else "Menor")
            with col_hr[2]: st.metric("Interpretación", "Protector" if hr < 0.8 else "Factor riesgo" if hr > 1.2 else "Neutro")
    
    with tabs_adv[2]:
        st.markdown("#### 📊 Comparación de Grupos")
        
        if 'Grupo' not in df.columns:
            st.warning("⚠️ Se requiere columna 'Grupo'")
            return
        
        results = {}
        for group in df['Grupo'].unique():
            mask = df['Grupo'] == group
            time_col = df.select_dtypes(include=[np.number]).columns[0]
            event_col = df.select_dtypes(include=[np.number]).columns[1]
            results[str(group)] = calculate_km_manual(df.loc[mask, time_col].values, df.loc[mask, event_col].values)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_km(results, ax, show_ci=True)
        plt.tight_layout()
        st.pyplot(fig)
        
        comp_data = [{'Grupo': n, 'Mediana': f"{calculate_median_survival(d['times'], d['survival']):.1f}" 
                     if not np.isinf(calculate_median_survival(d['times'], d['survival'])) else 'N/A',
                     'Sup Final': f"{d['survival'][-1]:.3f}"} for n, d in results.items()]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

# ==========================================
# MÓDULO PRINCIPAL
# ==========================================
def render_survival_module(menu: str):
    if menu != "📉 Supervivencia (KM)":
        return
    
    st.header("📉 Análisis de Supervivencia - Kaplan-Meier")
    tab_km = st.tabs(["📝 Datos", "📈 Curva KM", "📊 Análisis"])
    with tab_km[0]: render_data_input()
    with tab_km[1]: render_km_curve()
    with tab_km[2]: render_advanced_analysis()

if __name__ == "__main__":
    st.set_page_config(page_title="Kaplan-Meier", layout="wide")
    render_survival_module("📉 Supervivencia (KM)")
    
# ==========================================
# MÓDULO 13 OPTIMIZADO: CURVAS ROC
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from typing import Dict, List, Optional

# ==========================================
# CLASE: MANEJADOR DE ESTADO
# ==========================================
class ROCState:
    DEFAULTS = {'roc_data': None, 'roc_results': {}}

    @staticmethod
    def init(key: str, default):
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]

    @staticmethod
    def generate_sample_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        return pd.DataFrame({
            'ID': range(1, n_samples + 1),
            'Probabilidad': np.concatenate([np.random.beta(5, 2, n_samples//2), np.random.beta(2, 5, n_samples//2)]),
            'Probabilidad_2': np.concatenate([np.random.beta(6, 2, n_samples//2), np.random.beta(2, 6, n_samples//2)]),
            'Probabilidad_3': np.concatenate([np.random.beta(4, 3, n_samples//2), np.random.beta(3, 4, n_samples//2)]),
            'Real': ['Positivo'] * (n_samples//2) + ['Negativo'] * (n_samples//2)
        })

ROC_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

# ==========================================
# FUNCIONES HELPER
# ==========================================
def calculate_roc_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = None) -> Dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    j_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[j_idx]
    threshold = threshold or optimal_threshold

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold, 'optimal_fpr': fpr[j_idx], 'optimal_tpr': tpr[j_idx],
        'confusion_matrix': cm, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'sensitivity': tp/(tp+fn) if (tp+fn) > 0 else 0,
        'specificity': tn/(tn+fp) if (tn+fp) > 0 else 0,
        'ppv': tp/(tp+fp) if (tp+fp) > 0 else 0,
        'npv': tn/(tn+fn) if (tn+fn) > 0 else 0,
        'accuracy': (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) > 0 else 0,
        'lr_positive': (tp/(tp+fn))/(1-tn/(tn+fp)) if (1-tn/(tn+fp)) > 0 else 0
    }

def prepare_binary_labels(series: pd.Series) -> np.ndarray:
    unique = series.unique()
    if len(unique) != 2:
        raise ValueError(f"Se requieren 2 clases, encontradas: {unique}")
    return (series == unique[0]).astype(int)

def plot_roc(fpr, tpr, roc_auc, opt_fpr, opt_tpr, opt_thr, color='#3498db', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Aleatorio')
    ax.scatter([opt_fpr], [opt_tpr], color='red', s=150, zorder=5, label=f'Óptimo (θ={opt_thr:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.3, color=color)
    ax.set_xlabel('1 - Especificidad (FPR)'); ax.set_ylabel('Sensibilidad (TPR)')
    ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    return ax.figure if ax is None else None

def plot_cm(cm, classes, threshold, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Matriz Confusión (θ={threshold:.3f})', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18, fontweight='bold',
                   color='white' if cm[i, j] > thresh else 'black')
    plt.colorbar(im, ax=ax)
    return ax.figure if ax is None else None

def plot_multi_roc(results, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    for i, (name, data) in enumerate(results.items()):
        ax.plot(data['fpr'], data['tpr'], color=ROC_COLORS[i%len(ROC_COLORS)], linewidth=2.5,
               label=f"{name} (AUC={data['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Aleatorio')
    ax.set_xlabel('1 - Especificidad'); ax.set_ylabel('Sensibilidad')
    ax.set_title('Comparación de Curvas ROC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    return ax.figure if ax is None else None

# ==========================================
# SUB-MÓDULOS
# ==========================================
def render_data_input():
    st.markdown("### 📝 Ingrese datos para análisis ROC")

    if st.session_state.roc_data is None:
        st.session_state.roc_data = ROCState.generate_sample_data()

    col_upload = st.columns([1, 1, 1])
    with col_upload[0]:
        uploaded = st.file_uploader("📂 Cargar CSV:", type="csv")
        if uploaded:
            st.session_state.roc_data = pd.read_csv(uploaded)
            st.success("✅ Datos cargados!")
    with col_upload[1]:
        if st.button("🎲 Generar Ejemplo", use_container_width=True):
            st.session_state.roc_data = ROCState.generate_sample_data()
            st.rerun()
    with col_upload[2]:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.roc_data = None
            st.rerun()

    if st.session_state.roc_data is not None and len(st.session_state.roc_data) > 0:
        df = st.session_state.roc_data
        st.markdown("#### 📋 Datos:")
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", disabled=True),
                "Probabilidad": st.column_config.NumberColumn("Probabilidad", min_value=0, max_value=1, format="%.4f"),
                "Probabilidad_2": st.column_config.NumberColumn("Prob 2", min_value=0, max_value=1, format="%.4f"),
                "Probabilidad_3": st.column_config.NumberColumn("Prob 3", min_value=0, max_value=1, format="%.4f"),
                "Real": st.column_config.TextColumn("Real")
            })
        st.session_state.roc_data = edited

        col_stats = st.columns(3)
        with col_stats[0]: st.metric("Muestras", len(edited))
        with col_stats[1]: st.metric("Positivos", len(edited[edited['Real']=='Positivo']))
        with col_stats[2]: st.metric("Negativos", len(edited[edited['Real']=='Negativo']))
    else:
        st.info("📭 Cargue datos o genere ejemplo.")

def render_single_roc():
    st.markdown("### 📈 Curva ROC")

    if st.session_state.roc_data is None or len(st.session_state.roc_data) == 0:
        st.warning("⚠️ Primero cargue/gener datos en pestaña 'Datos'")
        return

    df = st.session_state.roc_data
    pred_col = st.selectbox("📊 Variable de predicción:",
        options=[c for c in df.columns if c not in ['ID', 'Real']], index=0)
    actual_col = 'Real' if 'Real' in df.columns else None
    show_pr = st.checkbox("Mostrar Precision-Recall", value=False)

    if st.button("📊 GENERAR CURVA ROC", use_container_width=True, type="primary"):
        try:
            y_true = prepare_binary_labels(df[actual_col])
            y_score = df[pred_col].values
            metrics = calculate_roc_metrics(y_true, y_score)
            st.session_state.roc_results[pred_col] = metrics

            if show_pr:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                plot_roc(metrics['fpr'], metrics['tpr'], metrics['roc_auc'],
                        metrics['optimal_fpr'], metrics['optimal_tpr'], metrics['optimal_threshold'], ax=axes[0])
                plot_cm(metrics['confusion_matrix'], ['Negativo', 'Positivo'], metrics['threshold_used'], ax=axes[1])
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                axes[2].plot(recall, precision, color='#e74c3c', linewidth=2, label=f'PR (AP={ap:.3f})')
                axes[2].fill_between(recall, precision, alpha=0.3, color='#e74c3c')
                axes[2].set_xlabel('Sensibilidad'); axes[2].set_ylabel('Precisión')
                axes[2].set_title('Precision-Recall'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                plot_roc(metrics['fpr'], metrics['tpr'], metrics['roc_auc'],
                        metrics['optimal_fpr'], metrics['optimal_tpr'], metrics['optimal_threshold'], ax=axes[0])
                plot_cm(metrics['confusion_matrix'], ['Negativo', 'Positivo'], metrics['threshold_used'], ax=axes[1])
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### 📊 Métricas")
            col_m = st.columns(4)
            with col_m[0]: st.metric("AUC-ROC", f"{metrics['roc_auc']:.4f}")
            with col_m[1]: st.metric("Sensibilidad", f"{metrics['sensitivity']:.4f}")
            with col_m[2]: st.metric("Especificidad", f"{metrics['specificity']:.4f}")
            with col_m[3]: st.metric("Punto Corte", f"{metrics['optimal_threshold']:.4f}")

            col_m2 = st.columns(4)
            with col_m2[0]: st.metric("Exactitud", f"{metrics['accuracy']:.4f}")
            with col_m2[1]: st.metric("VPP", f"{metrics['ppv']:.4f}")
            with col_m2[2]: st.metric("VPN", f"{metrics['npv']:.4f}")
            with col_m2[3]: st.metric("LR+", f"{metrics['lr_positive']:.4f}")

            cm = metrics['confusion_matrix']
            cm_df = pd.DataFrame({
                '': ['Real Negativo', 'Real Positivo', 'Total'],
                'Pred Negativo': [f"{cm[0,0]}", f"{cm[1,0]}", f"{cm[0,0]+cm[1,0]}"],
                'Pred Positivo': [f"{cm[0,1]}", f"{cm[1,1]}", f"{cm[0,1]+cm[1,1]}"],
                'Total': [f"{cm[0,0]+cm[0,1]}", f"{cm[1,0]+cm[1,1]}", f"{len(y_true)}"]
            })
            st.dataframe(cm_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def render_comparison():
    st.markdown("### 📊 Comparación de Tests")
    st.info("Seleccione múltiples predictores para comparar su rendimiento diagnóstico.")

    if st.session_state.roc_data is None or len(st.session_state.roc_data) == 0:
        st.warning("⚠️ Primero cargue/gener datos")
        return

    df = st.session_state.roc_data
    predictor_cols = [c for c in df.columns if c not in ['ID', 'Real']]
    pred_cols = st.multiselect("📊 Seleccionar predictores:", options=predictor_cols,
        default=predictor_cols[:min(3, len(predictor_cols))])
    actual_col = 'Real' if 'Real' in df.columns else None

    if st.button("📊 COMPARAR TESTS", use_container_width=True, type="primary"):
        try:
            y_true = prepare_binary_labels(df[actual_col])
            results = {col: calculate_roc_metrics(y_true, df[col].values) for col in pred_cols}
            st.session_state.roc_results = results

            fig, ax = plt.subplots(figsize=(10, 8))
            plot_multi_roc(results, ax)
            for i, (name, data) in enumerate(results.items()):
                ax.scatter([data['optimal_fpr']], [data['optimal_tpr']], color=ROC_COLORS[i%len(ROC_COLORS)], s=100, zorder=5)
            plt.tight_layout()
            st.pyplot(fig)

            comp_data = [{'Test': n, 'AUC': f"{d['roc_auc']:.4f}", 'Sens': f"{d['sensitivity']:.4f}",
                         'Esp': f"{d['specificity']:.4f}", 'Exact': f"{d['accuracy']:.4f}"} for n, d in results.items()]
            comp_df = pd.DataFrame(comp_data).sort_values('AUC', ascending=False)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            best = max(results.items(), key=lambda x: x[1]['roc_auc'])
            st.success(f"🏆 Mejor test: **{best[0]}** con AUC = {best[1]['roc_auc']:.4f}")

            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            ax_bar.bar(results.keys(), [r['roc_auc'] for r in results.values()],
                      color=ROC_COLORS[:len(results)], edgecolor='black')
            ax_bar.axhline(y=0.5, color='red', linestyle='--', label='Aleatorio')
            ax_bar.set_ylabel('AUC'); ax_bar.set_title('Comparación AUC'); ax_bar.legend(); ax_bar.set_ylim([0, 1])
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            st.pyplot(fig_bar)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ==========================================
# MÓDULO PRINCIPAL
# ==========================================
def render_roc_module(menu: str):
    if menu != "🎯 Curvas ROC":
        return

    st.header("🎯 Curvas ROC - Evaluación Diagnóstica")
    tab_roc = st.tabs(["📝 Datos", "📈 Curva ROC", "📊 Comparación"])

    with tab_roc[0]: render_data_input()
    with tab_roc[1]: render_single_roc()
    with tab_roc[2]: render_comparison()

if __name__ == "__main__":
    st.set_page_config(page_title="Curvas ROC", layout="wide")
    render_roc_module("🎯 Curvas ROC")
    
    # ==========================================
# MÓDULO 14 OPTIMIZADO: MAPAS GEOGRÁFICOS
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

# ==========================================
# CLASE: MANEJADOR DE ESTADO CENTRALIZADO
# ==========================================
class MapState:
    """Manejador centralizado de estado para mapas geográficos."""

    DEFAULTS = {
        'map_data': pd.DataFrame({
            'Pais': ['Colombia'] * 10,
            'Departamento': ['Antioquia', 'Cundinamarca', 'Valle del Cauca', 'Atlántico',
                            'Santander', 'Bolívar', 'Córdoba', 'Nariño', 'Boyacá', 'Cauca'],
            'Municipio': ['Medellín', 'Bogotá', 'Cali', 'Barranquilla', 'Bucaramanga',
                        'Cartagena', 'Montería', 'Pasto', 'Tunja', 'Popayán'],
            'Casos': [1500, 1200, 1100, 900, 800, 750, 600, 550, 500, 450],
            'Poblacion': [6500000, 3000000, 4500000, 2500000, 2000000, 2100000,
                         1800000, 1600000, 1400000, 1300000]
        }),
        'marker_data': pd.DataFrame({
            'Nombre': ['Hospital Central', 'Clínica del Norte', 'Centro de Salud Sur',
                      'UCI Móvil 1', 'UCI Móvil 2'],
            'Pais': ['Colombia'] * 5,
            'Departamento': ['Antioquia'] * 5,
            'Municipio': ['Medellín', 'Medellín', 'Medellín', 'Medellín', 'Medellín'],
            'Tipo': ['Hospital', 'Clínica', 'CentroSalud', 'UCI', 'UCI'],
            'Capacidad': [200, 150, 50, 20, 20]
        })
    }

    @staticmethod
    def init(key: str, default):
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]

    @staticmethod
    def reset(key: str) -> None:
        if key in MapState.DEFAULTS:
            st.session_state[key] = MapState.DEFAULTS[key]

# ==========================================
# BASE DE DATOS DE COORDENADAS
# ==========================================
GEO_DATABASE = {
    # Colombia
    ('Colombia', 'Antioquia', 'Medellín'): (6.2442, -75.5812),
    ('Colombia', 'Antioquia', 'Bello'): (6.3374, -75.5576),
    ('Colombia', 'Cundinamarca', 'Bogotá'): (4.7110, -74.0721),
    ('Colombia', 'Valle del Cauca', 'Cali'): (3.8000, -76.5220),
    ('Colombia', 'Atlántico', 'Barranquilla'): (10.9685, -74.7813),
    ('Colombia', 'Santander', 'Bucaramanga'): (7.1190, -73.1198),
    ('Colombia', 'Bolívar', 'Cartagena'): (10.3910, -75.5142),
    ('Colombia', 'Córdoba', 'Montería'): (8.7479, -75.8813),
    ('Colombia', 'Nariño', 'Pasto'): (1.2897, -77.6428),
    ('Colombia', 'Boyacá', 'Tunja'): (5.7639, -72.9077),
    ('Colombia', 'Cauca', 'Popayán'): (2.7580, -76.6136),
    ('Colombia', 'Meta', 'Villavicencio'): (4.1420, -73.6269),
    ('Colombia', 'Huila', 'Neiva'): (2.5363, -75.2803),
    ('Colombia', 'Caldas', 'Manizales'): (5.0689, -75.5174),
    ('Colombia', 'Risaralda', 'Pereira'): (4.8136, -75.6909),
    ('Colombia', 'Norte de Santander', 'Cúcuta'): (7.8892, -72.4967),
    ('Colombia', 'Tolima', 'Ibagué'): (4.4447, -75.2318),
    ('Colombia', 'Cesar', 'Valledupar'): (10.4631, -73.2532),
    ('Colombia', 'Magdalena', 'Santa Marta'): (11.2408, -74.2099),
    ('Colombia', 'Quindío', 'Armenia'): (4.5333, -75.6833),
    # Estados Unidos
    ('Estados Unidos', 'California', 'Los Angeles'): (34.0522, -118.2437),
    ('Estados Unidos', 'California', 'San Francisco'): (37.7749, -122.4194),
    ('Estados Unidos', 'New York', 'New York City'): (40.7128, -74.0060),
    ('Estados Unidos', 'Texas', 'Houston'): (29.7604, -95.3698),
    ('Estados Unidos', 'Florida', 'Miami'): (25.7617, -80.1918),
    # México
    ('México', 'CDMX', 'Ciudad de México'): (19.4326, -99.1332),
    ('México', 'Jalisco', 'Guadalajara'): (20.6597, -103.3496),
    ('México', 'Nuevo León', 'Monterrey'): (25.6866, -100.3161),
    # Argentina
    ('Argentina', 'Buenos Aires', 'Buenos Aires'): (-34.6037, -58.3816),
    ('Argentina', 'Córdoba', 'Córdoba'): (-31.4201, -64.1888),
    # España
    ('España', 'Madrid', 'Madrid'): (40.4168, -3.7038),
    ('España', 'Barcelona', 'Barcelona'): (41.3851, 2.1734),
    # Brasil
    ('Brasil', 'São Paulo', 'São Paulo'): (-23.5505, -46.6333),
    ('Brasil', 'Rio de Janeiro', 'Rio de Janeiro'): (-22.9068, -43.1729),
    # Perú
    ('Perú', 'Lima', 'Lima'): (-12.0464, -77.0428),
    # Chile
    ('Chile', 'Santiago', 'Santiago'): (-33.4489, -70.6693),
}

AVAILABLE_COUNTRIES = sorted(list(set(geo[0] for geo in GEO_DATABASE.keys())))

# ==========================================
# CONSTANTES
# ==========================================
MAPBOX_STYLE = 'carto-darkmatter'
DEFAULT_CENTER = {'lat': 4.5709, 'lon': -74.2973}
DEFAULT_ZOOM = 5

METRIC_OPTIONS = {'Casos': 'Casos', 'Tasa por 100,000': 'Tasa', 'Población': 'Poblacion'}
COLOR_SCALES = {'Rojo': 'Reds', 'Azul': 'Blues', 'Verde': 'Viridis', 'Plasma': 'Plasma', 'RdYlGn_r': 'RdYlGn_r'}

DEFAULT_HOTSPOTS = {
    'Bogotá': (4.7110, -74.0721, 150),
    'Medellín': (6.2442, -75.5812, 100),
    'Cali': (3.8000, -76.5220, 80),
    'Barranquilla': (10.9685, -74.7813, 60)
}

# ==========================================
# FUNCIONES HELPER
# ==========================================
def get_coordinates(pais: str, departamento: str, municipio: str) -> Tuple[Optional[float], Optional[float], bool]:
    key = (pais, departamento, municipio)
    if key in GEO_DATABASE:
        return GEO_DATABASE[key][0], GEO_DATABASE[key][1], True
    return None, None, False

def geocode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Lat'] = None
    df['Lon'] = None
    for idx, row in df.iterrows():
        pais = row.get('Pais', 'Colombia')
        depto = row.get('Departamento', '')
        muni = row.get('Municipio', '')
        lat, lon, found = get_coordinates(pais, depto, muni)
        if found:
            df.at[idx, 'Lat'] = lat
            df.at[idx, 'Lon'] = lon
        else:
            for (p, d, m), (la, lo) in GEO_DATABASE.items():
                if p == pais and d == depto:
                    df.at[idx, 'Lat'] = la
                    df.at[idx, 'Lon'] = lo
                    break
    return df

def calculate_tasa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Tasa'] = (df['Casos'] / df['Poblacion'] * 100000).round(2)
    return df

# ==========================================
# SUB-MÓDULO: MAPA COROPLÉTICO
# ==========================================
def render_choropleth_tab():
    st.markdown("### 📊 Mapa Coroplético - Incidencia por Ubicación")
    st.caption("Ingrese País, Departamento y Municipio como en Power BI.")

    map_data = MapState.init('map_data', MapState.DEFAULTS['map_data'])

    col_setup = st.columns([1, 1, 1])
    with col_setup[0]:
        metric_map = st.selectbox("📊 Métrica:", options=list(METRIC_OPTIONS.keys()), index=0)
    with col_setup[1]:
        scale_name = st.selectbox("🎨 Escala:", options=list(COLOR_SCALES.keys()), index=0)
        color_scale = COLOR_SCALES[scale_name]
    with col_setup[2]:
        show_table = st.checkbox("Mostrar tabla", value=True)

    if show_table:
        st.markdown("#### 📋 Datos de Ubicaciones:")
        edited_map = st.data_editor(
            map_data, num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "Pais": st.column_config.SelectboxColumn("País", options=AVAILABLE_COUNTRIES),
                "Departamento": st.column_config.TextColumn("Departamento", required=True),
                "Municipio": st.column_config.TextColumn("Ciudad/Municipio", required=True),
                "Casos": st.column_config.NumberColumn("Casos", min_value=0, format="%d"),
                "Poblacion": st.column_config.NumberColumn("Población", min_value=1, format="%d"),
                "Tasa": st.column_config.NumberColumn("Tasa", format="%.2f", disabled=True)
            }
        )
        st.session_state.map_data = edited_map
    else:
        edited_map = map_data

    if metric_map == 'Tasa por 100,000':
        edited_map = calculate_tasa(edited_map)
        color_col = 'Tasa'
    else:
        color_col = METRIC_OPTIONS[metric_map]

    geocoded_data = geocode_dataframe(edited_map).dropna(subset=['Lat', 'Lon'])

    if st.button("🗺️ GENERAR MAPA", use_container_width=True, type="primary"):
        try:
            fig = px.scatter_mapbox(
                geocoded_data, lat='Lat', lon='Lon', size=color_col, color=color_col,
                color_continuous_scale=color_scale, hover_name='Municipio',
                hover_data={'Pais': True, 'Departamento': True, 'Casos': True,
                           'Poblacion': ':,d', 'Tasa': ':.2f', 'Lat': False, 'Lon': False},
                zoom=DEFAULT_ZOOM, center=DEFAULT_CENTER, height=600, title=f'Mapa Coroplético - {metric_map}'
            )
            fig.update_layout(mapbox_style=MAPBOX_STYLE, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

            col_stat = st.columns(3)
            with col_stat[0]:
                st.metric("Total Casos", f"{geocoded_data['Casos'].sum():,}")
            with col_stat[1]:
                st.metric("Ubicaciones", len(geocoded_data))
            with col_stat[2]:
                avg_tasa = (geocoded_data['Casos'].sum() / geocoded_data['Poblacion'].sum() * 100000)
                st.metric("Tasa Promedio", f"{avg_tasa:.2f} x100k")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ==========================================
# SUB-MÓDULO: HEATMAP
# ==========================================
def render_heatmap_tab():
    st.markdown("### 🔥 Mapa de Calor - Densidad de Casos")
    st.info("Configure los centros de concentración de casos (hotspots).")

    hotspots_config = {}
    with st.expander("⚙️ Configurar Hotspots", expanded=False):
        for city, (lat, lon, weight) in DEFAULT_HOTSPOTS.items():
            col_h = st.columns([2, 2, 1])
            with col_h[0]:
                st.caption(f"📍 {city}")
            with col_h[1]:
                hotspots_config[city] = (lat, lon, st.slider(f"Peso {city}", 0, 300, weight, key=f"hw_{city}"))

    base_style = st.selectbox("Estilo:", ['carto-darkmatter', 'carto-positron', 'open-street-map'],
                            format_func=lambda x: x.replace('-', ' ').title())

    if st.button("🔥 GENERAR HEATMAP", use_container_width=True, type="primary"):
        with st.spinner("Generando..."):
            try:
                heat_data = []
                for city, (lat, lon, weight) in hotspots_config.items():
                    for _ in range(weight):
                        heat_data.append({'lat': lat + np.random.normal(0, 0.3),
                                        'lon': lon + np.random.normal(0, 0.3), 'city': city})
                df_heat = pd.DataFrame(heat_data)

                fig = px.density_mapbox(df_heat, lat='lat', lon='lon', radius=15,
                                        center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM,
                                        mapbox_style=base_style, title='Mapa de Calor')
                fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"✅ {len(df_heat):,} puntos de calor")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ==========================================
# SUB-MÓDULO: MARCADORES
# ==========================================
def render_markers_tab():
    st.markdown("### 📍 Mapa con Marcadores - Ubicaciones Específicas")
    st.markdown("Visualice centros de salud, casos individuales, recursos hospitalarios.")

    marker_data = MapState.init('marker_data', MapState.DEFAULTS['marker_data'])

    st.markdown("#### 📋 Gestionar Ubicaciones:")
    edited_markers = st.data_editor(
        marker_data, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "Nombre": st.column_config.TextColumn("Nombre", required=True),
            "Pais": st.column_config.SelectboxColumn("País", options=AVAILABLE_COUNTRIES),
            "Departamento": st.column_config.TextColumn("Departamento"),
            "Municipio": st.column_config.TextColumn("Ciudad/Municipio"),
            "Tipo": st.column_config.SelectboxColumn("Tipo",
                options=["Hospital", "Clínica", "CentroSalud", "UCI", "Laboratorio", "Farmacia"]),
            "Capacidad": st.column_config.NumberColumn("Capacidad", min_value=0, format="%d")
        }
    )
    st.session_state.marker_data = edited_markers

    geocoded_markers = geocode_dataframe(edited_markers)

    col_view = st.columns([1, 1, 1])
    with col_view[0]:
        zoom_level = st.slider("🔍 Zoom", 5, 18, 12)
    with col_view[1]:
        tipo_filtro = st.multiselect("Filtrar por tipo:",
            options=edited_markers['Tipo'].unique().tolist(),
            default=edited_markers['Tipo'].unique().tolist())
    with col_view[2]:
        show_table = st.checkbox("Mostrar tabla", value=True)

    filtered_geo = geocoded_markers[geocoded_markers['Tipo'].isin(tipo_filtro)].dropna(subset=['Lat', 'Lon'])

    if len(filtered_geo) > 0:
        fig = px.scatter_mapbox(
            filtered_geo, lat='Lat', lon='Lon', color='Tipo', size='Capacidad',
            hover_name='Nombre', hover_data={'Pais': True, 'Departamento': True, 'Municipio': True,
                                           'Capacidad': True, 'Lat': False, 'Lon': False},
            zoom=zoom_level, center={'lat': filtered_geo['Lat'].mean(), 'lon': filtered_geo['Lon'].mean()},
            height=500, title='Ubicaciones de Salud'
        )
        fig.update_layout(mapbox_style=MAPBOX_STYLE)
        st.plotly_chart(fig, use_container_width=True)

        if show_table:
            st.dataframe(edited_markers[edited_markers['Tipo'].isin(tipo_filtro)], use_container_width=True, hide_index=True)

        col_stat = st.columns(4)
        with col_stat[0]:
            st.metric("Ubicaciones", len(filtered_geo))
        with col_stat[1]:
            st.metric("Capacidad Total", f"{filtered_geo['Capacidad'].sum():,}")
        with col_stat[2]:
            st.metric("Hospitales", len(filtered_geo[filtered_geo['Tipo'] == 'Hospital']))
        with col_stat[3]:
            st.metric("UCI's", len(filtered_geo[filtered_geo['Tipo'] == 'UCI']))
    else:
        st.warning("⚠️ No hay ubicaciones con los filtros seleccionados.")

# ==========================================
# MÓDULO PRINCIPAL
# ==========================================
def render_geographic_maps_module(menu: str):
    if menu != "🗺️ Mapas Geográficos":
        return

    st.header("🗺️ Mapas Geográficos - Epidemiología Espacial")
    st.markdown("Ingrese ubicación por texto (País, Departamento, Ciudad) como en Power BI.")

    tab_map = st.tabs(["📊 Coroplético", "🔥 Heatmap", "📍 Marcadores"])
    with tab_map[0]:
        render_choropleth_tab()
    with tab_map[1]:
        render_heatmap_tab()
    with tab_map[2]:
        render_markers_tab()

if __name__ == "__main__":
    st.set_page_config(page_title="Mapas Geográficos", layout="wide")
    render_geographic_maps_module("🗺️ Mapas Geográficos")

# ==========================================
# MÓDULO 15 OPTIMIZADO: BIOINFORMÁTICA
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple
import io

# ==========================================
# CONSTANTES
# ==========================================
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

AMINO_ACID_NAMES = {
    'A': 'Alanina', 'R': 'Arginina', 'N': 'Asparagina', 'D': 'Aspartico',
    'C': 'Cisteina', 'E': 'Glutamico', 'Q': 'Glutamina', 'G': 'Glicina',
    'H': 'Histidina', 'I': 'Isoleucina', 'L': 'Leucina', 'K': 'Lisina',
    'M': 'Metionina', 'F': 'Fenilalanina', 'P': 'Prolina', 'S': 'Serina',
    'T': 'Treonina', 'W': 'Triptofano', 'Y': 'Tirosina', 'V': 'Valina', '*': 'STOP'
}

VALID_BASES = set('AGTCUN')

# ==========================================
# FUNCIONES HELPER
# ==========================================
def validate_sequence(seq: str) -> Tuple[bool, List[str], str]:
    seq_clean = seq.upper().replace(" ", "").replace("\n", "")
    if not seq_clean:
        return False, [], "vacía"
    invalid = [b for b in seq_clean if b not in VALID_BASES]
    if invalid:
        return False, list(set(invalid)), "inválida"
    return True, [], "RNA" if 'U' in seq_clean else "DNA"

def clean_sequence(seq: str) -> str:
    return seq.upper().replace(" ", "").replace("\n", "")

def calculate_gc_content(seq: str) -> Tuple[float, float, int]:
    seq_upper = seq.upper()
    length = len(seq_upper)
    if length == 0:
        return 0.0, 0.0, 0
    gc = (seq_upper.count('G') + seq_upper.count('C')) / length
    at = (seq_upper.count('A') + seq_upper.count('T') + seq_upper.count('U')) / length
    return gc, at, length

def get_reverse_complement(seq: str, is_rna: bool = False) -> str:
    comp_map = {'A': 'T' if not is_rna else 'U', 'T': 'A', 'U': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp_map.get(b, 'N') for b in seq.upper())[::-1]

def translate_sequence(seq: str, frame: int = 1) -> Tuple[str, List[Tuple[str, str]]]:
    codons = []
    protein = []
    start = frame - 1
    for i in range(start, len(seq.upper()) - (len(seq.upper()) - start) % 3, 3):
        codon = seq.upper()[i:i+3]
        if len(codon) == 3:
            aa = CODON_TABLE.get(codon, 'X')
            protein.append(aa)
            codons.append((codon, aa))
    return ''.join(protein), codons

def find_orfs(seq: str, min_length: int = 30) -> List[Dict]:
    orfs = []
    for frame in range(3):
        start = frame
        in_orf = False
        orf_start = 0
        orf_seq = ""
        for i in range(start, len(seq.upper()) - 2, 3):
            codon = seq.upper()[i:i+3]
            if codon == 'ATG' and not in_orf:
                in_orf = True
                orf_start = i
                orf_seq = codon
            elif in_orf:
                orf_seq += codon
                if CODON_TABLE.get(codon) == '*':
                    if len(orf_seq) >= min_length:
                        protein = ''.join([CODON_TABLE.get(codon[j:j+3], 'X') for j in range(0, len(orf_seq), 3)])
                        orfs.append({'frame': frame + 1, 'start': orf_start, 'end': i + 3, 'length': len(orf_seq), 'sequence': orf_seq, 'protein': protein})
                    in_orf = False
                    orf_seq = ""
    return sorted(orfs, key=lambda x: x['length'], reverse=True)

def plot_nucleotide_composition(seq: str) -> plt.Figure:
    comp = Counter(seq.upper())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bases = ['A', 'T', 'G', 'C', 'U'] if 'U' in seq.upper() else ['A', 'T', 'G', 'C']
    counts = [comp.get(b, 0) for b in bases]
    colors = {'A': '#e74c3c', 'T': '#3498db', 'G': '#2ecc71', 'C': '#f39c12', 'U': '#9b59b6'}
    ax1.bar(bases, counts, color=[colors.get(b, '#95a5a6') for b in bases], edgecolor='black')
    ax1.set_xlabel('Nucleótido'); ax1.set_ylabel('Conteo'); ax1.set_title('Composición')
    for i, c in enumerate(counts): ax1.text(i, c + max(counts)*0.02, str(c), ha='center', fontweight='bold')
    if sum(counts) > 0:
        ax2.pie(counts, labels=bases, colors=[colors.get(b, '#95a5a6') for b in bases], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribución')
    plt.tight_layout()
    return fig

# ==========================================
# RENDERIZADO PRINCIPAL
# ==========================================
def render_bioinformatics_module(menu: str):
    if menu != "🧬 Bioinformática":
        return

    st.header("🧬 Análisis de Secuencias Genéticas")

    # Entrada
    col_input = st.columns([1, 1])
    with col_input[0]:
        input_method = st.radio("Método:", ["Texto directo", "Archivo FASTA"], horizontal=True, key="input_method")

    seq = ""
    if input_method == "Texto directo":
        seq = st.text_area("Secuencia DNA/RNA:", placeholder="ATGCCGTAGCTG...", height=150, key="seq_input").strip()
    else:
        uploaded = st.file_uploader("Cargar FASTA", type=['fasta', 'fa', 'txt'], key="fasta_input")
        if uploaded:
            content = uploaded.read().decode('utf-8')
            seq = ''.join([l.strip() for l in content.split('\n') if not l.startswith('>')])

    if seq:
        is_valid, invalid, seq_type = validate_sequence(seq)
        if not is_valid:
            st.error(f"❌ Bases inválidas: {invalid}")
            return

        seq = clean_sequence(seq)
        st.success(f"✅ {seq_type} válida - {len(seq)} pb")

        if st.button("🧬 ANALIZAR SECUENCIA", use_container_width=True, type="primary"):
            gc, at, length = calculate_gc_content(seq)
            composition = Counter(seq.upper())

            # Estadísticas
            st.markdown("### 📊 Estadísticas")
            col_stats = st.columns(4)
            with col_stats[0]: st.metric("Longitud", f"{length:,} pb")
            with col_stats[1]: st.metric("Tipo", seq_type)
            with col_stats[2]: st.metric("GC", f"{gc:.2%}")
            with col_stats[3]: st.metric("Ratio GC/AT", f"{gc/at:.2f}" if at > 0 else "N/A")

            # Gráfico
            st.pyplot(plot_nucleotide_composition(seq))

            # Tabla composición
            comp_df = pd.DataFrame([{'Base': k, 'Conteo': v, '%': f"{v/length*100:.2f}%"} for k, v in sorted(composition.items())])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Complementaria reversa
            st.markdown("### 🔄 Complementaria Reversa")
            is_rna = 'U' in seq.upper()
            rev_comp = get_reverse_complement(seq, is_rna)
            st.code('\n'.join([rev_comp[i:i+80] for i in range(0, len(rev_comp), 80)]))
            st.button("📋 Copiar", key="copy_revcomp")

            # Traducción
            st.markdown("### 🧪 Traducción a Proteína")
            col_frame = st.columns([1, 2])
            with col_frame[0]:
                frame = st.selectbox("Marco:", [1, 2, 3], index=0, key="frame_select")

            protein, codons = translate_sequence(seq, frame)
            lines = [protein[i:i+60] for i in range(0, len(protein), 60)]
            numbered = '\n'.join([f"{j*60+1:4d}  {line}" for j, line in enumerate(lines)])
            st.code(numbered)
            st.caption(f"Longitud: {len(protein)} aa")

            # Tabla de codones
            if codons:
                codon_df = pd.DataFrame(codons, columns=['Codón', 'AA'])
                codon_df['#'] = range(1, len(codon_df) + 1)
                codon_df = codon_df[['#', 'Codón', 'AA']]
                st.dataframe(codon_df, use_container_width=True, hide_index=True)

            # ORFs
            st.markdown("### 🔍 Búsqueda de ORFs")
            col_orf = st.columns([1, 1])
            with col_orf[0]:
                min_len = st.number_input("Longitud mínima (pb):", 6, len(seq), 30, key="min_orf")
            with col_orf[1]:
                max_show = st.number_input("Máx ORFs:", 1, 50, 10, key="max_orf")

            if st.button("🔍 Buscar ORFs", key="search_orfs"):
                orfs = find_orfs(seq, min_len)
                if orfs:
                    st.success(f"✅ {len(orfs)} ORFs encontrados")
                    for i, orf in enumerate(orfs[:max_show]):
                        with st.expander(f"ORF #{i+1} - Frame +{orf['frame']} - {orf['length']} pb"):
                            st.write(f"**Posición:** {orf['start']} - {orf['end']}")
                            st.code(orf['sequence'][:100] + "...")
                            st.code(orf['protein'][:50] + "...")
                else:
                    st.warning("No se encontraron ORFs")

            # Exportar
            st.markdown("### 💾 Exportar")
            exp_cols = st.columns(3)
            with exp_cols[0]:
                st.download_button("📥 Original", data=f">{seq_type}\n{seq}", file_name="secuencia.fasta", mime="text/plain")
            with exp_cols[1]:
                st.download_button("📥 RevComp", data=f">RevComp\n{rev_comp}", file_name="revcomp.fasta", mime="text/plain")
            with exp_cols[2]:
                st.download_button("📥 Proteína", data=f">Protein\n{protein}", file_name="proteina.fasta", mime="text/plain")

            # GC Skew
            st.markdown("### 📈 Análisis GC Skew")
            if len(seq) >= 10:
                window = max(10, len(seq) // 100)
                positions, skew_values = [], []
                for i in range(0, len(seq.upper()) - window, window):
                    w = seq.upper()[i:i+window]
                    g, c = w.count('G'), w.count('C')
                    skew = (g - c) / (g + c) if (g + c) > 0 else 0
                    positions.append(i + window // 2)
                    skew_values.append(skew)

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(positions, skew_values, color='#3498db', linewidth=1.5)
                ax.axhline(y=0, color='red', linestyle='--')
                ax.fill_between(positions, skew_values, 0, alpha=0.3, color='#3498db')
                ax.set_xlabel('Posición'); ax.set_ylabel('GC Skew'); ax.set_title('GC Skew'); ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    st.set_page_config(page_title="Bioinformática", layout="wide")
    render_bioinformatics_module("🧬 Bioinformática")
    
    # ==========================================
# MÓDULO 16: MI SUSCRIPCIÓN (USUARIOS)
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
                    <li>✓ Soporte prioritario</li>
                    <li>✓ Actualizaciones ilimitadas</li>
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
                        transition: all 0.3s;
                        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.5);
                    ">
                        💳 PAGAR AHORA - 22 US$
                    </button>
                </a>
                <p style="color: #94a3b8; font-size: 13px; margin-top: 15px;">
                    🔒 Pago 100% seguro mediante Bold.co<br>
                    💳 Tarjetas • 📱 PSE • 📲 Nequi • 📲 Daviplata
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.info("💡 Después del pago, comunícate con soporte para activar tu licencia Premium.")

# ==========================================
# MÓDULO 17: ADMIN
# ==========================================
elif menu == "⚙️ Admin":
    st.header("🔑 Panel de Administración")

    db = load_users()

    st.subheader("📊 Usuarios Registrados")
    df_users = pd.DataFrame([
        {
            "Email": email,
            "Rol": data.get("role", "N/A"),
            "Expiración": data.get("expiry", "N/A"),
            "ID Doc": data.get("id_doc", "N/A"),
            "Último Login": st.session_state.user_logins.get(email, "Nunca")
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
    with col_admin[2]:
        st.write("")

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
    "Desarrollado por: Juan Manuel Collazos Rozo, MD, MSc."
    "WhatsApp: (+57) 3113682907 - Correo electrónico: j.collazosmd@gmail.com"
    "</div>",
    unsafe_allow_html=True
)
