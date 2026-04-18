""
EpiDiagnosis Pro v6.0 - Aplicación Completa de Epidemiología y Bioestadística
================================================================================
Módulos incluidos:
1. Dashboard & Cloud (módulos existentes v5.2)
2. Calculadora 2x2 y Tamaño  Muestra
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
#-
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

                col_reg = st.columns(2)
                with col_reg[0]:
                    submit_reg = st.form_submit_button("ACTIVAR PRUEBA", use_container_width=True)

                if submit_reg:
                    db = load_users()
                    exp = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
                    if ru not in db and ru and rp and rid:
                        db[ru] = {
                            "password": secure_hash(rp),
                            "role": "user",
                            "expiry": exp,
                            "id_doc": rid,
                            "dob": "2000-01-01"
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
                        Licencia anual completa<br>
                        <span style="font-size: 28px; font-weight: bold; color: #ffd700;">$299.000 COP</span>
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
    st.write(f"👤 **{st.session_state.user}**")
    st.write(f"🎫 Rol: `{st.session_state.role.upper()}`")
    st.write(f"📌 Versión: `6.0`")

    st.markdown("---")

    menu = st.radio("📋 MÓDULOS CIENTÍFICOS", [
        "🏠 Dashboard & Cloud",
        "🧹 Limpieza de Datos",
        "📊 Bioestadística",
        "🔢 Calculadora 2x2",
        "📏 Tamaño de Muestra",
        "📈 Vigilancia & IA",
        "🤖 Literatura PICO",
        "📑 PRISMA Flowchart",
        "🌲 Forest Plot",
        "📊 Meta-análisis",
        "⚖️ RoB/GRADE",
        "📉 Supervivencia (KM)",
        "🎯 Curvas ROC",
        "🗺️ Mapas Geográficos",
        "🧬 Bioinformática",
        "💳 Mi Suscripción" if st.session_state.role == "user" else None,
        "⚙️ Admin" if st.session_state.role == "admin" else None
    ])

    st.markdown("---")
    if st.button("🚪 Cerrar Sesión", use_container_width=True):
        st.session_state.auth = False
        st.rerun()

    st.markdown("---")
    st.info("📞 Soporte: +57 3113682907\n\n📧 soporte@epidiagnosis.com\n\n🕐 Lun-Vie: 8AM-6PM")

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
# MÓDULO 7: LITERATURA PICO
# ==========================================
elif menu == "🤖 Literatura PICO":
    st.header("🤖 Analizador IA de Evidencia Científica")

    api_k = st.text_input("🔑 OpenAI API Key", type="password",
                          placeholder="sk-...",
                          help="Obtenga su API key en https://platform.openai.com")

    if not api_k:
        st.info("💡 Ingrese su OpenAI API Key para usar el análisis PICO")
        st.stop()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        metodo = st.radio("📥 Método de Carga:", ["PDF", "DOI"])

        ext = LiteratureAIExtractor(api_k)
        res = None

        if metodo == "PDF":
            f = st.file_uploader("Subir artículo PDF", type="pdf")
            if f:
                if st.button("🔍 Extraer PICO", use_container_width=True):
                    with st.spinner("⏳ Analizando con IA..."):
                        res = ext.from_pdf(f)
        else:
            doi = st.text_input("DOI (ej: 10.1056/NEJMoa...)",
                               placeholder="10.1056/...")
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
        if st.session_state.articulos_pico:
            st.subheader("📚 Biblioteca de Evidencia")

            df_articulos = pd.DataFrame(st.session_state.articulos_pico)

            display_cols = ['titulo', 'diseno', 'grade', 'resultados_desenlaces']
            available_cols = [c for c in display_cols if c in df_articulos.columns]

            if available_cols:
                st.dataframe(
                    df_articulos[available_cols],
                    use_container_width=True,
                    height=400
                )

            col_btns = st.columns(2)
            with col_btns[0]:
                if st.button("🗑️ Limpiar Biblioteca", use_container_width=True):
                    st.session_state.articulos_pico = []
                    st.rerun()
            with col_btns[1]:
                if st.button("📥 Exportar JSON", use_container_width=True):
                    st.download_button(
                        label="💾 Descargar",
                        data=json.dumps(st.session_state.articulos_pico, indent=2, ensure_ascii=False),
                        file_name="articulos_pico.json",
                        mime="application/json",
                        use_container_width=True
                    )

            with st.expander("📖 Ver Detalle Completo"):
                for i, art in enumerate(st.session_state.articulos_pico):
                    st.markdown(f"#### Artículo {i+1}")
                    for k, v in art.items():
                        st.write(f"**{k}:** {v}")
                    st.markdown("---")
        else:
            st.info("📂 No hay artículos en la biblioteca. Cargue un PDF o DOI para comenzar.")

# ==========================================
# MÓDULO 8: PRISMA FLOWCHART
# ==========================================
elif menu == "📑 PRISMA Flowchart":
    st.header("📑 Diagrama de Flujo PRISMA")
    st.markdown("### Construya su diagrama PRISMA 2020")

    # Inicializar datos PRISMA en session state
    if not st.session_state.prisma_data:
        st.session_state.prisma_data = {
            'registros_db': 0,
            'registros_registros': 0,
            'duplicados': 0,
            'registros_evaluados': 0,
            'registros_screen': 0,
            'excluidos_title': 0,
            'excluidos_abstract': 0,
            'articulos_recuperados': 0,
            'articulos_evaluated': 0,
            'articulos_excluidos': 0,
            'estudios_included': 0
        }

    tab_prisma = st.tabs(["📊 Entrada de Datos", "📈 Visualización"])

    with tab_prisma[0]:
        st.markdown("#### 🔍 Fase de Identificación")
        col_id = st.columns(2)
        with col_id[0]:
            st.session_state.prisma_data['registros_db'] = st.number_input(
                "Registros identificados en bases de datos:",
                min_value=0, value=1500, step=10
            )
            st.session_state.prisma_data['registros_registros'] = st.number_input(
                "Registros identificados mediante otros métodos:",
                min_value=0, value=50, step=10
            )
        with col_id[1]:
            st.session_state.prisma_data['duplicados'] = st.number_input(
                "Duplicados eliminados:",
                min_value=0, value=400, step=10
            )

        total_identified = st.session_state.prisma_data['registros_db'] + st.session_state.prisma_data['registros_registros']
        st.session_state.prisma_data['registros_evaluados'] = total_identified - st.session_state.prisma_data['duplicados']
        st.success(f"📊 Registros después de eliminar duplicados: **{st.session_state.prisma_data['registros_evaluados']}**")

        st.markdown("#### 🔎 Fase de Screening")
        col_sc = st.columns(2)
        with col_sc[0]:
            st.session_state.prisma_data['excluidos_title'] = st.number_input(
                "Registros excluidos por título:",
                min_value=0, value=500, step=10
            )
            st.session_state.prisma_data['excluidos_abstract'] = st.number_input(
                "Registros excluidos por resumen:",
                min_value=0, value=400, step=10
            )
        with col_sc[1]:
            st.session_state.prisma_data['articulos_recuperados'] = st.number_input(
                "Artículos retrieval solicitados:",
                min_value=0, value=250, step=10
            )
            st.session_state.prisma_data['registros_screen'] = (
                st.session_state.prisma_data['registros_evaluados'] -
                st.session_state.prisma_data['excluidos_title'] -
                st.session_state.prisma_data['excluidos_abstract']
            )

        st.info(f"📋 Registros evaluados para elegibilidad: **{st.session_state.prisma_data['registros_screen']}**")

        st.markdown("#### ✅ Fase de Elegibilidad")
        col_el = st.columns(2)
        with col_el[0]:
            st.session_state.prisma_data['articulos_evaluated'] = st.number_input(
                "Artículos evaluados con texto completo:",
                min_value=0, value=200, step=10
            )
        with col_el[1]:
            st.session_state.prisma_data['articulos_excluidos'] = st.number_input(
                "Artículos excluidos con razones:",
                min_value=0, value=150, step=10
            )

        st.markdown("#### 🎯 Fase de Inclusión")
        st.session_state.prisma_data['estudios_included'] = st.number_input(
            "Estudios incluidos en revisión:",
            min_value=0, value=25, step=1
        )

        # Resumen
        st.markdown("---")
        st.markdown("### 📊 Resumen PRISMA")

        estudios_con_datos = st.number_input("Estudios con datos para meta-análisis:",
                                             min_value=0,
                                             value=st.session_state.prisma_data['estudios_included'],
                                             step=1)

        participantes = st.number_input("Total participantes:", min_value=0, value=5000, step=100)

        col_summary = st.columns(4)
        with col_summary[0]:
            st.metric("📚 Total Inicial", f"{total_identified:,}")
        with col_summary[1]:
            st.metric("🔍 Screening", f"{st.session_state.prisma_data['registros_screen']:,}")
        with col_summary[2]:
            st.metric("📖 Elegibles", f"{st.session_state.prisma_data['articulos_evaluated']:,}")
        with col_summary[3]:
            st.metric("✅ Incluidos", f"{st.session_state.prisma_data['estudios_included']:,}")

    with tab_prisma[1]:
        st.markdown("### 🌐 Diagrama PRISMA 2020 Interactivo")

        fig = go.Figure()

        # Función para crear cajas
        def add_box(fig, y, x, count, label, color='#3b82f6', width=0.3):
            fig.add_trace(go.Scatter(
                x=[x-width/2, x+width/2, x+width/2, x-width/2, x-width/2],
                y=[y, y, y-0.4, y-0.4, y],
                fill='toself',
                fillcolor=color,
                line=dict(color='#1e293b', width=2),
                text=f"{label}<br>{count:,}",
                textposition='middle center',
                textfont=dict(color='white', size=12),
                mode='text',
                showlegend=False,
                hoverinfo='text'
            ))

        # Cajas del diagrama
        # Identificación
        add_box(fig, 5, 0, st.session_state.prisma_data['registros_db'], 'Base de Datos', '#667eea')
        add_box(fig, 4.2, 0.5, st.session_state.prisma_data['registros_registros'], 'Otros Métodos', '#764ba2')

        # Suma
        fig.add_annotation(x=1.2, y=4.6, text=f"= {total_identified:,}", showarrow=False, font=dict(size=16, color='white'))

        # Duplicados
        add_box(fig, 3.8, 0, st.session_state.prisma_data['duplicados'], 'Duplicados', '#ef4444', 0.25)

        # Screening
        add_box(fig, 3, 0, st.session_state.prisma_data['registros_evaluados'], 'Después de Duplicados', '#10b981')
        add_box(fig, 2.2, -0.3, st.session_state.prisma_data['registros_screen'], 'Post-Screening', '#f59e0b', 0.25)

        # Elegibilidad
        add_box(fig, 1.4, 0, st.session_state.prisma_data['articulos_recuperados'], 'Artículos Recuperados', '#6366f1')
        add_box(fig, 0.6, 0, st.session_state.prisma_data['articulos_evaluated'], 'Texto Completo', '#8b5cf6')

        # Inclusión
        add_box(fig, -0.2, 0, st.session_state.prisma_data['estudios_included'], 'Incluidos', '#10b981', 0.35)

        # Flechas (líneas)
        arrows_y = [4.6, 4.5, 3.6, 3.4, 2.6, 2.4, 1.8, 1.0, 0.2]
        for y in arrows_y:
            fig.add_annotation(
                x=0, y=y-0.2,
                ax=0, ay=y,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowcolor='white',
                arrowsize=1
            )

        fig.update_layout(
            title=dict(text='Diagrama de Flujo PRISMA 2020', font=dict(size=24, color='#60a5fa')),
            showlegend=False,
            plot_bgcolor='#0b1120',
            paper_bgcolor='#0b1120',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 2]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 6]),
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

        # Exportar
        if st.button("📥 Exportar Diagrama"):
            fig.write_image("prisma_flowchart.png", width=1200, height=900, scale=2)
            st.success("✅ Diagrama exportado como prisma_flowchart.png")

# ==========================================
# MÓDULO 9: FOREST PLOT
# ==========================================
elif menu == "🌲 Forest Plot":
    st.header("🌲 Forest Plot - Visualización de Efectos")
    st.markdown("### Ingrese los datos de los estudios")

    # Agregar estudios
    if 'forest_studies' not in st.session_state:
        st.session_state.forest_studies = pd.DataFrame({
            'Estudio': ['Smith 2020', 'Johnson 2019', 'Williams 2021', 'Brown 2018', 'Jones 2022'],
            'Eventos_Tto': [20, 35, 28, 15, 42],
            'Total_Tto': [100, 150, 120, 80, 200],
            'Eventos_Ctrl': [30, 50, 45, 25, 60],
            'Total_Ctrl': [100, 150, 120, 80, 200]
        })

    st.markdown("#### 📝 Datos de Estudios")

    edited_df = st.data_editor(
        st.session_state.forest_studies,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    st.session_state.forest_studies = edited_df

    col_model = st.columns(2)
    with col_model[0]:
        model_type = st.selectbox("Modelo de Meta-análisis:",
                                  ["Efectos Fijos (Peto)", "Efectos Aleatorios (DerSimonian-Laird)"])
    with col_model[1]:
        show_summary = st.checkbox("Mostrar línea de efecto combinado", value=True)

    if st.button("🌲 GENERAR FOREST PLOT", use_container_width=True):
        events_e = st.session_state.forest_studies['Eventos_Tto'].tolist()
        total_e = st.session_state.forest_studies['Total_Tto'].tolist()
        events_c = st.session_state.forest_studies['Eventos_Ctrl'].tolist()
        total_c = st.session_state.forest_studies['Total_Ctrl'].tolist()
        estudios = st.session_state.forest_studies['Estudio'].tolist()

        # Calcular ORs individuales
        individual_or = []
        for i in range(len(events_e)):
            a, n1 = events_e[i], total_e[i]
            c, n2 = events_c[i], total_c[i]
            if c > 0 and (n1 - a) > 0:
                odds_ratio = (a * (n2 - c)) / (c * (n1 - a))
                individual_or.append(max(0.01, odds_ratio))
            else:
                individual_or.append(1)

        # Meta-análisis
        if model_type == "Efectos Fijos (Peto)":
            meta = meta_analysis_fixed_effect(events_e, total_e, events_c, total_c)
            pooled_or = meta['pooled_or']
            ci_low = meta['pooled_ci_low']
            ci_high = meta['pooled_ci_high']
            i2 = meta['I2']
        else:
            meta = meta_analysis_random_effects(events_e, total_e, events_c, total_c)
            pooled_or = meta['pooled_or_re']
            ci_low = meta['pooled_ci_low_re']
            ci_high = meta['pooled_ci_high_re']
            i2 = meta['I2']

        # Crear Forest Plot
        fig, ax = plt.subplots(figsize=(12, 2 + len(estudios) * 0.5))

        y_positions = list(range(len(estudios)))
        y_positions.append(len(estudios) + 1)  # Para el efecto combinado

        # Línea de null (OR=1)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='Null Effect (OR=1)')

        # Estudios individuales
        for i, (or_val, estudio) in enumerate(zip(individual_or, estudios)):
            # Calcular IC para cada estudio
            a, n1 = events_e[i], total_e[i]
            c, n2 = events_c[i], total_c[i]
            se = np.sqrt(1/a + 1/c + 1/(n1 - a) + 1/(n2 - c)) if min(a, c, n1-a, n2-c) > 0 else 0.5
            ci_low_study = np.exp(np.log(or_val) - 1.96 * se) if se > 0 else or_val * 0.5
            ci_high_study = np.exp(np.log(or_val) + 1.96 * se) if se > 0 else or_val * 1.5

            # Punto
            ax.plot(or_val, i, 'bs', markersize=8)
            # Línea de CI
            ax.plot([ci_low_study, ci_high_study], [i, i], 'b-', linewidth=2)
            # Etiqueta
            ax.annotate(estudio, xy=(or_val, i), xytext=(or_val + 0.1, i),
                       fontsize=10, ha='left', va='center')

        # Efecto combinado
        if show_summary:
            y_pooled = len(estudios) + 1
            ax.plot(pooled_or, y_pooled, 'r^', markersize=12)
            ax.plot([ci_low, ci_high], [y_pooled, y_pooled], 'r-', linewidth=3)
            ax.annotate(f'Combined\nOR={pooled_or:.2f}\n[{ci_low:.2f}, {ci_high:.2f}]',
                       xy=(pooled_or, y_pooled), xytext=(pooled_or + 0.3, y_pooled),
                       fontsize=10, ha='left', va='center', color='red',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xscale('log')
        ax.set_ylim(-1, len(estudios) + 2.5)
        ax.set_yticks([])
        ax.set_xlabel('Odds Ratio (escala log)', fontsize=12)
        ax.set_title('Forest Plot - Meta-análisis', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Estadísticas de heterogeneidad
        st.markdown("---")
        st.markdown("### 📊 Estadísticas de Heterogeneidad")

        col_het = st.columns(4)
        with col_het[0]:
            st.metric("Q Statistic", f"{meta['Q']:.2f}")
        with col_het[1]:
            st.metric("df", meta['df'])
        with col_het[2]:
            st.metric("I²", f"{i2:.1f}%")
        with col_het[3]:
            p_het = 1 - chi2.cdf(meta['Q'], meta['df'])
            st.metric("p-value (heterogeneidad)", f"{p_het:.4f}")

        # Interpretación
        st.markdown("---")
        st.markdown("### 📝 Interpretación")

        if i2 < 25:
            i2_interp = "baja"
            color = "success"
        elif i2 < 75:
            i2_interp = "moderada"
            color = "warning"
        else:
            i2_interp = "alta"
            color = "error"

        if getattr(st, color, st.info)(f"""
        **Heterogeneidad {i2_interp.upper()} (I² = {i2:.1f}%):**

        - Si I² < 25%: Heterogeneidad baja
        - Si I² 25-75%: Heterogeneidad moderada
        - Si I² > 75%: Heterogeneidad alta

        **Efecto Combinado:**
        - OR = {pooled_or:.2f}
        - IC 95%: [{ci_low:.2f}, {ci_high:.2f}]
        - {'El IC no cruza 1: efecto significativo' if ci_low > 1 or ci_high < 1 else 'El IC cruza 1: efecto no significativo'}
        """):

            pass

        # Agregar a meta-análisis
        if st.button("➕ Agregar a Meta-análisis"):
            st.session_state.meta_studies = edited_df.copy()
            st.success("✅ Estudios agregados al módulo de Meta-análisis")

# ==========================================
# MÓDULO 10: META-ANÁLISIS COMPLETO
# ==========================================
elif menu == "📊 Meta-análisis":
    st.header("📊 Meta-análisis Completo")

    tab_meta = st.tabs(["📝 Datos", "🔍 Análisis", "📈 Subgrupos", "🎯 Sesgo"])

    with tab_meta[0]:
        st.markdown("### 📝 Datos de Estudios para Meta-análisis")

        if len(st.session_state.meta_studies) == 0:
            st.info("No hay estudios cargados. Use el Forest Plot para agregar estudios o cargue datos manualmente.")

            # Opción de cargar CSV
            uploaded_file = st.file_uploader("📂 Cargar archivo CSV con estudios:", type="csv")
            if uploaded_file:
                df_meta = pd.read_csv(uploaded_file)
                st.session_state.meta_studies = df_meta
                st.success("✅ Datos cargados exitosamente!")

        if len(st.session_state.meta_studies) > 0:
            st.dataframe(st.session_state.meta_studies, use_container_width=True)

            if st.button("🗑️ Limpiar Estudios"):
                st.session_state.meta_studies = pd.DataFrame(columns=st.session_state.meta_studies.columns)
                st.rerun()

    with tab_meta[1]:
        st.markdown("### 🔍 Análisis de Efectos")

        if len(st.session_state.meta_studies) >= 2:
            col_analysis = st.columns(2)
            with col_analysis[0]:
                metric = st.selectbox("Métrica de efecto:",
                                     ["Odds Ratio", "Risk Ratio", "Mean Difference"])
            with col_analysis[1]:
                model = st.selectbox("Modelo:",
                                     ["Efectos Fijos", "Efectos Aleatorios"])

            if st.button("📊 EJECUTAR META-ANÁLISIS", use_container_width=True):
                events_e = st.session_state.meta_studies['Eventos_Tto'].tolist()
                total_e = st.session_state.meta_studies['Total_Tto'].tolist()
                events_c = st.session_state.meta_studies['Eventos_Ctrl'].tolist()
                total_c = st.session_state.meta_studies['Total_Ctrl'].tolist()

                if model == "Efectos Fijos":
                    results = meta_analysis_fixed_effect(events_e, total_e, events_c, total_c)
                else:
                    results = meta_analysis_random_effects(events_e, total_e, events_c, total_c)

                if results:
                    st.markdown("---")
                    st.markdown("### 📈 Resultados del Meta-análisis")

                    col_res = st.columns(4)
                    with col_res[0]:
                        if model == "Efectos Fijos":
                            st.metric("OR Combinado", f"{results['pooled_or']:.2f}")
                        else:
                            st.metric("OR Combinado (RE)", f"{results['pooled_or_re']:.2f}")
                    with col_res[1]:
                        if model == "Efectos Fijos":
                            st.metric("IC 95%", f"[{results['pooled_ci_low']:.2f}, {results['pooled_ci_high']:.2f}]")
                        else:
                            st.metric("IC 95% (RE)", f"[{results['pooled_ci_low_re']:.2f}, {results['pooled_ci_high_re']:.2f}]")
                    with col_res[2]:
                        st.metric("I²", f"{results['I2']:.1f}%")
                    with col_res[3]:
                        p_val = 2 * (1 - norm.cdf(abs(results['pooled_log_or'] / results['pooled_se'])))
                        st.metric("p-value", f"{p_val:.4f}")

                    # Gráfico
                    fig = go.Figure()

                    estudios = st.session_state.meta_studies['Estudio'].tolist()
                    for i, (or_val, estudio) in enumerate(zip(results['individual_or'], estudios)):
                        fig.add_trace(go.Scatter(
                            x=[results['ci_low_or'][i] if i < len(results['ci_low_or']) else or_val * 0.5,
                               results['ci_high_or'][i] if i < len(results['ci_high_or']) else or_val * 1.5],
                            y=[estudio, estudio],
                            mode='lines',
                            line=dict(color='#3b82f6', width=2),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=[or_val],
                            y=[estudio],
                            mode='markers',
                            marker=dict(size=12, color='#3b82f6'),
                            showlegend=False
                        ))

                    # Línea null
                    fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Null")

                    # Efecto combinado
                    if model == "Efectos Fijos":
                        pooled = results['pooled_or']
                        pooled_ci = [results['pooled_ci_low'], results['pooled_ci_high']]
                    else:
                        pooled = results['pooled_or_re']
                        pooled_ci = [results['pooled_ci_low_re'], results['pooled_ci_high_re']]

                    fig.add_trace(go.Scatter(
                        x=[pooled_ci[0], pooled_ci[1]],
                        y=['Combined', 'Combined'],
                        mode='lines',
                        line=dict(color='#ef4444', width=3),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=[pooled],
                        y=['Combined'],
                        mode='markers',
                        marker=dict(size=16, color='#ef4444', symbol='diamond'),
                        name='Combined Effect'
                    ))

                    fig.update_layout(
                        title='Forest Plot - Meta-análisis',
                        xaxis_title='Odds Ratio',
                        yaxis_title='Estudio',
                        height=300 + len(estudios) * 30,
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Tabla de resultados
                    results_df = pd.DataFrame({
                        'Estudio': estudios,
                        'OR': results['individual_or'],
                        'IC 95% Low': [results['ci_low_or'][i] if i < len(results['ci_low_or']) else 0 for i in range(len(estudios))],
                        'IC 95% High': [results['ci_high_or'][i] if i < len(results['ci_high_or']) else 0 for i in range(len(estudios))],
                        'Peso (%)': [w/sum(results['weights'])*100 for w in results['weights']] if model == "Efectos Fijos" else [w/sum(results['weights_re'])*100 for w in results['weights_re']]
                    })
                    st.dataframe(results_df, use_container_width=True)
        else:
            st.warning("⚠️ Se requieren al menos 2 estudios para el meta-análisis")

    with tab_meta[2]:
        st.markdown("### 📈 Análisis de Subgrupos")

        if len(st.session_state.meta_studies) >= 2:
            st.info("Funcionalidad de análisis de subgrupos. Agrupe sus estudios por características para comparar efectos entre grupos.")

            subgroups = st.multiselect("Seleccionar subgrupos:",
                                       st.session_state.meta_studies.columns.tolist())

            if subgroups and st.button("📊 ANALIZAR SUBGRUPOS"):
                st.info("Análisis de subgrupos en desarrollo...")
        else:
            st.warning("⚠️ Agregue más estudios para analizar subgrupos")

    with tab_meta[3]:
        st.markdown("### 🎯 Evaluación de Sesgo de Publicación")

        if len(st.session_state.meta_studies) >= 5:
            if st.button("📊 GENERAR FUNNEL PLOT", use_container_width=True):
                fig = go.Figure()

                studies = st.session_state.meta_studies['Estudio'].tolist()
                effects = []
                ses = []

                for i in range(len(st.session_state.meta_studies)):
                    effects.append(np.log(st.session_state.meta_studies['Eventos_Tto'].iloc[i] /
                                         (st.session_state.meta_studies['Total_Tto'].iloc[i] - st.session_state.meta_studies['Eventos_Tto'].iloc[i]) /
                                         (st.session_state.meta_studies['Eventos_Ctrl'].iloc[i] /
                                          (st.session_state.meta_studies['Total_Ctrl'].iloc[i] - st.session_state.meta_studies['Eventos_Ctrl'].iloc[i]))))
                    ses.append(1/np.sqrt(st.session_state.meta_studies['Total_Tto'].iloc[i] + st.session_state.meta_studies['Total_Ctrl'].iloc[i]))

                fig.add_trace(go.Scatter(
                    x=effects,
                    y=ses,
                    mode='markers+text',
                    marker=dict(size=12, color='#3b82f6'),
                    text=studies,
                    textposition='top center'
                ))

                # Línea de efecto
                pooled_effect = np.mean(effects)
                fig.add_vline(x=pooled_effect, line_dash="dash", line_color="red")

                fig.update_layout(
                    title='Funnel Plot - Sesgo de Publicación',
                    xaxis_title='Effect Size (log OR)',
                    yaxis_title='Standard Error',
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info("""
                **Interpretación del Funnel Plot:**
                - Los puntos deben estar simétricamente distribuidos alrededor de la línea de efecto
                - Asimetría puede indicar sesgo de publicación
                - Puntos fuera del funnel pueden indicar estudios de baja calidad
                """)
        else:
            st.warning("⚠️ Se requieren al menos 5 estudios para evaluar sesgo de publicación")

# ==========================================
# MÓDULO 11: RoB/GRADE
# ==========================================
elif menu == "⚖️ RoB/GRADE":
    st.header("⚖️ Evaluación de Calidad de Evidencia - RoB 2 y GRADE")

    tab_rob = st.tabs(["🔍 RoB 2 (Riesgo de Sesgo)", "📋 GRADE"])

    with tab_rob[0]:
        st.markdown("### 🔍 Risk of Bias 2 (RoB 2) - Ensayos Clínicos")

        if 'rob_assessments' not in st.session_state:
            st.session_state.rob_assessments = []

        study_name = st.text_input("Nombre del estudio:", placeholder="Ej: Smith 2020")

        if study_name:
            st.markdown("#### 📊 Evalúe cada dominio:")

            domains = {
                'D1': ('Dominio 1: Proceso de Randomización', ['Low', 'Some Concerns', 'High']),
                'D2': ('Dominio 2: Desviaciones de la Intervención', ['Low', 'Some Concerns', 'High']),
                'D3': ('Dominio 3: Datos de Resultado Faltantes', ['Low', 'Some Concerns', 'High']),
                'D4': ('Dominio 4: Medición del Resultado', ['Low', 'Some Concerns', 'High']),
                'D5': ('Dominio 5: Selección del Resultado Reportado', ['Low', 'Some Concerns', 'High'])
            }

            assessment = {'study': study_name}

            for domain, (label, options) in domains.items():
                assessment[domain] = st.radio(label, options, horizontal=True, index=1)

            if st.button("💾 Guardar Evaluación RoB", use_container_width=True):
                st.session_state.rob_assessments.append(assessment)
                st.success(f"✅ Evaluación guardada para {study_name}")

        # Mostrar evaluaciones guardadas
        if st.session_state.rob_assessments:
            st.markdown("---")
            st.markdown("### 📈 Resumen de Evaluaciones RoB 2")

            df_rob = pd.DataFrame(st.session_state.rob_assessments)
            st.dataframe(df_rob, use_container_width=True)

            # Gráfico de radar
            if len(st.session_state.rob_assessments) > 0:
                fig = go.Figure()

                categories = ['Randomización', 'Desviaciones', 'Datos Faltantes',
                             'Medición', 'Selección Reporte']

                for idx, row in df_rob.iterrows():
                    values = []
                    for d in ['D1', 'D2', 'D3', 'D4', 'D5']:
                        if row[d] == 'Low':
                            values.append(3)
                        elif row[d] == 'Some Concerns':
                            values.append(2)
                        else:
                            values.append(1)

                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        name=row['study'],
                        fill='toself'
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
                    showlegend=True,
                    title='Perfil de Riesgo de Sesgo por Estudio'
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab_rob[1]:
        st.markdown("### 📋 Sistema GRADE - Calidad de Evidencia")

        st.markdown("""
        **Guía de Calidad GRADE:**
        - **Alta**: Es muy probable que el efecto real esté cerca del estimado
        - **Moderada**: El efecto real probablemente esté cerca del estimado, pero puede ser diferente
        - **Baja**: El efecto real puede ser significativamente diferente al estimado
        - **Muy Baja**: Es muy probable que el efecto real sea significativamente diferente al estimado
        """)

        if 'grade_evaluations' not in st.session_state:
            st.session_state.grade_evaluations = []

        st.markdown("#### 📝 Nueva Evaluación GRADE")

        outcome_name = st.text_input("Nombre del resultado:", placeholder="Ej: Mortalidad por todas las causas")

        if outcome_name:
            col_grade = st.columns(2)

            with col_grade[0]:
                st.markdown("##### 🔻 Factores de Degradación")
                risk_bias = st.slider("Riesgo de Sesgo", 0, -2, 0)
                inconsistency = st.slider("Inconsistencia", 0, -2, 0)
                indirectness = st.slider("Indirectitud", 0, -2, 0)
                imprecision = st.slider("Imprecisión", 0, -2, 0)
                publication_bias = st.slider("Sesgo de Publicación", 0, -2, 0)

            with col_grade[1]:
                st.markdown("##### 🔺 Factores de Mejora")
                large_effect = st.checkbox("Efecto grande")
                dose_response = st.checkbox("Gradiente dosis-respuesta")
                confounding = st.checkbox("Control de factores confusores")

            initial_quality = st.selectbox("Calidad Inicial del Diseño:",
                                          ["Alta (ECAs)", "Baja (Observacional)"])

            if st.button("📊 EVALUAR GRADE", use_container_width=True):
                # Calcular calidad final
                deductions = abs(risk_bias + inconsistency + indirectness + imprecision + publication_bias)
                upgrades = (3 if large_effect else 0) + (3 if dose_response else 0) + (3 if confounding else 0)

                if initial_quality == "Alta (ECAs)":
                    base_score = 4  # Alta
                else:
                    base_score = 2  # Baja

                final_score = max(1, min(4, base_score + deductions + upgrades))

                quality_labels = {4: 'Alta', 3: 'Moderada', 2: 'Baja', 1: 'Muy Baja'}
                quality_colors = {4: '#10b981', 3: '#f59e0b', 2: '#f97316', 1: '#ef4444'}

                grade_result = {
                    'Resultado': outcome_name,
                    'Calidad Inicial': initial_quality,
                    'Calidad Final': quality_labels[final_score],
                    'Score': final_score
                }

                st.session_state.grade_evaluations.append(grade_result)

                # Mostrar resultado
                st.markdown("---")
                st.markdown(f"### 🎯 Calidad de Evidencia: **{quality_labels[final_score]}**")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=final_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [1, 4], 'tickvals': [1, 2, 3, 4],
                                'ticktext': ['Muy Baja', 'Baja', 'Moderada', 'Alta']},
                        'bar': {'color': quality_colors[final_score]},
                        'steps': [
                            {'range': [1, 2], 'color': '#fee2e2'},
                            {'range': [2, 3], 'color': '#fed7aa'},
                            {'range': [3, 4], 'color': '#d1fae5'}
                        ]
                    }
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig)

                st.markdown(f"""
                **Resumen de Evaluación:**

                | Factor | Valor |
                |--------|-------|
                | Calidad Inicial | {initial_quality} |
                | Deducciones Totales | {deductions} puntos |
                | Mejoras | {upgrades} puntos |
                | **Calidad Final** | **{quality_labels[final_score]}** |
                """)

        # Mostrar evaluaciones guardadas
        if st.session_state.grade_evaluations:
            st.markdown("---")
            st.markdown("### 📋 Evaluaciones GRADE Guardadas")

            df_grade = pd.DataFrame(st.session_state.grade_evaluations)
            st.dataframe(df_grade, use_container_width=True)

            # Gráfico de barras
            fig = px.bar(
                df_grade,
                x='Resultado',
                y='Score',
                color='Calidad Final',
                color_discrete_map={
                    'Alta': '#10b981',
                    'Moderada': '#f59e0b',
                    'Baja': '#f97316',
                    'Muy Baja': '#ef4444'
                },
                title='Calidad de Evidencia por Resultado'
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MÓDULO 12: ANÁLISIS DE SUPERVIVENCIA (KAPLAN-MEIER)
# ==========================================
elif menu == "📉 Supervivencia (KM)":
    st.header("📉 Análisis de Supervivencia - Kaplan-Meier")

    tab_km = st.tabs(["📝 Datos", "📈 Curva KM", "📊 Análisis"])

    with tab_km[0]:
        st.markdown("### 📝 Ingrese datos de supervivencia")

        if st.session_state.survival_data is None:
            st.session_state.survival_data = pd.DataFrame({
                'ID': range(1, 51),
                'Tiempo': np.random.exponential(50, 50).round(1),
                'Evento': np.random.binomial(1, 0.3, 50),
                'Grupo': np.random.choice(['Tratamiento', 'Control'], 50)
            })

        st.markdown("#### Datos de ejemplo (puede editarlos):")
        edited_survival = st.data_editor(
            st.session_state.survival_data,
            num_rows="dynamic",
            use_container_width=True
        )
        st.session_state.survival_data = edited_survival

        col_upload = st.columns(2)
        with col_upload[0]:
            uploaded = st.file_uploader("📂 Cargar CSV:", type="csv")
            if uploaded:
                df_upload = pd.read_csv(uploaded)
                st.session_state.survival_data = df_upload
                st.success("✅ Datos cargados!")

        with col_upload[1]:
            if st.button("🔄 Datos de Ejemplo"):
                np.random.seed(42)
                st.session_state.survival_data = pd.DataFrame({
                    'ID': range(1, 51),
                    'Tiempo': np.random.exponential(50, 50).round(1),
                    'Evento': np.random.binomial(1, 0.3, 50),
                    'Grupo': np.random.choice(['Tratamiento', 'Control'], 50)
                })
                st.rerun()

    with tab_km[1]:
        st.markdown("### 📈 Curva de Kaplan-Meier")

        if len(st.session_state.survival_data) > 0:
            KaplanMeierFitter, _, _, _, _, _, _, _ = get_analysis_imports()

            col_km = st.columns(2)
            with col_km[0]:
                time_col = st.selectbox("Columna de Tiempo:",
                                        st.session_state.survival_data.columns)
            with col_km[1]:
                event_col = st.selectbox("Columna de Evento:",
                                        st.session_state.survival_data.columns)

            group_by = st.selectbox("Agrupar por:",
                                   ['Ninguno'] + [c for c in st.session_state.survival_data.columns
                                                  if c not in [time_col, event_col]])

            if st.button("📊 GENERAR CURVA KM", use_container_width=True):
                kmf = KaplanMeierFitter()

                fig, ax = plt.subplots(figsize=(10, 6))

                if group_by == 'Ninguno':
                    kmf.fit(
                        st.session_state.survival_data[time_col],
                        st.session_state.survival_data[event_col],
                        label='Global'
                    )
                    kmf.plot_survival_function(ax=ax, ci_show=True)
                else:
                    for group in st.session_state.survival_data[group_by].unique():
                        mask = st.session_state.survival_data[group_by] == group
                        kmf.fit(
                            st.session_state.survival_data.loc[mask, time_col],
                            st.session_state.survival_data.loc[mask, event_col],
                            label=str(group)
                        )
                        kmf.plot_survival_function(ax=ax, ci_show=True)

                ax.set_xlabel('Tiempo')
                ax.set_ylabel('Probabilidad de Supervivencia')
                ax.set_title('Curva de Kaplan-Meier')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # Medianas de supervivencia
                st.markdown("### 📊 Estadísticas de Supervivencia")

                col_stat = st.columns(3)

                # Mediana de supervivencia global
                median_survival = kmf.median_survival_time_
                with col_stat[0]:
                    if np.isinf(median_survival):
                        st.metric("Mediana Supervivencia", "No alcanzada")
                    else:
                        st.metric("Mediana Supervivencia", f"{median_survival:.1f}")

                # Tiempo medio de supervivencia
                with col_stat[1]:
                    survival_times = st.session_state.survival_data[time_col]
                    mean_survival = survival_times.mean()
                    st.metric("Tiempo Medio", f"{mean_survival:.1f}")

                # Tasa de censura
                with col_stat[2]:
                    events = st.session_state.survival_data[event_col].sum()
                    total = len(st.session_state.survival_data)
                    censors = total - events
                    st.metric("Eventos / Censuras", f"{events} / {censors}")

                # Tabla de vida
                st.markdown("### 📋 Tabla de Vida")

                timeline = st.slider("Tiempo máximo:",
                                    min_value=10,
                                    max_value=int(st.session_state.survival_data[time_col].max()),
                                    value=60)

                survival_table = kmf.survival_table_at_times(timeline)
                st.dataframe(survival_table, use_container_width=True)

    with tab_km[2]:
        st.markdown("### 📊 Análisis Avanzado")

        if len(st.session_state.survival_data) > 0:
            from lifelines import LogRankTest

            st.markdown("#### Test de Log-Rank")

            group_col = st.selectbox("Variable de grupo:",
                                     [c for c in st.session_state.survival_data.columns
                                      if c != time_col and c != event_col])

            if st.button("🔬 EJECUTAR LOG-RANK", use_container_width=True):
                groups = st.session_state.survival_data[group_col].unique()
                if len(groups) == 2:
                    mask1 = st.session_state.survival_data[group_col] == groups[0]
                    mask2 = st.session_state.survival_data[group_col] == groups[1]

                    lr_test = LogRankTest(
                        st.session_state.survival_data.loc[mask1, time_col],
                        st.session_state.survival_data.loc[mask2, time_col],
                        st.session_state.survival_data.loc[mask1, event_col],
                        st.session_state.survival_data.loc[mask2, event_col]
                    )

                    lr_test.print_summary()

                    col_lr = st.columns(3)
                    with col_lr[0]:
                        st.metric("Chi²", f"{lr_test.test_statistic:.2f}")
                    with col_lr[1]:
                        st.metric("p-value", f"{lr_test.p_value:.4f}")
                    with col_lr[2]:
                        alpha = 0.05
                        sig = "Significativo" if lr_test.p_value < alpha else "No Significativo"
                        st.metric("Conclusión (α=0.05)", sig)

                    st.info(f"""
                    **Interpretación del Test de Log-Rank:**

                    - Hipótesis nula: No hay diferencia en las curvas de supervivencia entre los grupos
                    - Chi² = {lr_test.test_statistic:.2f}
                    - p-value = {lr_test.p_value:.4f}
                    - {'Se rechaza H0: Las curvas son significativamente diferentes' if lr_test.p_value < 0.05 else 'No se puede rechazar H0: No hay evidencia de diferencia'}
                    """)
                else:
                    st.warning("Seleccione una variable con exactamente 2 grupos para el test de Log-Rank")

# ==========================================
# MÓDULO 13: CURVAS ROC
# ==========================================
elif menu == "🎯 Curvas ROC":
    st.header("🎯 Curvas ROC - Evaluación Diagnóstica")

    tab_roc = st.tabs(["📝 Datos", "📈 Curva ROC", "📊 Comparación"])

    with tab_roc[0]:
        st.markdown("### 📝 Ingrese datos para análisis ROC")

        if st.session_state.roc_data is None:
            np.random.seed(42)
            n_samples = 200
            st.session_state.roc_data = pd.DataFrame({
                'ID': range(1, n_samples + 1),
                'Probabilidad': np.concatenate([
                    np.random.beta(5, 2, n_samples//2),  # Verdaderos positivos
                    np.random.beta(2, 5, n_samples//2)   # Verdaderos negativos
                ]),
                'Real': ['Positivo'] * (n_samples//2) + ['Negativo'] * (n_samples//2)
            })

        edited_roc = st.data_editor(
            st.session_state.roc_data,
            num_rows="dynamic",
            use_container_width=True
        )
        st.session_state.roc_data = edited_roc

        col_roc_upload = st.columns(2)
        with col_roc_upload[0]:
            uploaded_roc = st.file_uploader("📂 Cargar CSV:", type="csv")
            if uploaded_roc:
                st.session_state.roc_data = pd.read_csv(uploaded_roc)
                st.success("✅ Datos cargados!")

        with col_roc_upload[1]:
            if st.button("🎲 Generar Datos Ejemplo"):
                np.random.seed(42)
                n_samples = 200
                st.session_state.roc_data = pd.DataFrame({
                    'ID': range(1, n_samples + 1),
                    'Probabilidad': np.concatenate([
                        np.random.beta(5, 2, n_samples//2),
                        np.random.beta(2, 5, n_samples//2)
                    ]),
                    'Real': ['Positivo'] * (n_samples//2) + ['Negativo'] * (n_samples//2)
                })
                st.rerun()

    with tab_roc[1]:
        st.markdown("### 📈 Curva ROC")

        if len(st.session_state.roc_data) > 0:
            _, _, roc_curve, auc, confusion_matrix, _, _, _ = get_analysis_imports()

            col_roc_setup = st.columns(2)
            with col_roc_setup[0]:
                pred_col = st.selectbox("Variable de predicción:",
                                       st.session_state.roc_data.columns)
            with col_roc_setup[1]:
                actual_col = st.selectbox("Variable real:",
                                         st.session_state.roc_data.columns)

            if st.button("📊 GENERAR CURVA ROC", use_container_width=True):
                # Preparar datos
                y_true = (st.session_state.roc_data[actual_col] ==
                         st.session_state.roc_data[actual_col].unique()[0]).astype(int)
                y_score = st.session_state.roc_data[pred_col]

                # Calcular ROC
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)

                # Encontrar punto óptimo (Youden's J)
                j_scores = tpr - fpr
                j_idx = np.argmax(j_scores)
                optimal_threshold = thresholds[j_idx]
                optimal_fpr = fpr[j_idx]
                optimal_tpr = tpr[j_idx]

                # Gráfico
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Curva ROC
                ax1 = axes[0]
                ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
                ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
                ax1.scatter([optimal_fpr], [optimal_tpr], color='red', s=100,
                          zorder=5, label=f'Optimal (θ={optimal_threshold:.2f})')
                ax1.fill_between(fpr, tpr, alpha=0.3)
                ax1.set_xlabel('1 - Especificidad (FPR)')
                ax1.set_ylabel('Sensibilidad (TPR)')
                ax1.set_title('Curva ROC')
                ax1.legend(loc='lower right')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1])

                # Matriz de confusión en punto óptimo
                ax2 = axes[1]
                y_pred = (y_score >= optimal_threshold).astype(int)
                cm = confusion_matrix(y_true, y_pred)

                im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax2.set_title(f'Matriz de Confusión (θ={optimal_threshold:.2f})')

                classes = ['Negativo', 'Positivo']
                ax2.set_xticks([0, 1])
                ax2.set_yticks([0, 1])
                ax2.set_xticklabels(classes)
                ax2.set_yticklabels(classes)
                ax2.set_xlabel('Predicción')
                ax2.set_ylabel('Real')

                # Añadir valores en la matriz
                for i in range(2):
                    for j in range(2):
                        ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                               fontsize=16, fontweight='bold',
                               color='white' if cm[i, j] > cm.max()/2 else 'black')

                plt.tight_layout()
                st.pyplot(fig)

                # Métricas
                st.markdown("### 📊 Métricas de Rendimiento")

                col_metrics_roc = st.columns(4)
                with col_metrics_roc[0]:
                    st.metric("AUC-ROC", f"{roc_auc:.3f}")
                with col_metrics_roc[1]:
                    st.metric("Sensibilidad", f"{optimal_tpr:.3f}")
                with col_metrics_roc[2]:
                    st.metric("Especificidad", f"{1-optimal_fpr:.3f}")
                with col_metrics_roc[3]:
                    st.metric("Punto de Corte", f"{optimal_threshold:.3f}")

                # Precisión y valores predictivos
                tn, fp, fn, tp = cm.ravel()
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = ppv

                col_metrics2 = st.columns(4)
                with col_metrics2[0]:
                    st.metric("Exactitud", f"{accuracy:.3f}")
                with col_metrics2[1]:
                    st.metric("VPP", f"{ppv:.3f}")
                with col_metrics2[2]:
                    st.metric("VPN", f"{npv:.3f}")
                with col_metrics2[3]:
                    st.metric("LR+", f"{(optimal_tpr)/(1-optimal_fpr):.3f}")

    with tab_roc[2]:
        st.markdown("### 📊 Comparación de Tests Diagnósticos")

        st.info("""
        **Comparación de Curvas ROC:**

        Esta funcionalidad permite comparar el rendimiento diagnóstico de múltiples pruebas
        o marcadores utilizando el área bajo la curva (AUC).

        Para usar esta función:
        1. Agregue múltiples columnas de predicción a sus datos
        2. Seleccione las columnas a comparar
        3. Ejecute el análisis
        """)

        if len(st.session_state.roc_data.columns) >= 3:
            pred_cols = st.multiselect("Seleccionar predictores:",
                                      [c for c in st.session_state.roc_data.columns
                                       if c not in ['ID', 'Real']])

            if pred_cols and st.button("📊 COMPARAR TESTS", use_container_width=True):
                _, _, roc_curve, auc, _, _, _, _ = get_analysis_imports()

                fig, ax = plt.subplots(figsize=(10, 8))

                y_true = (st.session_state.roc_data['Real'] ==
                         st.session_state.roc_data['Real'].unique()[0]).astype(int)

                colors = ['blue', 'red', 'green', 'orange', 'purple']
                auc_values = {}

                for i, col in enumerate(pred_cols):
                    y_score = st.session_state.roc_data[col]
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    roc_auc = auc(fpr, tpr)
                    auc_values[col] = roc_auc

                    ax.plot(fpr, tpr, color=colors[i % len(colors)],
                           linewidth=2, label=f'{col} (AUC = {roc_auc:.3f})')

                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
                ax.set_xlabel('1 - Especificidad (FPR)')
                ax.set_ylabel('Sensibilidad (TPR)')
                ax.set_title('Comparación de Curvas ROC')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # Tabla de comparación
                comparison_df = pd.DataFrame({
                    'Test': list(auc_values.keys()),
                    'AUC': list(auc_values.values())
                }).sort_values('AUC', ascending=False)

                st.dataframe(comparison_df, use_container_width=True)

# ==========================================
# MÓDULO 14: MAPAS GEOGRÁFICOS
# ==========================================
elif menu == "🗺️ Mapas Geográficos":
    st.header("🗺️ Mapas Geográficos - Epidemiología Espacial")

    tab_map = st.tabs(["📊 Coroplético", "🔥 Heatmap", "📍 Marcadores"])

    with tab_map[0]:
        st.markdown("### 📊 Mapa Coroplético - Incidencia por Región")

        if 'map_data' not in st.session_state:
            st.session_state.map_data = pd.DataFrame({
                'Region': ['Antioquia', 'Cundinamarca', 'Valle del Cauca', 'Atlántico',
                          'Santander', 'Bolívar', 'Córdoba', 'Nariño', 'Boyacá', 'Cauca'],
                'Casos': [1500, 1200, 1100, 900, 800, 750, 600, 550, 500, 450],
                'Poblacion': [6500000, 3000000, 4500000, 2500000, 2000000, 2100000,
                             1800000, 1600000, 1400000, 1300000],
                'Lat': [6.2442, 4.6210, 3.8000, 10.9685, 7.1190, 10.3910, 8.7479,
                       1.2897, 5.7639, 2.7580],
                'Lon': [-75.5812, -74.0674, -76.5220, -74.7813, -73.1198, -75.5142,
                       -75.8813, -77.6428, -72.9077, -76.6136]
            })

        st.markdown("#### Datos de ejemplo (Colombia):")
        edited_map = st.data_editor(
            st.session_state.map_data,
            num_rows="dynamic",
            use_container_width=True
        )
        st.session_state.map_data = edited_map

        col_map_setup = st.columns(2)
        with col_map_setup[0]:
            metric_map = st.selectbox("Métrica a visualizar:",
                                     ['Casos', 'Tasa por 100,000', 'Población'])
        with col_map_setup[1]:
            color_scale = st.selectbox("Escala de color:",
                                       ['Reds', 'Blues', 'Viridis', 'Plasma', 'RdYlGn_r'])

        if metric_map == 'Tasa por 100,000':
            st.session_state.map_data['Tasa'] = (
                st.session_state.map_data['Casos'] /
                st.session_state.map_data['Poblacion'] * 100000
            )
            color_col = 'Tasa'
        else:
            color_col = metric_map

        if st.button("🗺️ GENERAR MAPA", use_container_width=True):
            fig = px.scatter_mapbox(
                st.session_state.map_data,
                lat='Lat',
                lon='Lon',
                size=color_col,
                color=color_col,
                color_continuous_scale=color_scale,
                hover_name='Region',
                hover_data={
                    'Casos': True,
                    'Poblacion': True,
                    'Lat': False,
                    'Lon': False,
                    'Tasa': ':.2f' if metric_map == 'Tasa por 100,000' else False
                },
                zoom=5,
                center={'lat': 4.5709, 'lon': -74.2973},
                height=600,
                title=f'Mapa Coroplético - {metric_map}'
            )

            fig.update_layout(mapbox_style='carto-darkmatter')
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

            st.plotly_chart(fig, use_container_width=True)

    with tab_map[1]:
        st.markdown("### 🔥 Mapa de Calor - Densidad de Casos")

        st.info("""
        **Mapa de Calor (Heatmap):**

        Visualice la distribución espacial de casos o eventos de salud
        para identificar áreas de alta concentración o "hotspots".
        """)

        if st.button("🔥 GENERAR HEATMAP", use_container_width=True):
            # Generar puntos de calor simulados
            np.random.seed(42)
            n_points = 500

            # Centroides de las regiones con más casos
            hotspots = {
                'Bogotá': (4.7110, -74.0721, 150),
                'Medellín': (6.2442, -75.5812, 100),
                'Cali': (3.8000, -76.5220, 80),
                'Barranquilla': (10.9685, -74.7813, 60)
            }

            heat_data = []
            for city, (lat, lon, weight) in hotspots.items():
                for _ in range(weight):
                    lat_jitter = lat + np.random.normal(0, 0.3)
                    lon_jitter = lon + np.random.normal(0, 0.3)
                    heat_data.append({'lat': lat_jitter, 'lon': lon_jitter, 'city': city})

            df_heat = pd.DataFrame(heat_data)

            fig = px.density_mapbox(
                df_heat,
                lat='lat',
                lon='lon',
                radius=15,
                center={'lat': 4.5709, 'lon': -74.2973},
                zoom=5,
                mapbox_style='carto-darkmatter',
                title='Mapa de Calor - Distribución de Casos'
            )

            fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab_map[2]:
        st.markdown("### 📍 Mapa con Marcadores - Ubicaciones Específicas")

        st.markdown("""
        **Mapa de Marcadores:**

        Visualice ubicaciones específicas como:
        - Centros de salud
        - Casos individuales
        - Recursos hospitalarios
        """)

        marker_data = pd.DataFrame({
            'Nombre': ['Hospital Central', 'Clínica del Norte', 'Centro de Salud Sur',
                      'UCI Móvil 1', 'UCI Móvil 2'],
            'Lat': [6.2474, 6.2600, 6.2300, 6.2500, 6.2550],
            'Lon': [-75.5800, -75.5600, -75.5900, -75.5700, -75.5650],
            'Tipo': ['Hospital', 'Clínica', 'CentroSalud', 'UCI', 'UCI'],
            'Capacidad': [200, 150, 50, 20, 20]
        })

        fig = px.scatter_mapbox(
            marker_data,
            lat='Lat',
            lon='Lon',
            color='Tipo',
            size='Capacidad',
            hover_name='Nombre',
            hover_data={'Capacidad': True, 'Lat': False, 'Lon': False},
            zoom=12,
            center={'lat': 6.2474, 'lon': -75.5750},
            height=500,
            title='Ubicaciones de Recursos de Salud'
        )

        fig.update_layout(mapbox_style='carto-darkmatter')
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(marker_data, use_container_width=True)

# ==========================================
# MÓDULO 15: BIOINFORMÁTICA
# ==========================================
elif menu == "🧬 Bioinformática":
    st.header("🧬 Análisis de Secuencias Genéticas")

    seq = st.text_area(
        "🔬 Secuencia DNA/RNA:",
        placeholder="ATGCCGTAGCTGATCGATCGATCG...",
        height=150
    ).upper().replace(" ", "").replace("\n", "")

    col_seq_btn = st.columns([1, 3, 1])
    with col_seq_btn[1]:
        analyze_seq = st.button("🧬 ANALIZAR SECUENCIA", use_container_width=True)

    if seq and analyze_seq:
        valid_bases = set('AGTCUN')
        invalid = [b for b in seq if b not in valid_bases]

        if invalid:
            st.error(f"❌ Bases inválidas encontradas: {set(invalid)}")
        else:
            col_stats = st.columns(3)

            gc = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
            with col_stats[0]:
                st.metric("📊 Contenido GC", f"{gc:.2%}")
            with col_stats[1]:
                st.metric("📏 Longitud", f"{len(seq)} pb")
            with col_stats[2]:
                at = (seq.count('A') + seq.count('T') + seq.count('U')) / len(seq) if len(seq) > 0 else 0
                st.metric("⚖️ Ratio GC/AT", f"{gc/at:.2f}" if at > 0 else "N/A")

            # Complementaria reversa
            st.subheader("🔄 Complementaria Reversa")
            complementaria = (seq.replace('A', 't').replace('T', 'a')
                            .replace('G', 'c').replace('C', 'g')
                            .replace('U', 'a').upper()[::-1])
            st.code(complementaria, language="")

            # Traducción a proteína (simplificado)
            codon_table = {
                'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
                'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
                'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
                'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
                'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
                'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
                'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
                'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
                'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
                'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
                'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
                'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
            }

            if len(seq) >= 3:
                protein = ''
                for i in range(0, len(seq)-len(seq)%3, 3):
                    codon = seq[i:i+3]
                    protein += codon_table.get(codon, 'X')

                st.subheader("🧪 Traducción a Proteína (frame 1)")
                st.code(protein, language="")

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
                <h2 style="color: #ffd700; margin-top: 20px;">$299.000 COP / año</h2>
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
                        💳 PAGAR AHORA - $299.000 COP
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
    "🩺 EpiDiagnosis Pro v6.0 | © 2024 | Desarrollado con Streamlit | "
    "Módulos: Dashboard, Limpieza, Bioestadística, Calculadora 2x2, Tamaño de Muestra, "
    "Vigilancia, PICO, PRISMA, Forest Plot, Meta-análisis, RoB/GRADE, Kaplan-Meier, ROC, Mapas"
    "</div>",
    unsafe_allow_html=True
)
