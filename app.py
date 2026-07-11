import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai
import re 
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Configuración de la página y Estética "InsurTrust"
st.set_page_config(page_title="InsurTrust AI", page_icon="🛡️", layout="wide")

st.title("🛡️ InsurTrust AI: Inteligencia en Suscripción Médica")
st.subheader("Sistema Agéntico de Evaluación de Riesgo para Vida y Salud")
st.markdown("---")

# 2. Cargar el "Cerebro" de Predicción (Machine Learning)
@st.cache_resource
def cargar_modelo():
    ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo_insurtrust.joblib')
    return joblib.load(ruta_modelo)

try:
    model = cargar_modelo()
except:
    st.error("❌ No se encontró el archivo 'modelo_insurtrust.joblib'.")

# 3. Cargar los Datos para el Agente de IA (Pandas)
@st.cache_data
def cargar_datos_csv():
    # Usamos ruta absoluta para máxima seguridad
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(ruta_base, 'medical_insurance_data.csv')
    
    if os.path.exists(ruta_archivo):
        return pd.read_csv(ruta_archivo)
    else:
        return None

df_ia = cargar_datos_csv()

# --- SIDEBAR (Entradas para el suscritor) ---
st.sidebar.header("Perfil del Asegurado")
age = st.sidebar.number_input("Edad", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sexo", ['female', 'male'])
bmi = st.sidebar.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.selectbox("Número de Hijos", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("¿Es fumador?", ['yes', 'no'])
region = st.sidebar.selectbox("Región", ['southwest', 'southeast', 'northwest', 'northeast'])

# --- INTERFAZ: PESTAÑAS (TABS) ---
tab_prediccion, tab_chat = st.tabs(["🔮 Simulador de Riesgo", "🤖 InsurTrust Smart Chat"])

with tab_prediccion:
    st.header("🔍 Evaluación de Riesgo de Suscripción Médica")
    if st.button("Ejecutar Dictaminación de Riesgo"):
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                                 columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        try:
            prediccion = model.predict(input_data)[0]
            probabilidad = model.predict_proba(input_data)[0][1]
            st.markdown("### 📊 Diagnóstico Actuarial Inteligente")
            if prediccion == 1:
                st.error(f"⚠️ ALTO RIESGO DE SINIESTRALIDAD (Probabilidad: {probabilidad:.2%})")
                st.info("📋 **Sugerencia:** Se recomienda solicitar exámenes médicos o aplicar extraprima.")
            else:
                st.success(f"✅ RIESGO ESTÁNDAR (Confianza: {(1-probabilidad):.2%})")
                st.info("💡 **Sugerencia:** El perfil es elegible para emisión inmediata.")
        except Exception as e:
            st.error(f"Error en el motor: {e}")
            
with tab_chat:
    st.header("🤖 InsurTrust Smart Chat (AXA Intelligence)")
    
    if df_ia is not None:
        # --- LÓGICA DE SEGURIDAD DE API KEY ---
        user_api_key = None
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                user_api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            pass

        if not user_api_key:
            user_api_key = st.text_input("Introduce tu Google API Key:", type="password")

        if user_api_key:
            pregunta = st.text_input("Hazle una pregunta estratégica a la base de datos de AXA:")
            if pregunta:
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=user_api_key, temperature=0)
                    
                    prefix = """
                    You are a Senior Medical Underwriter at AXA. The dataframe name is 'df'.

                    STRATEGIC CONTEXT (Talia's Research):
                    - Global average charges: $13,270.42.
                    - Atypical Segment (1.42%): Healthy people (smoker='no', bmi < 25) with charges > 13270.
                    - Key Finding: Even healthy profiles can reach $30k+ in charges (Case Index 62).
                    - Why it is a 'Fuga de Rentabilidad': Because standard underwriting rules accept these cases 
                      with low premiums, but they represent a high technical loss ratio for AXA.

                    TECHNICAL GUIDANCE:
                    - To analyze atypical cases, use: df[(df['smoker'] == 'no') & (df['bmi'] < 25) & (df['charges'] > 13270)]

                    RULES: 
                    1. Always start with 'Thought:'. 
                    2. Use 'Action: python_repl_ast'. 
                    3. Then 'Action Input:'. 
                    4. ALWAYS provide a 'Final Answer:' in SPANISH. 
                    5. Use professional insurance terms: 'Siniestralidad Atípica', 'Extraprima', 'Underwriting Leakage'.
                    """

                    agent = create_pandas_dataframe_agent(
                        llm, df_ia, verbose=True, allow_dangerous_code=True,
                        handle_parsing_errors="Check format: Thought/Action/Action Input/Final Answer",
                        max_iterations=15, agent_type="zero-shot-react-description", prefix=prefix
                    )

                    with st.spinner("InsurTrust AI analizando..."):
                        res_final = ""
                        try:
                            resultado = agent.invoke(pregunta)
                            res_final = resultado['output']
                        except Exception as parse_err:
                            error_str = str(parse_err)
                            if "429" in error_str or "quota" in error_str.lower():
                                res_final = "⚠️ Límite de cuota alcanzado. Espera 1 minuto."
                            elif "Final Answer:" in error_str:
                                res_final = error_str.split("Final Answer:")[-1]
                            elif "Could not parse LLM output: `" in error_str:
                                res_final = error_str.split("Could not parse LLM output: `")[-1]
                            else:
                                res_final = error_str

                        # Limpieza de links
                        for basura in ["For troubleshooting", "visit:", "https://", "Agent stopped"]:
                            res_final = res_final.split(basura)[0]
                        res_final = res_final.replace("`", "").strip()

                        if len(res_final) > 0:
                            st.success("✅ Análisis Completado")
                            st.markdown(f"### {res_final}")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.error("❌ El archivo 'medical_insurance_data.csv' no fue encontrado. Verifica la ruta.")

st.markdown("---")
st.caption("InsurTrust AI - Inteligencia Actuarial | Desarrollado por Talia González López")