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
    # Buscamos el modelo de seguros específicamente
    ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo_insurtrust.joblib')
    return joblib.load(ruta_modelo)

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"❌ Error al cargar el modelo actuarial: {e}")

# 3. Cargar los Datos para el Agente de IA (Pandas)
@st.cache_data
def cargar_datos_csv():
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
    st.info("Utilice el panel de la izquierda para configurar los parámetros del solicitante.")
    
    if st.button("Ejecutar Dictaminación de Riesgo"):
        input_data = pd.DataFrame([[
            age, sex, bmi, children, smoker, region 
        ]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        
        try:
            prediccion = model.predict(input_data)[0]
            probabilidad = model.predict_proba(input_data)[0][1]

            st.markdown("### 📊 Diagnóstico Actuarial Inteligente")

            if prediccion == 1:
                st.error(f"⚠️ ALTO RIESGO DE SINIESTRALIDAD (Probabilidad: {probabilidad:.2%})")
                st.info("📋 **Sugerencia de InsurTrust:** Se recomienda solicitar exámenes médicos de laboratorio o aplicar una extraprima por factores de riesgo detectados.")
            else:
                st.success(f"✅ RIESGO ESTÁNDAR DETECTADO (Confianza: {(1-probabilidad):.2%})")
                st.info("💡 **Sugerencia de InsurTrust:** El perfil es elegible para **emisión inmediata** con tarifa preferente. No se requieren requisitos médicos adicionales.")
                st.write("El perfil cumple con los parámetros de suscripción estándar.")
        except Exception as e:
            st.error(f"Error en el motor: {e}")
            
with tab_chat:
    st.header("🤖 InsurTrust Smart Chat (AXA Intelligence)")
    st.markdown("Analice la siniestralidad y el riesgo de suscripción en lenguaje natural.")
    
    if df_ia is not None:
        # --- LÓGICA DE SEGURIDAD DE API KEY ---
        user_api_key = None
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                user_api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            pass

        if not user_api_key:
            user_api_key = st.text_input("Introduce tu Google API Key para activar el Oráculo:", type="password")

        if user_api_key:
            pregunta = st.text_input("Hazle una pregunta estratégica a la base de datos de AXA:")
            
            if pregunta:
                res_final = "" # Limpieza de pizarra, inicialización de seguridad
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=user_api_key, temperature=0)
                    
                    # PREFIX MAESTRO (Versión Talia)
                    prefix = """
                    You are a Senior Medical Underwriter at AXA. The dataframe name is 'df'.
                    
                    STRATEGIC CONTEXT (Talia's Research):
                    - Global average charges: $13,270.42.
                    - Atypical Segment (1.42%): 19 healthy people with charges > $13,270.
                    - Key Finding: Even healthy profiles can reach $30k+ in charges (Case Index 62).
                    - 'Fuga de Rentabilidad': Cases like Index 62 ($30k+) are accepted with low premiums but have huge medical costs, draining AXA's margin.
                    FORMAT RULES:
                    1. Use Thought/Action/Action Input/Observation cycle.
                    2. ALWAYS provide a 'Final Answer:' in SPANISH.
                    3. Use terms: 'Siniestralidad Atípica', 'Underwriting Leakage'.
                    """

                    agent = create_pandas_dataframe_agent(
                        llm, df_ia, verbose=True, allow_dangerous_code=True,
                        max_iterations=15, agent_type="zero-shot-react-description", prefix=prefix
                    )

                    with st.spinner("InsurTrust AI analizando..."):
                        res_final = ""
                        try:
                            resultado = agent.invoke({"input": pregunta}, {"handle_parsing_errors": True})
                            res_final = resultado['output']
                        except Exception as parse_err:
                            # --- ESCUDO DE RESCATE ---
                            error_str = str(parse_err)
                            res_final = ""
                            # ¿Es un problema de cuota/dinero?
                            if "429" in error_str or "quota" in error_str.lower():
                                st.warning("⚠️ Límite de Google alcanzado. Espera 1 minuto para que se resetee la cuota.")
                                st.stop()
                            if "503" in error_str:
                                st.warning("🌐 **InsurTrust AI: Alta Demanda en el Servidor.**")
                                st.info("Estamos experimentando latencia en los nodos de Google. Por favor, reintente en 30 segundos. En una implementación Enterprise (Vertex AI), este pico se elimina con capacidad reservada.")
                                st.stop()
                            # Intentamos rescatar la respuesta si está ahí
                           
                            if "Final Answer:" in error_str:
                                res_final = error_str.split("Final Answer:")[-1]
                            elif "Could not parse LLM output: `" in error_str:
                                res_final = error_str.split("Could not parse LLM output: `")[-1]
                            else:
                                # 3. SI NO HAY ETIQUETAS, MOSTRAR EL ERROR CRUDO
                                # Esto evita que salga el cuadro rojo de "ningún dato"
                                res_final = error_str

                        # --- LA GUILLOTINA DE LINKS (Limpiamos solo si hay link técnico) ---
                        for basura in ["For troubleshooting", "visit:", "https://","Agent stopped"]:
                            if basura in res_final:
                                res_final = res_final.split(basura)[0]
                        
                        res_final = res_final.strip(" `*").replace("\n ", "\n")

                        # --- MOSTRAR RESULTADO ---
                        if len(res_final) > 0:
                            st.success("✅ Análisis Completado (Extracción Cruda)")
                            st.write(res_final) # Usamos write para que se vea todo    

                        else:
                            st.error("La IA no respondió nada. Revisa si tu API Key sigue activa.")

                except Exception as e:
                    st.error(f"Error técnico en el motor: {e}")
    else:
        st.error("❌ No se encontró el archivo 'medical_insurance_data.csv'.")

st.markdown("---")
st.caption("InsurTrust AI - Inteligencia Actuarial | Desarrollado por Talia González López")