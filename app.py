import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la página y Estética "Lux"
st.set_page_config(page_title="LuxLogistics DaaS", page_icon="💡")

st.title("💡 LuxLogistics DaaS")
st.subheader("Sistema de Alerta Temprana de Riesgo Logístico")
st.markdown("---")

# 2. Cargar el "Cerebro" de LuxLogistics
@st.cache_resource
def cargar_modelo():
    # Asegúrate de que el nombre del archivo coincida exactamente con el de tu carpeta
    return joblib.load('modelo_luxlogistics.joblib')

model = cargar_modelo()

# --- DICCIONARIO DE JERARQUÍA GEOGRÁFICA ---
# Esta es la inteligencia geográfica que detectaste. 
# En este dataset, México (Toluca) se encuentra en Market: LATAM / Region: Central America.
regiones_por_mercado = {
    'LATAM': ['Central America', 'South America', 'Caribbean'],
    'USCA': ['USCA'],
    'Europe': ['Western Europe', 'Southern Europe', 'Northern Europe', 'Eastern Europe'],
    'Africa': ['North Africa', 'East Africa', 'West Africa', 'Central Africa', 'Southern Africa'],
    'Pacific Asia': ['Southeast Asia', 'South Asia', 'Oceania', 'Eastern Asia', 'Western Asia', 'Central Asia']
}

# 3. Interfaz de Usuario (Entradas de datos en el Sidebar)
st.sidebar.header("Configuración del Envío")
st.sidebar.write("Defina los parámetros del pedido.")

# Selección Dinámica de Mercado y Región
market = st.sidebar.selectbox("Mercado de Destino", list(regiones_por_mercado.keys()))

# La lista de regiones se filtra automáticamente según el mercado elegido
lista_regiones_filtrada = regiones_por_mercado[market]
order_region = st.sidebar.selectbox("Región del Orden", lista_regiones_filtrada)

# Resto de variables categóricas
type_envio = st.sidebar.selectbox("Tipo de Pago", ['CASH', 'PAYMENT', 'DEBIT', 'TRANSFER'])
shipping_mode = st.sidebar.selectbox("Modo de Envío", ['Standard Class', 'First Class', 'Second Class', 'Same Day'])
customer_segment = st.sidebar.selectbox("Segmento de Cliente", ['Consumer', 'Corporate', 'Home Office'])

# Variables Numéricas y de Tiempo
sales = st.sidebar.number_input("Valor de la Venta (USD)", min_value=1.0, value=150.0)
order_month = st.sidebar.slider("Mes del Pedido", 1, 12, 6)
order_day = st.sidebar.slider("Día de la Semana (0=Lun, 6=Dom)", 0, 6, 2)

# 4. Lógica de Predicción
if st.button("Evaluar Riesgo de Retraso"):
    # Creamos el DataFrame con las 8 columnas en el orden EXACTO del modelo
    input_data = pd.DataFrame([[
        type_envio, 
        market, 
        shipping_mode, 
        customer_segment, 
        order_region, 
        sales, 
        order_month, 
        order_day
    ]], columns=['Type', 'Market', 'Shipping Mode', 'Customer Segment', 'Order Region', 'Sales', 'order_month', 'order_day_of_week'])
    
    try:
        # Realizamos la predicción con el modelo honesto
        prediccion = model.predict(input_data)[0]
        probabilidad = model.predict_proba(input_data)[0][1]

        # 5. Visualización de Resultados con Identidad Corporativa
        st.markdown("### Resultado del Análisis Predictivo")
        
        if prediccion == 1:
            st.error(f"⚠️ ALTO RIESGO DE RETRASO (Probabilidad: {probabilidad:.2%})")
            st.info("💡 Sugerencia de LuxLogistics: Considere cambiar el modo de envío o revisar la prioridad de despacho para este valor de venta.")
        else:
            st.success(f"✅ ENVÍO SEGURO (Probabilidad de retraso: {probabilidad:.2%})")
            st.write("El pedido se encuentra dentro de los parámetros de cumplimiento histórico.")
            
    except Exception as e:
        st.error(f"Se detectó un error técnico en el motor: {e}")

st.markdown("---")
st.caption("LuxLogistics DaaS - Iluminando la ruta de tus decisiones | Toluca-Lerma, México.")