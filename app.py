import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la página y Estética "Lux"
st.set_page_config(page_title="LuxLogistics DaaS", page_icon="💡")

st.title("💡 LuxLogistics DaaS")
st.subheader("Sistema de Alerta Temprana de Riesgo Logístico")
st.markdown("---")

# 2. Cargar el "Cerebro"
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_luxlogistics.joblib')

model = cargar_modelo()

# 3. Interfaz de Usuario (Entradas de datos)
st.sidebar.header("Configuración del Envío")
st.sidebar.write("Ingrese los detalles del pedido para evaluar el riesgo.")

# --- ENTRADAS DE DATOS (Asegurando que coincidan con el entrenamiento) ---
type_envio = st.sidebar.selectbox("Tipo de Pago", ['CASH', 'PAYMENT', 'DEBIT', 'TRANSFER'])
market = st.sidebar.selectbox("Mercado de Destino", ['Pacific Asia', 'USCA', 'Africa', 'Europe', 'LATAM'])
shipping_mode = st.sidebar.selectbox("Modo de Envío", ['Standard Class', 'First Class', 'Second Class', 'Same Day'])

# --- ESTAS SON LAS DOS QUE FALTABAN ---
customer_segment = st.sidebar.selectbox("Segmento de Cliente", ['Consumer', 'Corporate', 'Home Office'])
order_region = st.sidebar.selectbox("Región del Orden", [
    'Southeast Asia', 'South Asia', 'Oceania', 'Central America', 'Caribbean', 
    'South America', 'East Africa', 'Western Europe', 'Northern Europe', 'Southern Europe'
])

sales = st.sidebar.number_input("Valor de la Venta (USD)", min_value=1.0, value=150.0)
order_month = st.sidebar.slider("Mes del Pedido", 1, 12, 6)
order_day = st.sidebar.slider("Día de la Semana (0=Lun, 6=Dom)", 0, 6, 2)

# 4. Lógica de Predicción
if st.button("Evaluar Riesgo de Retraso"):
    # Creamos el dataframe con las 8 columnas EXACTAS que el modelo espera
    input_data = pd.DataFrame([[
        type_envio, 
        market, 
        shipping_mode, 
        customer_segment, # Añadida
        order_region,     # Añadida
        sales, 
        order_month, 
        order_day
    ]], columns=['Type', 'Market', 'Shipping Mode', 'Customer Segment', 'Order Region', 'Sales', 'order_month', 'order_day_of_week'])
    
    # Hacemos la predicción
    try:
        prediccion = model.predict(input_data)[0]
        probabilidad = model.predict_proba(input_data)[0][1]

        # 5. Mostrar Resultados con el estilo de LuxLogistics
        st.markdown("### Resultado del Análisis")
        
        if prediccion == 1:
            st.error(f"⚠️ ALTO RIESGO DE RETRASO (Probabilidad: {probabilidad:.2%})")
            st.write("Se recomienda revisar la ruta y el modo de envío para mitigar el impacto.")
        else:
            st.success(f"✅ ENVÍO SEGURO (Probabilidad de retraso: {probabilidad:.2%})")
            st.write("El pedido cumple con los parámetros de eficiencia logística.")
            
    except Exception as e:
        st.error(f"Error en la predicción: {e}")

st.markdown("---")
st.caption("LuxLogistics DaaS - Iluminando la ruta de tus decisiones | Toluca-Lerma, México.")