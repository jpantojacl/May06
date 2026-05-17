import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.title("Predicción de Enfermedad Cardíaca ❤️")
st.write("Ingresa la información del paciente y elige un modelo para obtener una predicción sencilla y fácil de entender.")

# -----------------------------
# Función para cargar modelos desde GitHub
# -----------------------------
def cargar_modelo(url):
    response = requests.get(url)

    if response.status_code != 200:
        st.error("❌ No se pudo descargar el modelo. Revisa el enlace RAW en GitHub.")
        st.stop()

    try:
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error("❌ El archivo del modelo no es válido o está dañado.")
        st.write(e)
        st.stop()

# -----------------------------
# URLs RAW de tus modelos
# -----------------------------
url_lr = "https://raw.githubusercontent.com/jpantojacd/May06/main/model1/modelo_regresion_logistica.pkl"
url_rf = "https://raw.githubusercontent.com/jpantojacd/May06/main/model1/modelo_random_forest.pkl"

modelo_lr = cargar_modelo(url_lr)
modelo_rf = cargar_modelo(url_rf)

# -----------------------------
# Inputs amigables
# -----------------------------
st.subheader("Información del paciente")

age = st.slider("Edad", 18, 100, 50)
sex = st.radio("Sexo", [0, 1], format_func=lambda x: "Mujer" if x == 0 else "Hombre")

st.subheader("Datos clínicos")

cp = st.selectbox(
    "Tipo de dolor en el pecho",
    [0, 1, 2, 3],
    format_func=lambda x: [
        "Angina típica (dolor fuerte)",
        "Angina atípica (dolor moderado)",
        "Dolor no anginoso",
        "Asintomático (sin dolor)"
    ][x]
)

trestbps = st.number_input("Presión arterial en reposo (mm Hg)", 80, 200, 120)
chol = st.number_input("Colesterol total (mg/dl)", 100, 600, 200)
fbs = st.radio("¿Azúcar en sangre alta en ayunas?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

restecg = st.selectbox(
    "Resultado del electrocardiograma",
    [0, 1, 2],
    format_func=lambda x: [
        "Normal",
        "Anormalidad ST-T",
        "Hipertrofia ventricular izquierda"
    ][x]
)

thalach = st.slider("Frecuencia cardíaca máxima alcanzada", 60, 250, 150)
exang = st.radio("¿Angina inducida por ejercicio?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
oldpeak = st.slider("Depresión ST por ejercicio", 0.0, 10.0, 1.0, 0.1)

slope = st.selectbox(
    "Pendiente del segmento ST",
    [0, 1, 2],
    format_func=lambda x: [
        "Ascendente (mejora)",
        "Plano",
        "Descendente (empeora)"
    ][x]
)

ca = st.selectbox("Número de vasos coloreados", [0, 1, 2, 3, 4])
thal = st.selectbox(
    "Resultado Thal",
    [1, 2, 3],
    format_func=lambda x: [
        "Normal",
        "Defecto fijo",
        "Defecto reversible"
    ][x]
)

# Crear dataframe con los datos ingresados
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# -----------------------------
# Selección del modelo
# -----------------------------
st.subheader("Modelo de predicción")
modelo_seleccionado = st.selectbox(
    "Elige el modelo que deseas usar:",
    ["Regresión Logística (más simple)", "Random Forest (más preciso)"]
)

# -----------------------------
# Botón de predicción
# -----------------------------
if st.button("Obtener resultado"):
    if modelo_seleccionado.startswith("Regresión"):
        pred = modelo_lr.predict(input_data)[0]
    else:
        pred = modelo_rf.predict(input_data)[0]

    st.subheader("Resultado de la predicción")

    if pred == 1:
        st.error("⚠️ El modelo indica **riesgo de enfermedad cardíaca**.\n\nTe recomendamos consultar con un profesional de salud.")
    else:
        st.success("✅ El modelo indica **bajo riesgo de enfermedad cardíaca**.\n\nMantén hábitos saludables y controles regulares.")

    st.info("Esta herramienta es solo una ayuda basada en datos. No reemplaza una evaluación médica real.")
