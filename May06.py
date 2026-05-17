import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.title("Predicción de Enfermedad Cardíaca ❤️")
st.write("Complete la información del paciente y seleccione un modelo para obtener una predicción.")

# -----------------------------
# Cargar modelos desde GitHub
# -----------------------------

def cargar_modelo(url):
    response = requests.get(url)
    return joblib.load(BytesIO(response.content))

url_lr = "https://raw.githubusercontent.com/tu_usuario/tu_repo/main/modelos/modelo_regresion_logistica.pkl"
url_rf = "https://raw.githubusercontent.com/tu_usuario/tu_repo/main/modelos/modelo_random_forest.pkl"

modelo_lr = cargar_modelo(url_lr)
modelo_rf = cargar_modelo(url_rf)

# -----------------------------
# Inputs amigables
# -----------------------------

st.subheader("Datos personales")
age = st.slider("Edad", 18, 100, 50)
sex = st.radio("Sexo", options=[0, 1], format_func=lambda x: "Mujer" if x == 0 else "Hombre")

st.subheader("Información clínica")
cp = st.selectbox(
    "Tipo de dolor en el pecho",
    options=[0, 1, 2, 3],
    format_func=lambda x: [
        "0: Angina típica",
        "1: Angina atípica",
        "2: Dolor no anginoso",
        "3: Asintomático"
    ][x]
)

trestbps = st.number_input("Presión arterial en reposo (mm Hg)", 80, 200, 120)
chol = st.number_input("Colesterol (mg/dl)", 100, 600, 200)
fbs = st.radio("Azúcar en sangre en ayunas > 120 mg/dl", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

restecg = st.selectbox(
    "Resultado del electrocardiograma",
    options=[0, 1, 2],
    format_func=lambda x: [
        "0: Normal",
        "1: Anormalidad ST-T",
        "2: Hipertrofia ventricular izquierda"
    ][x]
)

thalach = st.slider("Frecuencia cardíaca máxima alcanzada", 60, 250, 150)
exang = st.radio("Angina inducida por ejercicio", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
oldpeak = st.slider("Depresión ST inducida por ejercicio", 0.0, 10.0, 1.0, 0.1)

slope = st.selectbox(
    "Pendiente del segmento ST",
    options=[0, 1, 2],
    format_func=lambda x: [
        "0: Pendiente ascendente",
        "1: Plano",
        "2: Pendiente descendente"
    ][x]
)

ca = st.selectbox("Número de vasos coloreados por fluoroscopía", [0, 1, 2, 3, 4])
thal = st.selectbox(
    "Resultado Thal",
    options=[1, 2, 3],
    format_func=lambda x: [
        "1: Normal",
        "2: Defecto fijo",
        "3: Defecto reversible"
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

modelo_seleccionado = st.selectbox(
    "Seleccione el modelo para la predicción:",
    ["Regresión Logística", "Random Forest"]
)

# -----------------------------
# Botón de predicción
# -----------------------------

if st.button("Obtener predicción"):
    if modelo_seleccionado == "Regresión Logística":
        pred = modelo_lr.predict(input_data)[0]
    else:
        pred = modelo_rf.predict(input_data)[0]

    if pred == 1:
        st.error("⚠️ El modelo predice **riesgo de enfermedad cardíaca**.")
    else:
        st.success("✅ El modelo predice **sin enfermedad cardíaca**.")

    st.info("Esta predicción es solo una estimación basada en datos clínicos.")
