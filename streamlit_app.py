import os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

np.random.bit_generator = np.random._bit_generator
# -------------------------------
# 1. Cargar artefactos
# -------------------------------
ART_DIR = os.path.join("artefactos", "v1")

PIPE = joblib.load(os.path.join(ART_DIR, "pipeline_NNM.joblib"))
INPUT_SCHEMA = json.load(open(os.path.join(ART_DIR, "input_schema.json"), "r", encoding="utf-8"))
LABEL_MAP = json.load(open(os.path.join(ART_DIR, "label_map.json"), "r", encoding="utf-8"))
POLICY = json.load(open(os.path.join(ART_DIR, "decision_policy.json"), "r", encoding="utf-8"))

REV_LABEL = {v: k for k, v in LABEL_MAP.items()}
THRESHOLD = POLICY.get("threshold", 0.5)

FEATURES = list(INPUT_SCHEMA.keys())


# ---------------------------------------
# 2. Función para preparar datos de entrada
# ---------------------------------------
def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    for c, t in INPUT_SCHEMA.items():
        if c not in df.columns:
            df[c] = np.nan
        if str(t).startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif str(t).lower() in ("bool", "boolean"):
            df[c] = df[c].astype("bool")
        else:
            df[c] = df[c].astype("string")
    return df[FEATURES]


# ---------------------------------------
# 3. Función de predicción
# ---------------------------------------
def predict_one(record: dict):
    df = _coerce_and_align(pd.DataFrame([record]))
    proba = PIPE.predict_proba(df)[:, 1][0]
    pred = int(proba >= THRESHOLD)
    
    return {
        "proba": float(proba),
        "pred_int": pred,
        "pred_label": REV_LABEL[pred]
    }


# ---------------------------------------
# 4. Interfaz en Streamlit
# ---------------------------------------
st.set_page_config(page_title="Predicción de Preeclampsia", layout="centered")
st.title("🔮 Predicción de Riesgo de Preeclampsia")
st.write("Modelo clínico basado en Machine Learning para estimar la probabilidad de preeclampsia.")

# Formulario
st.header("📋 Ingrese los datos de la paciente")

record = {}

record["edad"] = st.number_input("Edad", min_value=10, max_value=60, step=1)
record["imc"] = st.number_input("IMC", min_value=10.0, max_value=60.0, step=0.1)
record["p_a_sistolica"] = st.number_input("Presión Arterial Sistólica (mmHg)", min_value=70, max_value=200)
record["p_a_diastolica"] = st.number_input("Presión Arterial Diastólica (mmHg)", min_value=40, max_value=150)

record["hipertension"] = st.selectbox("Antecedente de hipertensión", [0, 1])
record["diabetes"] = st.selectbox("Antecedente de diabetes", [0, 1])

record["creatinina"] = st.number_input("Creatinina (mg/dL)", min_value=0.1, max_value=3.0, step=0.1)
record["ant_fam_hiper"] = st.selectbox("Antecedentes familiares de hipertensión", [0, 1])
record["tec_repro_asistida"] = st.selectbox("Técnica de reproducción asistida", [0, 1])

# Botón de predicción
if st.button("🔍 Predecir riesgo"):
    res = predict_one(record)
    
    st.subheader("🧾 Resultado")
    st.write(f"**Probabilidad estimada de riesgo:** {res['proba']*100:.2f}%")
    st.write(f"**Clasificación:** {res['pred_label']}")
    
    if res["pred_label"] == "RIESGO":
        st.error("⚠ La paciente presenta riesgo elevado de preeclampsia.")
    else:

        st.success("✔ La paciente no presenta riesgo significativo según el modelo.")
