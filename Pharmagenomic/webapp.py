import streamlit as st
import torch
from encoding import encode_sample, idx2drug
from model import PharmaModel

import json
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix

@st.cache_resource(show_spinner=False)
def load_model():
    model = PharmaModel()
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def predict(sample):
    x_encoded, _ = encode_sample(sample)
    x_tensor = torch.tensor([x_encoded], dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return idx2drug[pred_idx]

st.title("💊 Predizione Farmacogenomica")

st.markdown("""
Inserisci i dati del paziente per ottenere il farmaco consigliato basato su diagnosi e profilo genetico.
""")

diagnosis = st.selectbox("Diagnosi", ["Depressione", "Ipertensione", "Asma"])
cyp2d6 = st.selectbox("CYP2D6", ["PM", "IM", "EM", "UM"])
cyp2c19 = st.selectbox("CYP2C19", ["PM", "IM", "EM", "UM"])

if st.button("Calcola farmaco"):
    sample = {
        "diagnosis": diagnosis,
        "CYP2D6": cyp2d6,
        "CYP2C19": cyp2c19,
    }
    try:
        pred = predict(sample)
        st.success(f"💊 Farmaco consigliato: **{pred}**")
    except Exception as e:
        st.error(f"Errore nella predizione: {e}")

st.subheader("📉 Andamento della Loss durante l'allenamento")

try:
    with open("loss_log.json", "r") as f:
        losses = json.load(f)

    plt.style.use("dark_background")

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#0e1117')  
    ax.set_facecolor('#0e1117')

    ax.plot(range(1, len(losses)+1), losses, label="Loss", color="#00ff7f", linewidth=2)

    ax.set_xlabel("Epoca", fontsize=12, color='white')
    ax.set_ylabel("Loss", fontsize=12, color='white')
    ax.set_title("Loss durante l'allenamento", fontsize=14, color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')

    st.pyplot(fig)

except FileNotFoundError:
    st.warning("⚠️ Il file della loss non è stato trovato. Esegui `run_train.py` per generarlo.")

st.subheader("📊 Confronto tra farmaci reali e predetti")

try:
    with open("prediction_vs_actual.json", "r") as f:
        data = json.load(f)
        true = data["true"]
        pred = data["pred"]

    labels = list(idx2drug.values())
    cm = confusion_matrix(true, pred)

    plt.style.use("dark_background")

    fig_cm, ax_cm = plt.subplots()
    fig_cm.patch.set_facecolor('#0e1117')
    ax_cm.set_facecolor('#0e1117')

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",           
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor='#0e1117',
        annot_kws={"color": "black", "fontsize": 10}
    )

    ax_cm.set_xlabel("Predetto", fontsize=12, color='white')
    ax_cm.set_ylabel("Reale", fontsize=12, color='white')
    ax_cm.set_title("Matrice di Confusione", fontsize=14, color='white')
    ax_cm.tick_params(colors='white')

    st.pyplot(fig_cm)

except FileNotFoundError:
    st.warning("⚠️ Nessun file di predizione trovato. Esegui `run_train.py` per generarlo.")
