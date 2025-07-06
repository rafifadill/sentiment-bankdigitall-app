import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import os
import gdown

# Fungsi untuk load model & tokenizer
@st.cache_resource
def load_model():
    model_path = "model/best_model.bin"
    os.makedirs("model", exist_ok=True)

    # Download model dari Google Drive jika belum ada
    if not os.path.exists(model_path):
        st.info("")
        gdown.download(id="10SzSH8DnYZ1Vtp_sTL5GGZMX-1iIiHEc", output=model_path, quiet=False)

    tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Mapping label
id2label = {
    0: "Positif",
    1: "Netral",
    2: "Negatif"
}

# Fungsi prediksi
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return id2label[pred], probs.squeeze().tolist()

# UI Streamlit
st.title("🇮🇩 Analisis Sentimen Ulasan Bank Digital")
st.write("Masukkan ulasan pengguna, dan model IndoBERT akan memprediksi sentimennya.")

user_input = st.text_area("✍️ Masukkan ulasan di sini", height=150)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")
    else:
        label, probas = predict_sentiment(user_input)
        st.success(f"Sentimen: **{label}**")
        st.write("📊 Probabilitas:")
        st.write(f"- Positif: {probas[0]*100:.2f}%")
        st.write(f"- Netral : {probas[1]*100:.2f}%")
        st.write(f"- Negatif: {probas[2]*100:.2f}%")
