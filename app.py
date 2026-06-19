import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# CONFIG


st.set_page_config(
    page_title="TaHoma Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)



st.markdown("""
<style>
    /* Fond de l'application clair et lumineux */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Force la police et la couleur du texte global en sombre pour le contraste */
    html, body, [data-testid="stWidgetLabel"] p, .stMarkdown {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        color: #1E293B !important;
    }

    /* Titre Principal (TRÈS GRAND) */
    .main-title {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        margin-bottom: 8px !important;
        line-height: 1.2 !important;
    }

    /* Sous-titre (Grand et Pro) */
    .sub-title {
        font-size: 1.4rem !important;
        color: #475569 !important;
        margin-bottom: 40px !important;
        font-weight: 400 !important;
    }

    /* Label de la zone de texte aggrandi */
    .input-label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        margin-bottom: 10px !important;
    }

    /* Cartes de Résultat Modernes et Lumineuses (Style SaaS) */
    .result-box {
        padding: 35px;
        border-radius: 16px;
        text-align: center;
        margin-top: 10px;
        background-color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }

    /* Titre du sentiment (TRÈS GRAND) */
    .result-box h2 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 15px !important;
    }

    /* Score de confiance */
    .result-box h3 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        color: #475569 !important;
    }

    /* Variations de bordures pour le thème clair */
    .positive {
        border-top: 8px solid #22C55E;
        background: linear-gradient(to bottom, #F0FDF4, #FFFFFF);
    }
    .positive h2 { color: #166534 !important; }

    .negative {
        border-top: 8px solid #EF4444;
        background: linear-gradient(to bottom, #FEF2F2, #FFFFFF);
    }
    .negative h2 { color: #991B1B !important; }

    .neutral {
        border-top: 8px solid #EAB308;
        background: linear-gradient(to bottom, #FEFCE8, #FFFFFF);
    }
    .neutral h2 { color: #854D0E !important; }

    /* Grand titre pour la partie probabilités */
    .section-prob-title {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-bottom: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_all():
    model = load_model("bilstm_sentiment_model.h5")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_all()

# =====================================================
# AUTOMATIC MAX LENGTH
# =====================================================
MAX_LEN = model.input_shape[1]


# DEBUG INFO
with st.sidebar:
    st.markdown("<h2 style='font-size: 1.5rem; font-weight:700;'>⚙️ Paramètres du modèle</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.write(f"**Input Shape :** {model.input_shape}")
    st.write(f"**Output Shape :** {model.output_shape}")
    st.write(f"**MAX_LEN :** {MAX_LEN}")


# HEADER 

st.markdown('<p class="main-title">📱 Analyse des sentiments — TaHoma Somfy</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analyse avancée de la satisfaction client basée sur un réseau de neurones BiLSTM.</p>', unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================
st.markdown('<p class="input-label">Saisissez le commentaire  :</p>', unsafe_allow_html=True)
user_text = st.text_area(
    label="Texte invisible pour accessibilité", 
    label_visibility="collapsed",
    height=160,
    
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔍 Lancer l'analyse automatique", use_container_width=True):
    if len(user_text.strip()) == 0:
        st.warning("⚠️ Veuillez saisir un texte avant de lancer l'analyse.")
    else:
        try:
            # Tokenisation & Padding
            seq = tokenizer.texts_to_sequences([user_text])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

            # Prediction
            pred = model.predict(padded, verbose=0)
            probs = pred[0]
            class_id = np.argmax(probs)
            
            sentiment = label_encoder.inverse_transform([class_id])[0]
            confidence = float(np.max(probs)) * 100

            st.markdown("---")
            st.markdown("<h3 style='font-size: 1.8rem; font-weight: 800; margin-bottom: 25px;'> Résultats de l'évaluation</h3>", unsafe_allow_html=True)
            
            # Layout Pro en 2 colonnes bien espacées
            col1, col2 = st.columns([1.1, 1], gap="large")

            with col1:
                # Affichage des cartes claires haut de gamme
                if sentiment.lower() == "positive":
                    st.markdown(f"""
                        <div class="result-box positive">
                            <h2>😊 Positif</h2>
                            <h3>Fiabilité du verdict : {confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                elif sentiment.lower() == "negative":
                    st.markdown(f"""
                        <div class="result-box negative">
                            <h2>😡 Négatif</h2>
                            <h3>Fiabilité du verdict : {confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-box neutral">
                            <h2>😐 Neutre</h2>
                            <h3>Fiabilité du verdict : {confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown('<p class="section-prob-title">Distribution des probabilités :</p>', unsafe_allow_html=True)
                for i, classe in enumerate(label_encoder.classes_):
                    # Affichage grand format du texte des probabilités
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; font-size:1.1rem; font-weight:600; margin-bottom:6px; color:#334155;'>"
                        f"<span>{classe.capitalize()}</span>"
                        f"<span>{probs[i]*100:.2f}%</span>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                    st.progress(float(probs[i]))
                    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Une erreur technique est survenue : {str(e)}")
