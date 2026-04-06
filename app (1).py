import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model (1) (1).h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_words(text, top_n=3):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    preds = model.predict(sequence, verbose=0)[0]
    top_indices = np.argsort(preds)[-top_n:][::-1]

    predictions = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                predictions.append((word, preds[idx]))
                break
    return predictions

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .prediction-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .confidence {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .input-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 10px;
        font-size: 1.1rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses an LSTM model to predict the next word in a sentence.")
    st.write("**Model Details:**")
    st.write("- Architecture: LSTM")
    st.write("- Training Data: Text corpus")
    st.write("- Max Sequence Length:", max_len)
    
    st.header("🔧 Settings")
    top_n = st.slider("Number of predictions", 1, 5, 3)

# Main content
st.markdown('<h1 class="main-header">🧠 Next Word Prediction</h1>', unsafe_allow_html=True)
st.markdown("### Enter a sentence and watch the AI predict the next words in real-time!")

# Input section
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Enter your text:",
            placeholder="Start typing a sentence...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        predict_button = st.button("🔮 Predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Real-time prediction
if user_input.strip():
    with st.spinner("🤖 Thinking..."):
        predictions = predict_next_words(user_input, top_n)
    
    st.markdown("### 🎯 Predictions")
    
    cols = st.columns(min(len(predictions), 3))
    for i, (word, confidence) in enumerate(predictions):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-text">{word}</div>
                <div class="confidence">Confidence: {confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & TensorFlow")
