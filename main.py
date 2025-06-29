import streamlit as st
import joblib
import string

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(
    page_title="📧 Email SpamGuard AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("spam_classifier_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

model, vectorizer = load_model()

# --- Text Preprocessing ---
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stTextArea textarea {
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-size: 16px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 7px 14px rgba(0, 0, 0, 0.15) !important;
    }
    
    .result-box {
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .spam-result {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    
    .ham-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .title-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    .footer {
        text-align: center;
        padding: 10px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        margin-top: 20px;
    }
    
    .header-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- App Interface ---
st.markdown("""
<div style='text-align: center;'>
    <div class='header-icon'>📧</div>
    <h1 class='title-text'>Email Spam Detection System</h1>
    <h3 style='color: #555; font-weight: 400;'>Smart Detection for Suspicious Emails</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;'>
    <p style='font-size: 16px; color: #555;'>Enter any email content below to check if it's spam or legitimate.</p>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    label="",
    placeholder="Paste your email content here...",
    height=200,
    key="message_input"
)

predict_btn = st.button("🔍 Analyze Email")

if predict_btn:
    if not user_input.strip():
        st.warning("Please enter an email to analyze")
    elif model is None or vectorizer is None:
        st.error("Model not loaded properly - please check your model files")
    else:
        with st.spinner("Analyzing email content..."):
            cleaned = preprocess(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)
            
            if prediction[0] == 1:
                st.markdown("""
                <div class='result-box spam-result'>
                    <h2><span style='font-size: 1.5em;'>❌</span> Spam Detected!</h2>
                    <p>This email appears to be suspicious.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-box ham-result'>
                    <h2><span style='font-size: 1.5em;'>✅</span> Safe Email</h2>
                    <p>This email appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    <p>Made with ❤️ by <strong>Abdul Razzaq</strong> </p>
</div>
""", unsafe_allow_html=True)