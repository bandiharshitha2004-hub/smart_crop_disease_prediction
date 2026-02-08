import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq  # Updated library

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq Client
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    st.warning("⚠️ Groq API Key not found. Please check your .env file.")

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriVision AI", page_icon="🌿", layout="centered")

# Determine the absolute path to this script's directory
BASE_DIR = Path(__file__).parent

# --- CUSTOM UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .instruction-card {
        background-color: #f1f8e9;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #a5d6a7;
        color: #1b5e20;
        margin-bottom: 25px;
    }
    .instruction-card h4 { color: #2e7d32; margin-top: 0; }
    [data-testid="stImage"] img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    .suggestion-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load model using the absolute path
    model_path = BASE_DIR / "final_model.keras"
    model = tf.keras.models.load_model(str(model_path))
    
    class_indices = {"0": "Apple___Apple_scab", "1": "Apple___Black_rot", "2": "Apple___Cedar_apple_rust", "3": "Apple___healthy", "4": "Blueberry___healthy", "5": "Cherry_(including_sour)___Powdery_mildew", "6": "Cherry_(including_sour)___healthy", "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "8": "Corn_(maize)___Common_rust_", "9": "Corn_(maize)___Northern_Leaf_Blight", "10": "Corn_(maize)___healthy", "11": "Grape___Black_rot", "12": "Grape___Esca_(Black_Measles)", "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "14": "Grape___healthy", "15": "Orange___Haunglongbing_(Citrus_greening)", "16": "Peach___Bacterial_spot", "17": "Peach___healthy", "18": "Pepper,_bell___Bacterial_spot", "19": "Pepper,_bell___healthy", "20": "Potato___Early_blight", "21": "Potato___Late_blight", "22": "Potato___healthy", "23": "Raspberry___healthy", "24": "Soybean___healthy", "25": "Squash___Powdery_mildew", "26": "Strawberry___Leaf_scorch", "27": "Strawberry___healthy", "28": "Tomato___Bacterial_spot", "29": "Tomato___Early_blight", "30": "Tomato___Late_blight", "31": "Tomato___Leaf_Mold", "32": "Tomato___Septoria_leaf_spot", "33": "Tomato___Spider_mites Two-spotted_spider_mite", "34": "Tomato___Target_Spot", "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "36": "Tomato___Tomato_mosaic_virus", "37": "Tomato___healthy"}
    return model, class_indices

model, class_indices = load_assets()

# --- AI SUGGESTION LOGIC (GROQ) ---
def get_ai_suggestions(plant_name, disease_name):
    """Fetches treatment suggestions from Groq."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert agronomist. Provide professional, concise, and actionable advice."
                },
                {
                    "role": "user",
                    "content": f"The plant '{plant_name}' has been diagnosed with '{disease_name}'. Provide 3 clear treatment steps (organic or chemical) and 1 prevention tip. Keep it brief."
                }
            ],
            model="llama-3.3-70b-versatile", # Using Llama 3 for fast, high-quality responses
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error fetching suggestions: {str(e)}"

# --- SIDEBAR ---
st.sidebar.header("🌿 Supported Plants")
st.sidebar.info("Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato")

# --- MAIN UI ---
st.title("AgriVision: Plant Health AI")

st.markdown("""
<div class="instruction-card">
    <h4>📸 Photo Instructions:</h4>
    <b>1. Focus:</b> Take a clear, sharp photo of a single leaf.<br>
    <b>2. Background:</b> Use a plain, light-colored background.<br>
    <b>3. Lighting:</b> Avoid shadows; bright natural light is best.<br>
    <b>4. Distance:</b> Don't get too close; keep the whole leaf in frame.
</div>
""", unsafe_allow_html=True)

# --- SAMPLE SECTION ---
st.subheader("🧪 Quick Test with Samples")
sample_col1, sample_col2, sample_col3 = st.columns(3)

target_image = None

def get_sample(filename):
    sample_path = BASE_DIR / "samples" / filename
    if sample_path.exists():
        return Image.open(sample_path)
    else:
        st.error(f"File not found: {sample_path}")
        return None

with sample_col1:
    if st.button("Sample 1"):
        target_image = get_sample("sample1.JPG")
with sample_col2:
    if st.button("Sample 2"):
        target_image = get_sample("sample2.JPG")
with sample_col3:
    if st.button("Sample 3"):
        target_image = get_sample("sample3.JPG")

st.markdown("---")
uploaded_file = st.file_uploader("Or upload your own leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    target_image = Image.open(uploaded_file)

# --- ANALYSIS ---
if target_image:
    left_space, img_col, result_col, right_space = st.columns([0.1, 1, 1, 0.1])
    
    with img_col:
        st.write("**Leaf Image:**")
        st.image(target_image, use_container_width=True)
    
    with result_col:
        # Preprocessing
        processed_img = target_image.resize((224, 224))
        if processed_img.mode != "RGB":
            processed_img = processed_img.convert("RGB")
            
        img_array = np.array(processed_img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)
        idx = str(np.argmax(preds[0]))
        
        raw_label = class_indices[idx]
        plant = raw_label.split('___')[0].replace('_', ' ').title()
        status = raw_label.split('___')[1].replace('_', ' ').title()
        
        st.write("**Analysis Result:**")
        st.metric("Crop Type", plant)
        
        if "Healthy" in status:
            st.success(f"Status: {status} ✅")
            st.write("Your plant looks great! Keep up the good work.")
        else:
            st.error(f"Status: {status} ⚠️")
            
            # --- GROQ AI SUGGESTION BUTTON ---
            if st.button("🔍 Get Treatment Suggestions"):
                with st.spinner(f"Analyzing treatment for {status}..."):
                    advice = get_ai_suggestions(plant, status)
                    st.markdown("### 💡 Expert Advice")
                    st.info(advice)
