import streamlit as st
import pandas as pd
import joblib

# ------------------- Load label encoders for UI options -------------------
le_fuel = joblib.load("label_encode_fuel.pkl")
le_owner = joblib.load("label_encode_owner.pkl")
le_seller = joblib.load("label_encode_seller_type.pkl")
le_trans = joblib.load("label_encode_transmission.pkl")

# ------------------- Load trained model pipeline -------------------
model = joblib.load("best_model_pipeline.pkl")

# ------------------- Set Page Configuration -------------------
st.set_page_config(page_title="AutoValuator AI", page_icon="🚘", layout="wide", initial_sidebar_state="collapsed")

# ------------------- Custom CSS & Styling -------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main background gradient: deep luxurious dark mode */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Containers */
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="column"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }

    /* Input Field Styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: rgba(0, 0, 0, 0.2) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Labels */
    .stNumberInput label p, .stSelectbox label p {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
    }

    /* Primary Gradient Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #0ea5e9 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-size: 1.25rem;
        font-weight: 600;
        border: none;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3b82f6 0%, #0ea5e9 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.5);
        color: white !important;
    }
    .stButton>button:active {
        transform: translateY(1px) scale(0.98);
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.3);
    }

    /* Prediction Card */
    .prediction-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        text-align: center;
        margin-top: 1.5rem;
        border: 1px solid rgba(56, 189, 248, 0.3);
        animation: fadeIn 0.8s ease-out forwards;
        position: relative;
        overflow: hidden;
    }
    .prediction-card::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0) 100%);
        transform: skewX(-25deg);
        animation: shine 4s infinite 1s;
    }
    .prediction-title {
        color: #94a3b8;
        font-size: 1.3rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .prediction-value {
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0px 4px 20px rgba(56, 189, 248, 0.2);
    }
    .prediction-subtitle {
        color: #cbd5e1;
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.8;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes shine {
        0% { left: -100%; }
        20% { left: 200%; }
        100% { left: 200%; }
    }

    /* Titles */
    h1 {
        font-size: 4rem !important;
        font-weight: 800 !important;
        text-align: center;
        background: linear-gradient(to right, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.2rem;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 3.5rem;
        opacity: 0.9;
    }
    
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------- Application Header -------------------
st.markdown('<h1>🚘 AutoValuator AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your vehicle specifications to get a highly accurate, AI-powered market appraisal.</p>', unsafe_allow_html=True)

# ------------------- Input Sections -------------------
# Use custom ratio for layout: wider center columns
col1, space, col2 = st.columns([1, 0.1, 1])

with col1:
    st.markdown("<h3 style='color: #f1f5f9; font-weight: 600; text-align: center; margin-bottom: 1.5rem; font-size: 1.6rem;'>⚙️ Performance Specs</h3>", unsafe_allow_html=True)
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2026, value=2015, help="What year was the car manufactured?")
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000, help="Total distance the car has travelled")
    mileage = st.number_input("Mileage (kmpl)", min_value=1.0, value=20.0, step=0.5, help="Fuel efficiency in km per liter")
    engine = st.number_input("Engine Capacity (CC)", min_value=500.0, value=1197.0, step=100.0, help="Engine displacement in CC")
    max_power = st.number_input("Max Power (bhp)", min_value=10.0, value=82.0, step=5.0)

with col2:
    st.markdown("<h3 style='color: #f1f5f9; font-weight: 600; text-align: center; margin-bottom: 1.5rem; font-size: 1.6rem;'>📝 General Details</h3>", unsafe_allow_html=True)
    seats = st.number_input("Number of Seats", min_value=1.0, max_value=10.0, value=5.0)
    fuel = st.selectbox("Fuel Type", le_fuel.classes_)
    owner = st.selectbox("Owner Type", le_owner.classes_)
    seller = st.selectbox("Seller Type", le_seller.classes_)
    transmission = st.selectbox("Transmission", le_trans.classes_)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------- Calculate Button & Output -------------------
btn_col1, btn_col2, btn_col3 = st.columns([1, 1.2, 1])

with btn_col2:
    calculate_pressed = st.button("🔮 Calculate Estimated Value")

pred_placeholder = st.empty()

if calculate_pressed:
    with st.spinner("🧠 AI is analyzing market trends..."):
        try:
            # ------------------- Prepare input for the model -------------------
            input_df = pd.DataFrame([[year, km_driven, mileage, engine, max_power, seats,
                                      fuel, owner, seller, transmission]],
                                    columns=['year','km_driven','mileage','engine','max_power','seats',
                                             'fuel','owner','seller_type','transmission'])

            # ------------------- Make prediction -------------------
            predicted_price = model.predict(input_df)[0]

            # ------------------- Show robust result -------------------
            st.balloons()
            pred_placeholder.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-title">Estimated Market Value</div>
                    <div class="prediction-value">₹ {predicted_price:,.0f}</div>
                    <div class="prediction-subtitle">Based on historical data and real-time deep learning analysis</div>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# Optional Info panel
st.markdown("<br><br><br>", unsafe_allow_html=True)
with st.expander("ℹ️ About the Machine Learning Model"):
    st.markdown("""
    <div style='color: #cbd5e1; font-weight: 400; line-height: 1.7; font-size: 1.05rem; padding: 1rem;'>
    This <b>Random Forest Regression</b> model was trained on an extensive dataset of thousands of used car listings.<br><br>
    It learns complex patterns from critical features—such as <b>fuel type, engine power, and mileage</b>—to approximate a highly accurate <b>fair market value</b>. By exploring non-linear patterns in depreciation and vehicle specifications, it provides you with an unbiased evaluation of what a vehicle is worth in the current market.
    </div>
    """, unsafe_allow_html=True)