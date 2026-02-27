import streamlit as st
import pandas as pd
import joblib

# ------------------- Load label encoders for UI options -------------------
le_fuel = joblib.load("label_encode_fuel.pkl")         # LabelEncoder for Fuel Type
le_owner = joblib.load("label_encode_owner.pkl")       # LabelEncoder for Owner
le_seller = joblib.load("label_encode_seller_type.pkl")# LabelEncoder for Seller Type
le_trans = joblib.load("label_encode_transmission.pkl")# LabelEncoder for Transmission

# ------------------- Load trained model pipeline -------------------
model = joblib.load("best_model_pipeline.pkl")         # Trained regression pipeline

# ------------------- Set Page Configuration -------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# ------------------- Custom CSS & Styling -------------------
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 2rem;
        border-top: 6px solid #2a5298;
    }
    .prediction-title {
        color: #4a5568;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-value {
        color: #1a202c;
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    h1 {
        color: #1e3c72;
        font-weight: 800;
        text-align: center;
        padding-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Streamlit UI -------------------
st.markdown('<h1>Used Car Price Predictor 🚗</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter the specifications of a used car below to get an intelligent AI-powered appraisal.</p>', unsafe_allow_html=True)

# Organize inputs into two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Primary Specifications")
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2026, value=2015, help="What year was the car manufactured?")
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000, help="Total distance the car has travelled")
    mileage = st.number_input("Mileage (kmpl)", min_value=1.0, value=20.0, step=0.5, help="Fuel efficiency in km per liter")
    engine = st.number_input("Engine Capacity (CC)", min_value=500.0, value=1197.0, step=100.0, help="Engine displacement in CC")
    max_power = st.number_input("Max Power (bhp)", min_value=10.0, value=82.0, step=5.0)

with col2:
    st.markdown("### 🏷️ General Details")
    seats = st.number_input("Number of Seats", min_value=1.0, max_value=10.0, value=5.0)
    fuel = st.selectbox("Fuel Type", le_fuel.classes_)
    owner = st.selectbox("Owner Type", le_owner.classes_)
    seller = st.selectbox("Seller Type", le_seller.classes_)
    transmission = st.selectbox("Transmission", le_trans.classes_)

st.markdown("---")

# Empty container to hold prediction output so it looks nice
pred_placeholder = st.empty()

# Prediction button
if st.button("Calculate Estimated Value"):
    with st.spinner("Analyzing market data..."):
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
                    <div class="prediction-value">₹ {predicted_price:,.2f}</div>
                    <div style="color: #718096; font-size: 0.9rem;">Based on current historical data trends</div>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# Optional Info panel
with st.expander("ℹ️ About the Model"):
    st.write("""
    This Random Forest Regression model was trained on thousands of used car listings to help approximate fair market values based on critical features such as fuel type, engine capacity, mileage, and ownership history.
    """)