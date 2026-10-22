import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Gym Calorie Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px;
           background: linear-gradient(120deg, #111827 0%, #243b53 55%, #1f766e 100%);
           color: white; margin-bottom: 1.2rem;}
    .hero h1 {font-size: 2.25rem; margin: 0 0 .35rem 0;}
    .hero p {color: #e5e7eb; margin: 0; font-size: 1.02rem;}
    div[data-testid="stMetric"] {background: var(--secondary-background-color);
        padding: 1rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.18);}
    .note {font-size: .88rem; color: #6b7280;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path(__file__).parent / "gym_calorie_model_fixed.pkl"

@st.cache_resource(show_spinner="Loading your trained model…")
def load_model():
    # Keep scikit-learn version aligned with the environment used to train the model.
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def model_input_columns(model):
    """Try to discover the exact input columns stored with the fitted model."""
    candidates = [
        getattr(model, "feature_names_in_", None),
        getattr(getattr(model, "named_steps", {}).get("preprocessor", None), "feature_names_in_", None),
    ]
    for names in candidates:
        if names is not None:
            return list(names)
    return None

def make_input(model, values):
    columns = model_input_columns(model)
    frame = pd.DataFrame([values])
    if columns:
        # If the model records training column names, pass them in their original order.
        for col in columns:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[columns]
    return frame

st.sidebar.title("🔥 Calorie Predictor")
page = st.sidebar.radio("Navigate", ["Predict", "How it works", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("A machine-learning demo powered by your uploaded model.")

st.markdown("""
<div class="hero">
  <h1>Gym Calorie Predictor</h1>
  <p>Estimate calories burned during a workout using your trained machine-learning model.</p>
</div>
""", unsafe_allow_html=True)

if page == "Predict":
    st.subheader("Enter workout details")
    st.write("Fill in the details below, then run the prediction.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("Gender", ["Male", "Female"], help="Choose the gender category used by your training data.")
            age = st.number_input("Age (years)", min_value=1, max_value=100, value=25, step=1)
            height = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=170.0, step=1.0)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=1.0)
        with right:
            duration = st.number_input("Workout duration (minutes)", min_value=1.0, max_value=600.0, value=30.0, step=1.0)
            heart_rate = st.number_input("Average heart rate (bpm)", min_value=30.0, max_value=240.0, value=120.0, step=1.0)
            body_temp = st.number_input("Body temperature (°C)", min_value=30.0, max_value=45.0, value=38.0, step=0.1)
        submitted = st.form_submit_button("🔥 Predict calories burned", type="primary", use_container_width=True)

    if submitted:
        values = {
            "Gender": gender,
            "Age": int(age),
            "Height": float(height),
            "Weight": float(weight),
            "Duration": float(duration),
            "Heart_Rate": float(heart_rate),
            "Body_Temp": float(body_temp),
        }
        try:
            model = load_model()
            X = make_input(model, values)
            prediction = np.asarray(model.predict(X)).reshape(-1)[0]
            if not np.isfinite(float(prediction)):
                raise ValueError("The model returned a non-finite prediction.")
            st.markdown("### Your prediction")
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated calories", f"{float(prediction):,.1f} kcal")
            c2.metric("Workout duration", f"{duration:g} min")
            c3.metric("Calories per minute", f"{float(prediction) / duration:.2f} kcal/min")
            st.progress(min(max(float(prediction) / max(float(prediction), 1.0), 0.0), 1.0))
            st.info("This is an estimate from a machine-learning model, not a medical measurement. Actual calorie burn varies by individual and workout intensity.")
        except Exception as e:
            st.error("The model could not make a prediction. Check the model file and the package versions listed in requirements.txt.")
            st.code(f"{type(e).__name__}: {e}")
            st.markdown("If this is a scikit-learn compatibility error, deploy with the training version pinned in `requirements.txt`.")

elif page == "How it works":
    st.subheader("How it works")
    st.markdown("""
    1. Enter your personal and workout measurements.
    2. The app formats them as a single-row table.
    3. Your saved model predicts the estimated calories burned.
    4. The result is displayed with a per-minute estimate.

    **Important:** the feature names and units should match the data used to train the model.
    """)
    st.warning("The model file was created with scikit-learn 1.6.1. The provided requirements file pins that version to reduce compatibility problems.")

else:
    st.subheader("About this project")
    st.write("A Streamlit dashboard for a trained gym calorie prediction model.")
    st.markdown("""
    **Built with:** Python · Streamlit · pandas · NumPy · scikit-learn

    **Model file:** `gym_calorie_model_fixed.pkl`

    This app is intended for learning and demonstration. Do not use its output as a clinical or medical result.
    """)

st.markdown("---")
st.markdown('<p class="note">Gym Calorie Predictor · Machine-learning demo</p>', unsafe_allow_html=True)
