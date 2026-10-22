# Gym Calorie Predictor — Streamlit

A Streamlit dashboard that loads `gym_calorie_model_fixed.pkl` and predicts estimated calories burned during a workout.

## Run locally

Use Python 3.11 or 3.12 for the smoothest compatibility with the pinned scikit-learn version.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Deploy on GitHub + Streamlit Community Cloud

1. Create a new **public GitHub repository** (for example `gym-calorie-predictor`).
2. Upload `app.py`, `requirements.txt`, `README.md`, and `gym_calorie_model_fixed.pkl` to the repository root.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository, branch, and `app.py`.
5. Deploy.

Keep the repository public if using the usual free Community Cloud workflow. Do not upload private or sensitive data.

## Model compatibility note

The uploaded pickle appears to have been saved with scikit-learn 1.6.1. This project pins that version. Pickle files can execute code when loaded, so deploy only a model file you trust. If the model expects different input column names or units, update the `values` dictionary in `app.py` to match the model's training features.
