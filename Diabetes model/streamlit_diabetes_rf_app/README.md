# Diabetes Prediction Streamlit App

This project deploys a Random Forest diabetes prediction model from the notebook.

## Files

- `train_model.py` trains and saves the Random Forest model.
- `compare_smote.py` compares baseline and SMOTE-enhanced model behavior with evaluation plots.
- `app.py` runs the Streamlit prediction interface.
- `requirements.txt` lists dependencies.

## VS Code setup

1. Put `diabetes_prediction_dataset.csv` in this same folder.
2. Open this folder in VS Code.
3. In the terminal, run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

For Mac/Linux activation:

```bash
source .venv/bin/activate
```

## Deploy to Streamlit Community Cloud

Upload these files to GitHub:

- `app.py`
- `train_model.py`
- `requirements.txt`
- `diabetes_rf_model.joblib`
- `model_features.joblib`

Then create a new app on Streamlit Community Cloud and select `app.py` as the main file.
