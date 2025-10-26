
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import io
import os
import pickle

# try to import reportlab for PDF generation; we'll fallback if not available
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

DATA_PATH = "diet_recommendation_sample (1).csv"

@st.cache_resource
def load_and_train(path=DATA_PATH):
    df = pd.read_csv(path)
    # derive BMI for training records (useful for nearest-match later)
    df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)
    # encode categorical columns used for model
    encoders = {}
    for col in ['Gender', 'Fitness_Goal', 'Diet_Preference', 'Diet_Plan']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    # features and target
    X = df[['Age','Gender','Height_cm','Weight_kg','BMI','Fitness_Goal','Diet_Preference']]
    y = df['Diet_Plan']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return df, model, encoders

def predict_plan(model, encoders, age, gender, height_cm, weight_kg, fitness_goal, diet_pref):
    bmi = weight_kg / ((height_cm / 100) ** 2)
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': encoders['Gender'].transform([gender])[0],
        'Height_cm': height_cm,
        'Weight_kg': weight_kg,
        'BMI': bmi,
        'Fitness_Goal': encoders['Fitness_Goal'].transform([fitness_goal])[0],
        'Diet_Preference': encoders['Diet_Preference'].transform([diet_pref])[0]
    }])
    pred_encoded = model.predict(input_df)[0]
    pred_label = encoders['Diet_Plan'].inverse_transform([pred_encoded])[0]
    return pred_label, round(bmi,2), pred_encoded

def pick_representative_row(df_original, predicted_plan_label, gender, fitness_goal, diet_pref, bmi):
    # df_original still has original string labels; ensure we use originals
    # find rows that match predicted plan and diet preference and fitness goal if possible
    candidates = df_original[ (df_original['Diet_Plan'] == predicted_plan_label) &
                              (df_original['Diet_Preference'] == diet_pref) ]
    # if none, relax by plan only
    if candidates.empty:
        candidates = df_original[df_original['Diet_Plan'] == predicted_plan_label]
    # compute BMI diff and pick nearest BMI
    candidates['BMI_diff'] = (candidates['BMI'] - bmi).abs()
    best = candidates.sort_values('BMI_diff').iloc[0]
    return best

def create_pdf_report(result_dict, filename):
    if not REPORTLAB_AVAILABLE:
        return None
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    x_margin = 40
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, "Personalized Diet Report")
    y -= 30
    c.setFont("Helvetica", 11)
    for k,v in result_dict.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        c.drawString(x_margin, y, f"{k}: {v}")
        y -= 18
        if y < 80:
            c.showPage()
            y = height - 50
    c.save()
    return filename

st.set_page_config(page_title="AI Diet Planner", layout="centered")

st.title("🥗 AI Diet Planner (CSV-trained)")
st.write("This app uses a static CSV (`diet_recommendation_sample (1).csv`) bundled with the app to train a model at startup. Enter your details below and press Generate.")

# load dataset and model
with st.spinner("Loading training data and training model..."):
    df_encoded, model, encoders = load_and_train(DATA_PATH)

# Also load the string-labelled dataframe for display/nearest-match (we saved original strings earlier)
df_display = pd.read_csv(DATA_PATH)
df_display['BMI'] = df_display['Weight_kg'] / ((df_display['Height_cm']/100)**2)

st.sidebar.header("Your Details")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=28, step=1)
gender = st.sidebar.selectbox("Gender", options=list(encoders['Gender'].classes_))
height_cm = st.sidebar.number_input("Height (cm)", min_value=100, max_value=230, value=170)
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
fitness_goal = st.sidebar.selectbox("Fitness Goal", options=list(encoders['Fitness_Goal'].classes_))
diet_pref = st.sidebar.selectbox("Diet Preference", options=list(encoders['Diet_Preference'].classes_))

generate = st.sidebar.button("Generate Diet Plan")

if generate:
    pred_label, bmi_value, pred_encoded = predict_plan(model, encoders, age, gender, height_cm, weight_kg, fitness_goal, diet_pref)
    st.metric("Your BMI", bmi_value)
    st.subheader("Recommended Diet Plan")
    st.success(pred_label)
    # find representative example row to show meal plan & macros
    rep = pick_representative_row(df_display, pred_label, gender, fitness_goal, diet_pref, bmi_value)
    # prepare result dict
    result = {
        "Age": age,
        "Gender": gender,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "BMI": bmi_value,
        "Fitness_Goal": fitness_goal,
        "Diet_Preference": diet_pref,
        "Suggested_Diet_Plan": pred_label,
        "Breakfast": rep['Breakfast'],
        "Lunch": rep['Lunch'],
        "Dinner": rep['Dinner'],
        "Snacks": rep['Snacks'],
        "Calories": rep['Calories'],
        "Protein_g": rep['Protein_g'],
        "Carbs_g": rep['Carbs_g'],
        "Fats_g": rep['Fats_g']
    }
    # show table
    st.subheader("Personalized Diet Chart (example)")
    df_out = pd.DataFrame([result])
    st.dataframe(df_out.T, height=320)
    # download CSV
    csv_buf = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Plan as CSV", data=csv_buf, file_name="my_diet_plan.csv", mime="text/csv")
    # Create PDF if possible
    if REPORTLAB_AVAILABLE:
        pdf_fname = f"my_diet_plan_{age}_{int(weight_kg)}.pdf"
        create_pdf_report(result, pdf_fname)
        with open(pdf_fname, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("⬇️ Download Plan as PDF", data=pdf_bytes, file_name=pdf_fname, mime="application/pdf")
        # remove the temporary file to keep workspace tidy
        try:
            os.remove(pdf_fname)
        except Exception:
            pass
    else:
        # fallback: offer TXT download
        txt = "\n".join([f"{k}: {v}" for k, v in result.items()])
        st.download_button("⬇️ Download Plan as TXT (PDF not available)", data=txt.encode('utf-8'), file_name="my_diet_plan.txt", mime="text/plain")

st.markdown("---")
st.write("### Training dataset preview (first 10 rows)")
st.dataframe(df_display.head(10))

st.write("If you want me to package this app (`app.py`) and the CSV into a zip, tell me and I'll prepare the files for download.")
