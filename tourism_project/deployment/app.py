import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load the trained pipeline from the Hugging Face Model Hub
MODEL_REPO = "treddy333/tourism-wellness-model"
MODEL_FILE = "best_tourism_prod_taken_model_v1.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(model_path)

st.title("Wellness Tourism Package — Purchase Prediction")
st.write(
    "Enter customer attributes below. Inputs are assembled into a **pandas DataFrame** "
    "and passed to the model (same column names as training)."
)

typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"],
)
gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
product_pitched = st.selectbox(
    "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)
marital = st.selectbox(
    "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
)
designation = st.selectbox(
    "Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

age = st.number_input("Age", 18, 100, 35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_pitch = st.number_input("Duration of Pitch (minutes)", 0.0, 120.0, 15.0)
n_person = st.number_input("Number of Persons Visiting", 1, 10, 2)
n_follow = st.number_input("Number of Follow-ups", 0.0, 10.0, 3.0)
pref_star = st.number_input("Preferred Property Star", 1.0, 7.0, 3.0)
n_trips = st.number_input("Number of Trips", 0.0, 30.0, 2.0)
passport = st.selectbox("Passport", [0, 1])
pitch_sat = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Own Car", [0, 1])
n_children = st.number_input("Number of Children Visiting", 0.0, 10.0, 0.0)
monthly_income = st.number_input("Monthly Income", 0.0, 100000.0, 20000.0)

input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": typeof_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": n_person,
            "NumberOfFollowups": n_follow,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": pref_star,
            "MaritalStatus": marital,
            "NumberOfTrips": n_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_sat,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": n_children,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }
    ]
)

if st.button("Predict"):
    prediction = int(model.predict(input_data)[0])
    label = "Likely to purchase the package" if prediction == 1 else "Unlikely to purchase"
    st.subheader("Prediction")
    st.success(label)
