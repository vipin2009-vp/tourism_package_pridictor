import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="Vipin0287/tourism_package_model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
""")

# User input for Wellness Tourism Package Prediction

age = st.number_input("Age of Customer", min_value=18, max_value=100, value=30, step=1)

typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "FreeLancer"])
gender = st.selectbox("Gender", ["Male", "Female"])

number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1, step=1)
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=1, step=1)

passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0, step=1)

designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=20000.0, step=100.0)

pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=20, value=2, step=1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("Customer is likely to purchase the Wellness Tourism Package ✅")
    else:
        st.warning("Customer is unlikely to purchase the Wellness Tourism Package ❌")
