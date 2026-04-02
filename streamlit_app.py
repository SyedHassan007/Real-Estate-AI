import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Dubai Real Estate Price Predictor",
    page_icon="🏙️",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
MODEL_PATH = os.path.join("models", "best_sale_price_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# MASTER DATA
# --------------------------------------------------
AREA_COMMUNITIES = {
    "Dubai Marina": ["Marina Gate", "Princess Tower", "Cayan Tower"],
    "Downtown Dubai": ["Burj Views", "Opera District", "The Residences"],
    "Business Bay": ["Executive Towers", "DAMAC Towers", "Bay Square"],
    "JVC": ["District 10", "District 12", "District 15"],
    "Arjan": ["Lincoln Park", "Vincitore", "Miraclz"],
    "Al Furjan": ["Masakin", "Azizi Tulip", "Murooj"],
    "Palm Jumeirah": ["Shoreline", "Golden Mile", "Marina Residences"],
    "Dubai Hills": ["Park Heights", "Sidra", "Mulberry"],
    "International City": ["China Cluster", "England Cluster", "France Cluster"],
    "Silicon Oasis": ["Silicon Gates", "Palace Towers", "Axis Residences"],
}

DEVELOPERS = ["Emaar", "DAMAC", "Nakheel", "Azizi", "Sobha", "Ellington", "Danube", "Meraas"]
PROPERTY_TYPES = ["Apartment", "Villa", "Townhouse", "Studio", "Penthouse"]
FURNISHED_OPTIONS = ["Yes", "No"]
HANDOVER_OPTIONS = ["Ready", "Off-Plan"]

PREMIUM_AREAS = {"Downtown Dubai", "Palm Jumeirah", "Dubai Marina"}

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def calculate_accessibility_score(distance_to_metro, distance_to_downtown, nearby_malls):
    return (
        (10 - min(max(distance_to_metro, 0), 10)) * 0.5
        + (10 - min(max(distance_to_downtown / 3, 0), 10)) * 0.3
        + min(max(nearby_malls, 0), 5) * 0.2
    )

def calculate_neighborhood_convenience_score(nearby_schools, nearby_malls):
    return nearby_schools * 0.4 + nearby_malls * 0.6

def format_aed(value):
    return f"AED {value:,.0f}"

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🏙️ Dubai Real Estate Price Predictor")
st.markdown("Enter property details below to predict the estimated sale price.")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("About")
st.sidebar.write(
    """
    This app uses a trained machine learning model to estimate Dubai property sale prices
    based on location, size, property type, accessibility, and related features.
    """
)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.selectbox("Area", list(AREA_COMMUNITIES.keys()))
        community = st.selectbox("Community", AREA_COMMUNITIES[area])
        property_type = st.selectbox("Property Type", PROPERTY_TYPES)
        developer = st.selectbox("Developer", DEVELOPERS)
        furnished = st.selectbox("Furnished", FURNISHED_OPTIONS)
        handover_status = st.selectbox("Handover Status", HANDOVER_OPTIONS)

    with col2:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        size_sqft = st.number_input("Size (SqFt)", min_value=300, max_value=15000, value=1200, step=50)
        floor_no = st.number_input("Floor No", min_value=0, max_value=100, value=10, step=1)
        parking_spaces = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1, step=1)
        property_age = st.number_input("Property Age", min_value=0, max_value=50, value=3, step=1)

    with col3:
        service_charges = st.number_input("Service Charges", min_value=0.0, max_value=500000.0, value=20000.0, step=1000.0)
        distance_to_metro = st.number_input("Distance to Metro (km)", min_value=0.0, max_value=20.0, value=1.2, step=0.1)
        distance_to_downtown = st.number_input("Distance to Downtown (km)", min_value=0.0, max_value=50.0, value=20.0, step=0.5)
        nearby_schools = st.number_input("Nearby Schools", min_value=0, max_value=20, value=4, step=1)
        nearby_malls = st.number_input("Nearby Malls", min_value=0, max_value=10, value=3, step=1)
        listing_month = st.slider("Listing Month", min_value=1, max_value=12, value=5)

    submitted = st.form_submit_button("Predict Price")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submitted:
    premium_location_flag = 1 if area in PREMIUM_AREAS else 0
    accessibility_score = calculate_accessibility_score(
        distance_to_metro, distance_to_downtown, nearby_malls
    )
    neighborhood_convenience_score = calculate_neighborhood_convenience_score(
        nearby_schools, nearby_malls
    )

    input_df = pd.DataFrame([{
        "Area": area,
        "Community": community,
        "Property_Type": property_type,
        "Developer": developer,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Size_SqFt": size_sqft,
        "Floor_No": floor_no,
        "Parking_Spaces": parking_spaces,
        "Furnished": furnished,
        "Handover_Status": handover_status,
        "Property_Age": property_age,
        "Service_Charges": service_charges,
        "Distance_To_Metro": distance_to_metro,
        "Distance_To_Downtown": distance_to_downtown,
        "Nearby_Schools": nearby_schools,
        "Nearby_Malls": nearby_malls,
        "Listing_Month": listing_month,
        "Premium_Location_Flag": premium_location_flag,
        "Accessibility_Score": accessibility_score,
        "Neighborhood_Convenience_Score": neighborhood_convenience_score
    }])

    prediction = model.predict(input_df)[0]
    price_per_sqft = prediction / size_sqft if size_sqft else 0

    st.success("Prediction completed successfully.")

    out1, out2, out3 = st.columns(3)
    out1.metric("Predicted Sale Price", format_aed(prediction))
    out2.metric("Estimated Price / SqFt", f"AED {price_per_sqft:,.0f}")
    out3.metric("Premium Location", "Yes" if premium_location_flag == 1 else "No")

    st.subheader("Input Summary")
    st.dataframe(input_df, use_container_width=True)

    st.subheader("Scoring Insights")
    score_df = pd.DataFrame({
        "Metric": [
            "Accessibility Score",
            "Neighborhood Convenience Score"
        ],
        "Value": [
            round(accessibility_score, 2),
            round(neighborhood_convenience_score, 2)
        ]
    })
    st.dataframe(score_df, use_container_width=True)

    st.info(
        "This estimate is model-based and should be treated as an analytical prediction, "
        "not an official market valuation."
    )