import os
import sqlite3
import random
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
DATA_PRED_DIR = os.path.join(BASE_DIR, "data", "predictions")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DB_DIR = os.path.join(BASE_DIR, "database")

for folder in [DATA_RAW_DIR, DATA_CLEANED_DIR, DATA_PRED_DIR, MODELS_DIR, OUTPUTS_DIR, DB_DIR]:
    os.makedirs(folder, exist_ok=True)

np.random.seed(42)
random.seed(42)


# ============================================================
# 2. GENERATE SYNTHETIC DUBAI REAL ESTATE DATA
# ============================================================

def generate_dubai_real_estate_data(n_rows=3000):
    areas = {
        "Dubai Marina": {"base_price_sqft": 2200, "rent_sqft": 95, "distance_downtown": 20, "distance_metro": 1.2},
        "Downtown Dubai": {"base_price_sqft": 2800, "rent_sqft": 120, "distance_downtown": 2, "distance_metro": 0.8},
        "Business Bay": {"base_price_sqft": 2100, "rent_sqft": 100, "distance_downtown": 5, "distance_metro": 1.0},
        "JVC": {"base_price_sqft": 1200, "rent_sqft": 60, "distance_downtown": 18, "distance_metro": 5.0},
        "Arjan": {"base_price_sqft": 1100, "rent_sqft": 55, "distance_downtown": 22, "distance_metro": 6.0},
        "Al Furjan": {"base_price_sqft": 1300, "rent_sqft": 62, "distance_downtown": 25, "distance_metro": 4.2},
        "Palm Jumeirah": {"base_price_sqft": 3500, "rent_sqft": 145, "distance_downtown": 18, "distance_metro": 6.5},
        "Dubai Hills": {"base_price_sqft": 1850, "rent_sqft": 82, "distance_downtown": 15, "distance_metro": 4.0},
        "International City": {"base_price_sqft": 750, "rent_sqft": 35, "distance_downtown": 28, "distance_metro": 7.0},
        "Silicon Oasis": {"base_price_sqft": 980, "rent_sqft": 48, "distance_downtown": 24, "distance_metro": 6.0},
    }

    communities = {
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

    developers = ["Emaar", "DAMAC", "Nakheel", "Azizi", "Sobha", "Ellington", "Danube", "Meraas"]
    property_types = ["Apartment", "Villa", "Townhouse", "Studio", "Penthouse"]
    furnished_options = ["Yes", "No"]
    handover_status_options = ["Ready", "Off-Plan"]
    listing_status_options = ["Active", "Sold", "Rented"]
    transaction_types = ["Sale", "Rent"]
    agents = ["Ahmed", "Sara", "Fatima", "Omar", "Ali", "Zain", "Maryam", "Hassan"]

    records = []
    start_date = datetime(2023, 1, 1)

    for i in range(1, n_rows + 1):
        area = random.choice(list(areas.keys()))
        community = random.choice(communities[area])
        property_type = random.choices(
            property_types,
            weights=[45, 15, 15, 15, 10],
            k=1
        )[0]
        developer = random.choice(developers)
        furnished = random.choice(furnished_options)
        handover_status = random.choices(handover_status_options, weights=[75, 25], k=1)[0]
        listing_status = random.choice(listing_status_options)
        transaction_type = random.choices(transaction_types, weights=[70, 30], k=1)[0]
        agent_name = random.choice(agents)

        if property_type == "Studio":
            bedrooms = 0
            bathrooms = random.choice([1, 1, 2])
            size_sqft = np.random.randint(350, 650)
        elif property_type == "Apartment":
            bedrooms = random.choice([1, 2, 3, 4])
            bathrooms = max(1, bedrooms + random.choice([0, 1]))
            size_sqft = np.random.randint(650, 2400)
        elif property_type == "Villa":
            bedrooms = random.choice([3, 4, 5, 6])
            bathrooms = bedrooms + random.choice([0, 1, 2])
            size_sqft = np.random.randint(2000, 7000)
        elif property_type == "Townhouse":
            bedrooms = random.choice([2, 3, 4, 5])
            bathrooms = bedrooms + random.choice([0, 1])
            size_sqft = np.random.randint(1600, 4000)
        else:  # Penthouse
            bedrooms = random.choice([3, 4, 5])
            bathrooms = bedrooms + random.choice([1, 2])
            size_sqft = np.random.randint(2500, 8000)

        floor_no = random.randint(1, 70) if property_type in ["Apartment", "Studio", "Penthouse"] else random.randint(0, 3)
        parking_spaces = random.choice([1, 1, 1, 2, 2, 3])

        property_age = random.randint(0, 15) if handover_status == "Ready" else 0
        nearby_schools = random.randint(1, 8)
        nearby_malls = random.randint(1, 5)
        service_charges = round(np.random.uniform(8, 35) * size_sqft, 2)

        base_price_sqft = areas[area]["base_price_sqft"]
        rent_sqft = areas[area]["rent_sqft"]
        distance_to_downtown = max(1, areas[area]["distance_downtown"] + np.random.normal(0, 2))
        distance_to_metro = max(0.2, areas[area]["distance_metro"] + np.random.normal(0, 0.8))

        type_multiplier = {
            "Studio": 0.90,
            "Apartment": 1.00,
            "Townhouse": 1.10,
            "Villa": 1.18,
            "Penthouse": 1.45
        }[property_type]

        furnished_multiplier = 1.06 if furnished == "Yes" else 1.00
        ready_multiplier = 1.08 if handover_status == "Ready" else 0.97
        age_penalty = max(0.82, 1 - (property_age * 0.008))
        premium_adjustment = np.random.normal(1.0, 0.08)

        price_per_sqft = base_price_sqft * type_multiplier * furnished_multiplier * ready_multiplier * age_penalty * premium_adjustment
        sale_price = size_sqft * price_per_sqft

        # Add influence from bedrooms, schools, malls, metro, downtown
        sale_price += bedrooms * 25000
        sale_price += nearby_schools * 8000
        sale_price += nearby_malls * 12000
        sale_price -= distance_to_metro * 12000
        sale_price -= distance_to_downtown * 5000
        sale_price = max(180000, round(sale_price, 2))

        annual_rent = size_sqft * rent_sqft * type_multiplier * furnished_multiplier
        annual_rent += bedrooms * 2500
        annual_rent -= distance_to_metro * 1500
        annual_rent = max(18000, round(annual_rent, 2))

        listing_date = start_date + timedelta(days=random.randint(0, 800))
        completion_status = "Completed" if handover_status == "Ready" else "Upcoming"

        # Approximate lat/lon
        base_lat_lon = {
            "Dubai Marina": (25.0800, 55.1400),
            "Downtown Dubai": (25.1972, 55.2744),
            "Business Bay": (25.1860, 55.2708),
            "JVC": (25.0560, 55.2110),
            "Arjan": (25.0600, 55.2450),
            "Al Furjan": (25.0280, 55.1450),
            "Palm Jumeirah": (25.1124, 55.1390),
            "Dubai Hills": (25.1020, 55.2400),
            "International City": (25.1640, 55.4080),
            "Silicon Oasis": (25.1220, 55.3790),
        }
        base_lat, base_lon = base_lat_lon[area]
        latitude = round(base_lat + np.random.normal(0, 0.01), 6)
        longitude = round(base_lon + np.random.normal(0, 0.01), 6)

        record = {
            "Property_ID": f"PROP_{i:05d}",
            "Listing_Date": listing_date.date(),
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
            "Sale_Price": sale_price,
            "Annual_Rent": annual_rent,
            "Service_Charges": service_charges,
            "Distance_To_Metro": round(distance_to_metro, 2),
            "Distance_To_Downtown": round(distance_to_downtown, 2),
            "Nearby_Schools": nearby_schools,
            "Nearby_Malls": nearby_malls,
            "Latitude": latitude,
            "Longitude": longitude,
            "Agent_Name": agent_name,
            "Listing_Status": listing_status,
            "Transaction_Type": transaction_type,
            "Completion_Status": completion_status
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Intentionally inject some missing values for realism
    for col in ["Bathrooms", "Floor_No", "Parking_Spaces", "Service_Charges"]:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df



# ============================================================
# 3. CLEAN + FEATURE ENGINEERING
# ============================================================

def clean_and_engineer_data(df):
    df = df.copy()

    df["Listing_Date"] = pd.to_datetime(df["Listing_Date"], errors="coerce")

    # Standardize text columns
    text_cols = ["Area", "Community", "Property_Type", "Developer", "Furnished", "Handover_Status", "Agent_Name"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # Fill numeric nulls with median
    numeric_cols_to_fill = ["Bathrooms", "Floor_No", "Parking_Spaces", "Service_Charges"]
    for col in numeric_cols_to_fill:
        df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    df["Price_Per_SqFt"] = df["Sale_Price"] / df["Size_SqFt"]
    df["Rent_Per_SqFt"] = df["Annual_Rent"] / df["Size_SqFt"]
    df["Rental_Yield"] = (df["Annual_Rent"] / df["Sale_Price"]) * 100
    df["Listing_Month"] = df["Listing_Date"].dt.month
    df["Listing_Year"] = df["Listing_Date"].dt.year
    df["Luxury_Flag"] = np.where(df["Sale_Price"] >= 2500000, 1, 0)
    df["Premium_Location_Flag"] = np.where(
        df["Area"].isin(["Downtown Dubai", "Palm Jumeirah", "Dubai Marina"]), 1, 0
    )
    df["Accessibility_Score"] = (
        (10 - np.clip(df["Distance_To_Metro"], 0, 10)) * 0.5 +
        (10 - np.clip(df["Distance_To_Downtown"] / 3, 0, 10)) * 0.3 +
        np.clip(df["Nearby_Malls"], 0, 5) * 0.2
    )
    df["Neighborhood_Convenience_Score"] = (
        df["Nearby_Schools"] * 0.4 +
        df["Nearby_Malls"] * 0.6
    )

    # Investment score
    def investment_grade(row):
        score = 0
        if row["Rental_Yield"] >= 6:
            score += 2
        elif row["Rental_Yield"] >= 4:
            score += 1

        if row["Accessibility_Score"] >= 6:
            score += 1

        if row["Premium_Location_Flag"] == 1:
            score += 1

        if row["Service_Charges"] < 50000:
            score += 1

        if score >= 4:
            return "High"
        elif score >= 2:
            return "Medium"
        else:
            return "Low"

    df["Investment_Grade"] = df.apply(investment_grade, axis=1)

    return df



# ============================================================
# 4. SAVE TO EXCEL + CSV
# ============================================================

def save_excel_and_csv(raw_df, cleaned_df):
    raw_csv_path = os.path.join(DATA_RAW_DIR, "dubai_real_estate_raw.csv")
    cleaned_csv_path = os.path.join(DATA_CLEANED_DIR, "dubai_real_estate_cleaned.csv")
    excel_path = os.path.join(DATA_CLEANED_DIR, "dubai_real_estate_master.xlsx")

    raw_df.to_csv(raw_csv_path, index=False)
    cleaned_df.to_csv(cleaned_csv_path, index=False)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        raw_df.to_excel(writer, sheet_name="properties_raw", index=False)
        cleaned_df.to_excel(writer, sheet_name="properties_cleaned", index=False)

        summary = cleaned_df.groupby(["Area", "Property_Type"]).agg(
            Total_Listings=("Property_ID", "count"),
            Avg_Sale_Price=("Sale_Price", "mean"),
            Avg_Annual_Rent=("Annual_Rent", "mean"),
            Avg_Rental_Yield=("Rental_Yield", "mean")
        ).reset_index()
        summary.to_excel(writer, sheet_name="summary_kpis", index=False)

    print(f"Saved raw CSV: {raw_csv_path}")
    print(f"Saved cleaned CSV: {cleaned_csv_path}")
    print(f"Saved Excel workbook: {excel_path}")


# ============================================================
# 5. LOAD INTO SQLITE DATABASE
# ============================================================

def load_into_sqlite(cleaned_df):
    db_path = os.path.join(DB_DIR, "real_estate.db")
    conn = sqlite3.connect(db_path)

    cleaned_df.to_sql("fact_property_listing", conn, if_exists="replace", index=False)

    area_summary_query = """
    SELECT
        Area,
        Property_Type,
        COUNT(*) AS total_listings,
        ROUND(AVG(Sale_Price), 2) AS avg_sale_price,
        ROUND(AVG(Annual_Rent), 2) AS avg_annual_rent,
        ROUND(AVG(Rental_Yield), 2) AS avg_rental_yield
    FROM fact_property_listing
    GROUP BY Area, Property_Type
    ORDER BY avg_sale_price DESC
    """

    monthly_trend_query = """
    SELECT
        Listing_Year,
        Listing_Month,
        COUNT(*) AS total_listings,
        ROUND(AVG(Sale_Price), 2) AS avg_sale_price
    FROM fact_property_listing
    GROUP BY Listing_Year, Listing_Month
    ORDER BY Listing_Year, Listing_Month
    """

    roi_query = """
    SELECT
        Area,
        ROUND(AVG(Annual_Rent / Sale_Price * 100), 2) AS avg_rental_yield
    FROM fact_property_listing
    GROUP BY Area
    ORDER BY avg_rental_yield DESC
    """

    area_summary_df = pd.read_sql_query(area_summary_query, conn)
    monthly_trend_df = pd.read_sql_query(monthly_trend_query, conn)
    roi_df = pd.read_sql_query(roi_query, conn)

    area_summary_df.to_csv(os.path.join(DATA_CLEANED_DIR, "area_summary.csv"), index=False)
    monthly_trend_df.to_csv(os.path.join(DATA_CLEANED_DIR, "monthly_market_trend.csv"), index=False)
    roi_df.to_csv(os.path.join(DATA_CLEANED_DIR, "roi_by_area.csv"), index=False)

    conn.close()

    print(f"Saved SQLite DB: {db_path}")
    print("Saved SQL output CSV files for Power BI.")


# ============================================================
# 6. TRAIN ML MODEL
# ============================================================

def train_model(cleaned_df):
    df = cleaned_df.copy()

    feature_cols = [
        "Area",
        "Community",
        "Property_Type",
        "Developer",
        "Bedrooms",
        "Bathrooms",
        "Size_SqFt",
        "Floor_No",
        "Parking_Spaces",
        "Furnished",
        "Handover_Status",
        "Property_Age",
        "Service_Charges",
        "Distance_To_Metro",
        "Distance_To_Downtown",
        "Nearby_Schools",
        "Nearby_Malls",
        "Listing_Month",
        "Premium_Location_Flag",
        "Accessibility_Score",
        "Neighborhood_Convenience_Score",
    ]

    target_col = "Sale_Price"

    X = df[feature_cols]
    y = df[target_col]

    categorical_features = ["Area", "Community", "Property_Type", "Developer", "Furnished", "Handover_Status"]
    numeric_features = [col for col in feature_cols if col not in categorical_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model 1
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)

    # Model 2
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)

    def evaluate_model(name, actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = mean_squared_error(actual, predicted) ** 0.5
        r2 = r2_score(actual, predicted)
        return {
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 4)
        }

    results = [
        evaluate_model("Linear Regression", y_test, lr_preds),
        evaluate_model("Random Forest", y_test, rf_preds)
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)

    # Use best model
    best_model_name = results_df.sort_values(by="R2_Score", ascending=False).iloc[0]["Model"]
    best_pipeline = rf_pipeline if best_model_name == "Random Forest" else lr_pipeline
    best_preds = rf_preds if best_model_name == "Random Forest" else lr_preds

    # Save model
    model_path = os.path.join(MODELS_DIR, "best_sale_price_model.pkl")
    joblib.dump(best_pipeline, model_path)

    # Export predictions
    predictions_df = X_test.copy()
    predictions_df["Actual_Sale_Price"] = y_test.values
    predictions_df["Predicted_Sale_Price"] = best_preds
    predictions_df["Prediction_Error"] = predictions_df["Actual_Sale_Price"] - predictions_df["Predicted_Sale_Price"]
    predictions_df.to_csv(os.path.join(DATA_PRED_DIR, "sale_price_predictions.csv"), index=False)

    # Feature importance for RF only
    if best_model_name == "Random Forest":
        preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
        model_fitted = best_pipeline.named_steps["model"]

        feature_names_num = numeric_features
        feature_names_cat = preprocessor_fitted.named_transformers_["cat"] \
            .named_steps["onehot"].get_feature_names_out(categorical_features)

        all_feature_names = list(feature_names_num) + list(feature_names_cat)
        importances = model_fitted.feature_importances_

        feature_importance_df = pd.DataFrame({
            "Feature": all_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        feature_importance_df.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance.csv"), index=False)

    print("Model training completed.")
    print(results_df)
    print(f"Best model saved at: {model_path}")

    return results_df, predictions_df


# ============================================================
# 7. CREATE CHARTS
# ============================================================

def create_charts(cleaned_df, predictions_df):
    # Chart 1: Average sale price by area
    avg_price_by_area = cleaned_df.groupby("Area")["Sale_Price"].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    avg_price_by_area.plot(kind="bar")
    plt.title("Average Sale Price by Area")
    plt.xlabel("Area")
    plt.ylabel("Average Sale Price")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "avg_sale_price_by_area.png"))
    plt.close()

    # Chart 2: Rental yield by area
    avg_yield_by_area = cleaned_df.groupby("Area")["Rental_Yield"].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    avg_yield_by_area.plot(kind="bar")
    plt.title("Average Rental Yield by Area")
    plt.xlabel("Area")
    plt.ylabel("Rental Yield (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "rental_yield_by_area.png"))
    plt.close()

    # Chart 3: Actual vs predicted
    sample_df = predictions_df.head(150).copy()

    plt.figure(figsize=(10, 6))
    plt.plot(sample_df["Actual_Sale_Price"].values, label="Actual")
    plt.plot(sample_df["Predicted_Sale_Price"].values, label="Predicted")
    plt.title("Actual vs Predicted Sale Price")
    plt.xlabel("Sample Index")
    plt.ylabel("Sale Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png"))
    plt.close()

    print("Charts created successfully.")



# ============================================================
# 8. MAIN RUNNER
# ============================================================

def main():
    print("Step 1: Generating synthetic Dubai real estate data...")
    raw_df = generate_dubai_real_estate_data(n_rows=3000)

    print("Step 2: Cleaning and engineering features...")
    cleaned_df = clean_and_engineer_data(raw_df)

    print("Step 3: Saving raw and cleaned files...")
    save_excel_and_csv(raw_df, cleaned_df)

    print("Step 4: Loading data into SQLite and exporting SQL summaries...")
    load_into_sqlite(cleaned_df)

    print("Step 5: Training ML model...")
    results_df, predictions_df = train_model(cleaned_df)

    print("Step 6: Creating charts...")
    create_charts(cleaned_df, predictions_df)

    print("\nProject execution completed successfully.")
    print(f"\nCheck these folders:\n- data/\n- database/\n- models/\n- outputs/")


if __name__ == "__main__":
    main()