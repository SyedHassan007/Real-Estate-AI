from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
model_path = os.path.join("models", "best_sale_price_model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Real Estate AI Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_data = pd.DataFrame([{
        "Area": data["Area"],
        "Community": data["Community"],
        "Property_Type": data["Property_Type"],
        "Developer": data["Developer"],
        "Bedrooms": data["Bedrooms"],
        "Bathrooms": data["Bathrooms"],
        "Size_SqFt": data["Size_SqFt"],
        "Floor_No": data["Floor_No"],
        "Parking_Spaces": data["Parking_Spaces"],
        "Furnished": data["Furnished"],
        "Handover_Status": data["Handover_Status"],
        "Property_Age": data["Property_Age"],
        "Service_Charges": data["Service_Charges"],
        "Distance_To_Metro": data["Distance_To_Metro"],
        "Distance_To_Downtown": data["Distance_To_Downtown"],
        "Nearby_Schools": data["Nearby_Schools"],
        "Nearby_Malls": data["Nearby_Malls"],
        "Listing_Month": data["Listing_Month"],
        "Premium_Location_Flag": data["Premium_Location_Flag"],
        "Accessibility_Score": data["Accessibility_Score"],
        "Neighborhood_Convenience_Score": data["Neighborhood_Convenience_Score"]
    }])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "Predicted Price": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)