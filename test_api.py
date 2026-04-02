import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Area": "Dubai Marina",
    "Community": "Marina Gate",
    "Property_Type": "Apartment",
    "Developer": "Emaar",
    "Bedrooms": 2,
    "Bathrooms": 2,
    "Size_SqFt": 1200,
    "Floor_No": 15,
    "Parking_Spaces": 1,
    "Furnished": "Yes",
    "Handover_Status": "Ready",
    "Property_Age": 3,
    "Service_Charges": 20000,
    "Distance_To_Metro": 1.2,
    "Distance_To_Downtown": 20,
    "Nearby_Schools": 4,
    "Nearby_Malls": 3,
    "Listing_Month": 5,
    "Premium_Location_Flag": 1,
    "Accessibility_Score": 7,
    "Neighborhood_Convenience_Score": 5
}

response = requests.post(url, json=data)

print(response.json())