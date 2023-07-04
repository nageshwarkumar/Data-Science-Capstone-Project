import streamlit as st
import numpy as np
from PIL import Image
import pickle
import requests

def main():
  
    st.title("Used Car Price Predictor")




    owner1 = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3, "Fourth Owner and Above Owner": 4,
              "Test Drive Car": 5}
    seller1 = {"Individual": 1, "Dealer": 2, "Trust mark Dealer": 3}
    transmission1 = {"Manual": 1, "Automatic": 2}
    brand1 = {"Maruti": 1, "Hyundai": 2, "Mahindra": 3, "Tata": 4, "Honda": 5, "Ford": 6, "Toyota": 7, "Chevrolet": 8,
              "Renault": 9, "Volkswagen": 10,
              "Skoda": 11, "Nissan": 12, "Audi": 13, "BMW": 14, "Fiat": 15, "Datsun": 16, "Mercedes-Benz": 17,
              "Jaguar": 18, "Mitsubishi": 19, "Land": 20,
              "Volvo": 21, "Ambassador": 22, "Jeep": 23, "MG": 24, "OpelCorsa": 25, "Daewoo": 26, "Force": 27,
              "Isuzu": 28, "Kia ": 29}
    engine1 = {"Diesel": 1, "Patrol": 2, "CNG": 3, "LPG": 4,"Electric": 5}

    brand = st.selectbox("Brand", tuple(brand1.keys()))
    year = st.number_input("Year of purchase", 1900, 2023)
    driver = st.number_input("Driver(KM)")
    owner_type = st.selectbox("Owner Type", tuple(owner1.keys()))
    engine_type = st.selectbox("Engine type", tuple(engine1.keys()))
    transmission_type = st.selectbox("Transmission", tuple(transmission1.keys()))
    seller_type = st.selectbox("Seller type", tuple(seller1.keys()))

    def get_value(val, my_dict):
        for key, value in my_dict.items():
            if val == key:
                return value

    def load_model(model_file):
        model = pickle.load(open(model_file, "rb"))
        return model

    if st.button("Predict"):
        feature_list = [get_value(brand, brand1), int(year), int(driver), get_value(owner_type, owner1),
                        get_value(engine_type, engine1), get_value(transmission_type, transmission1),
                        get_value(seller_type, seller1)]
       
        st.subheader("Predicted Selling Price")
        input_data = np.array(feature_list).reshape(1, -1)
        model =load_model("final_model.pkl")
        prediction = model.predict(input_data)
        st.write("Predicted Selling Price :" + " " + "₹" +" " + str(np.round(prediction[0], 2)))

     


if __name__ == "__main__":
    main()
