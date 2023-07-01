import streamlit as st
import numpy as np
from PIL import Image
import pickle
import requests

def main():
    # Categorical inputs

    image = Image.open('logo.png')
    st.title("Car_Price_Predictor")

    st.sidebar.image(image, caption=f"Suri Enterprises Pvt. Ltd.", use_column_width=True)
    st.sidebar.subheader("Welcome to The Suri Enterprises Pvt. Ltd !")

    st.sidebar.subheader(
        "We rent and sales cars. This company is based on second hand or branded new cars. All the facility are available here.")
    st.sidebar.write("Ravi Ranjan ", "\n", "Chairman & CEO")
    st.sidebar.subheader('Contact Us.  \n'
                         'Email:-  ranjeet.suri241@gmail.com')

    st.sidebar.subheader("+91 75491 19745")

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

    name = st.text_input("Enter Car model Name")
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
        # st.write(feature_list)
        st.subheader("Your Input")
        user_input_data = {"Name": name, "Brand": brand, "Year of purchase": year, "Drive(KM)": driver,
                           "Owner_Type": owner_type, "Engine Type": engine_type, "Transmission Type": transmission_type,
                           "Seller Type": seller_type}
        st.write(user_input_data)
        st.subheader("Predicted Selling Price")
        input_data = np.array(feature_list).reshape(1, -1)
        model =load_model("f_model.pkl")
        prediction = model.predict(input_data)
        st.write("Predicted Selling Price :" + " " + "₹" +" " + str(np.round(prediction[0], 2)))

        st.subheader(''' Thank you for your visit !''')


if __name__ == "__main__":
    main()
