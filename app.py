import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
car_data = pd.read_csv('CAR DETAILS.csv')

# Select relevant features and target variable
X = car_data[['Brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
y = car_data['selling_price']

# Perform label encoding for categorical variables
label_encoder = LabelEncoder()
X['Brand'] = label_encoder.fit_transform(X['Brand'])
X['fuel'] = label_encoder.fit_transform(X['fuel'])
X['seller_type'] = label_encoder.fit_transform(X['seller_type'])
X['transmission'] = label_encoder.fit_transform(X['transmission'])
X['owner'] = label_encoder.fit_transform(X['owner'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest regression model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Define the Streamlit app
def main():
    st.title('Car Price Prediction')
    st.markdown('Enter the details of the car to get the predicted price.')

    # Collect input from the user
    brand = st.selectbox('Brand', car_data['Brand'].unique())
    year = st.number_input('Year')
    km_driven = st.number_input('Kilometers Driven')
    fuel = st.selectbox('Fuel', car_data['fuel'].unique())
    seller_type = st.selectbox('Seller Type', car_data['seller_type'].unique())
    transmission = st.selectbox('Transmission', car_data['transmission'].unique())
    owner = st.selectbox('Owner', car_data['owner'].unique())

    # Encode the input
    brand_encoded = label_encoder.transform([brand])
    fuel_encoded = label_encoder.transform([fuel])
    seller_type_encoded = label_encoder.transform([seller_type])
    transmission_encoded = label_encoder.transform([transmission])
    owner_encoded = label_encoder.transform([owner])

    # Make prediction on the input
    new_car = [[brand_encoded[0], year, km_driven, fuel_encoded[0], seller_type_encoded[0], transmission_encoded[0], owner_encoded[0]]]
    predicted_price = model.predict(new_car)

    # Display the predicted price
    st.subheader('Predicted Price')
    st.write(predicted_price)

if __name__ == '__main__':
    main()
