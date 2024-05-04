import streamlit as st
import pickle
import numpy as np

# Load the regression model
try:
    with open('regmodel.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("The regression model file ('regmodel.pkl') was not found.")

# Load the scaler
try:
    with open('scaling.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("The scaler file ('scaling.pkl') was not found.")

# Function to preprocess input data and make prediction
def predict_price(data):
    # Preprocess input data using the scaler
    input_data = np.array([[data['MedInc'], data['HouseAge'], data['AveRooms'], data['AveBedrms'],
                            data['Population'], data['AveOccup'], data['Latitude'], data['Longitude']]])
    scaled_input_data = scaler.transform(input_data)

    # Make prediction using the regression model
    prediction = model.predict(scaled_input_data)[0]
    return prediction*100000 

# Streamlit UI
def main():
    st.title('California Housing Price Prediction')

    # Input form
    st.header('Input Data')
    med_inc = st.slider('MedInc', min_value=1, max_value=15, step=1)
    house_age = st.slider('HouseAge', min_value=1, max_value=50, step=1)
    ave_rooms = st.slider('AveRooms', min_value=1, max_value=10, step=1)
    ave_bedrms = st.slider('AveBedrms', min_value=1, max_value=5, step=1)
    population = st.slider('Population', min_value=1, max_value=5000, step=10)
    ave_occup = st.slider('AveOccup', min_value=1, max_value=10, step=1)
    latitude = st.slider('Latitude', min_value=32, max_value=42, step=1)
    longitude = st.slider('Longitude', min_value=-125, max_value=-114, step=1)

    input_data = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }

    # Prediction
    if st.button('Predict'):
        prediction = predict_price(input_data)
        st.success(f'Predicted Price: ${prediction:.2f}')

    st.markdown('Creator:-')
    st.markdown('[Aaryan Palve](https://aaryanpalve5.github.io/PortfolioASP/index.html)')

if __name__ == '__main__':
    main()
