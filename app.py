import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import pickle

# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

# App title
st.title('Iris Species Prediction')

# User input
sepal_length = st.number_input('Sepal length (cm)', min_value=0.0, max_value=10.0, value=0.0)
sepal_width = st.number_input('Sepal width (cm)', min_value=0.0, max_value=10.0, value=0.0)
petal_length = st.number_input('Petal length (cm)', min_value=0.0, max_value=10.0, value=0.0)
petal_width = st.number_input('Petal width (cm)', min_value=0.0, max_value=10.0, value=0.0)

# Predict button
if st.button('Predict'):
    # Create new data
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predictions
    prediction = model.predict(X_new)
    
    # Display prediction
    st.write(f"Prediction of Species: {prediction[0]}")
