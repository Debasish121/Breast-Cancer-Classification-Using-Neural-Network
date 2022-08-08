# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 23:59:25 2022

@author: debav
"""

import tensorflow
import numpy as np
import pickle
from tensorflow import keras
from keras.models import load_model
import streamlit as st


# Loading the Saved Model
loaded_model = load_model('D:/Debasish/ML/Projects/Breast Cancer Classification/trained_model.h5')


# Loading saved StandardScaler model
loaded_scaler = pickle.load(open('D:/Debasish/ML/Projects/Breast Cancer Classification/scaler.pkl', 'rb'))



# Creating a function for Prediction
def breastCancer_classification(input_data):

    # change the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    
    # reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    
    # standardizing the input data
    input_data_std = loaded_scaler.transform(input_data_reshaped)
    
    
    prediction = loaded_model.predict(input_data_std)
    print(prediction)
    
    
    prediction_label = [np.argmax(prediction)]
    print(prediction_label)


    if(prediction_label[0] == 0):
      return 'The Tumor is Malignant'
    
    else:
      return 'The Tumor is Benign'
 
    
 
# Main Function   
  
def main():
    
    # Giving a Title
    st.title('Breast Cancer Classification Web App')
    
    # Getting the input data from the user
    
    meanRadius = st.text_input("Mean Radius")
    meanPerimeter = st.text_input("Mean Perimeter")
    meanArea = st.text_input("Mean Area")
    meanCompactness = st.text_input("Mean Compactness")
    meanConcavity = st.text_input("Mean Concavity")
    meanConcavePoints = st.text_input("Mean Concave Points")
    worstRadius = st.text_input("Worst Radius")
    worstPerimeter = st.text_input("Worst Perimeter")
    worstArea = st.text_input("Worst Area")
    worstConcavity = st.text_input("Worst Concavity")
    worstConcavePoints = st.text_input("Worst Cancave Points")
    
    
    # Code for Prediction
    diagnosis = ''
    
    # Creating a button for Prediction
    
    if st.button('Diabetes test Result'):
        diagnosis = breastCancer_classification([meanRadius, meanPerimeter, meanArea, meanCompactness, meanConcavity, meanConcavePoints, worstRadius, worstPerimeter, worstArea, worstConcavity, worstConcavePoints])
        
    
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
 
    
 