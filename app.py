import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for Prediction

def stress_detection(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is amused'
    elif(prediction[0] == 1):
      return 'The person is neutral'
    else:
      return 'The person is stressed'
  
    
  
def main():
    
    
    # giving a title
    st.title('Stress Detection Web App')
    
    
    # getting the input data from the user

    
    BVP_mean = st.text_input('BVP_mean')
    BVP_std = st.text_input('BVP_std')
    EDA_phasic_mean = st.text_input('EDA_phasic_mean')
    EDA_phasic_min = st.text_input('EDA_phasic_min')
    EDA_smna_min = st.text_input('EDA_smna_min')
    EDA_tonic_mean = st.text_input('EDA_tonic_mean')
    Resp_mean = st.text_input('Resp_mean')
    TEMP_mean = st.text_input('TEMP_mean')
    TEMP_std = st.text_input('TEMP_std')
    TEMP_slope = st.text_input('TEMP_slope')
    BVP_peak_freq = st.text_input('BVP_peak_freq')
    age = st.text_input('Age')
    height = st.text_input('Height')
    weight= st.text_input('Weight')
    
    
    # code for Prediction
    pred = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        pred = stress_detection([BVP_mean, BVP_std, EDA_phasic_mean, EDA_phasic_min, EDA_smna_min, EDA_tonic_mean, Resp_mean, TEMP_mean, TEMP_std, TEMP_slope, BVP_peak_freq, age, height, weight])
        
        
    st.success(pred)
    
    
    
    
    
if __name__ == '__main__':
    main()