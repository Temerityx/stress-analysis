# combined_app.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("merged.csv", index_col=0)

# Define labels
labels = {0: "Amused", 1: "Neutral", 2: "Stressed"}

# Define selected features
selected_feats = [
    "BVP_mean", "BVP_std", "EDA_phasic_mean",
    "EDA_tonic_mean", "Resp_mean", "TEMP_mean",
    "BVP_peak_freq", "age", "height", "weight",
]

# Prepare features and target variable
X = df[selected_feats]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.text(f"Model Accuracy: {accuracy * 100:.2f}%")

# creating a function for Prediction and Pie Chart
def stress_detection(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    result = f"The person is {labels[prediction[0]]}"

    # Plot Pie Chart
    fig = plot_pie_chart(y_test)
    st.pyplot(fig)

    return result

def plot_pie_chart(y_test):
    labels_count = y_test.value_counts().sort_index()

    # Define labels
    labels = ['Amused', 'Neutral', 'Stressed']

    # Create a list of counts corresponding to the order of labels
    values = [labels_count.get(label, 0) for label in labels]

    # Handle NaN values by replacing them with 0
    values = [0 if pd.isna(value) else value for value in values]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig


def main():
    # giving a title
    st.title('Stress Detection Web App')

    # getting the input data from the user
    BVP_mean = st.text_input('BVP_mean')
    BVP_std = st.text_input('BVP_std')
    EDA_phasic_mean = st.text_input('EDA_phasic_mean')
    EDA_tonic_mean = st.text_input('EDA_tonic_mean')
    Resp_mean = st.text_input('Resp_mean')
    TEMP_mean = st.text_input('TEMP_mean')
    BVP_peak_freq = st.text_input('BVP_peak_freq')
    age = st.number_input("Age")
    height = st.number_input("Height")
    weight = st.number_input("Weight")

    # code for Prediction
    pred = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        pred = stress_detection([BVP_mean, BVP_std, EDA_phasic_mean,
                                 EDA_tonic_mean, Resp_mean, TEMP_mean,
                                 BVP_peak_freq, age, height, weight])

    st.success(pred)

if __name__ == '__main__':
    main()
