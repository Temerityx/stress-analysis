# Importing necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

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

# Creating a function for Prediction and Pie Chart using Plotly
def stress_detection(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Predict the probabilities for each class
    probabilities = model.predict_proba(input_data_reshaped)

    result = f"The person is {labels[np.argmax(probabilities)]} with probabilities:"
    for label, prob in zip(labels.values(), probabilities[0]):
        result += f"\n{label}: {prob * 100:.2f}%"

    # Pie Chart
    st.plotly_chart(plot_pie_chart(probabilities[0]))

    return result

def plot_pie_chart(probabilities):
    # Create a Plotly pie chart
    fig = px.pie(
        names=list(labels.values()),
        values=probabilities * 100,
        title='Class Percentages',
        labels={'names': 'Label', 'values': 'Percentage'}
    )

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
    if st.button('Test Result'):
        pred = stress_detection([BVP_mean, BVP_std, EDA_phasic_mean,
                                 EDA_tonic_mean, Resp_mean, TEMP_mean,
                                 BVP_peak_freq, age, height, weight])

    st.success(pred)

if __name__ == '__main__':
    main()
