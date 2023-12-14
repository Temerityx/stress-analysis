import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("C:/Users/Vignesh/Desktop/stress-analysis/merged.csv", index_col=0)


# Define labels
labels = {0: "Amused", 1: "Neutral", 2: "Stressed"}

# Define selected features
selected_feats = [
    "BVP_mean", "BVP_std", "EDA_phasic_mean", "EDA_phasic_min", "EDA_smna_min",
    "EDA_tonic_mean", "Resp_mean", "TEMP_mean", "TEMP_std", "TEMP_slope",
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

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy using sklearn.metrics.accuracy_score
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

#model.predict([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
###############
input_data = (0.547002262443439,	48.2830910845477,	0.165009618094931,	0.0380501748975315,	7.70619833362317,	1.99308965200501,	-0.202560799509997,	32.8636936936937,	0.0328229739342443,	-0.000941909441909418,	0.0995475113122172,		27,	175,	80)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is amused')
elif(prediction[0] == 1):
  print('The person is neutral')
else:
  print('The person is stressed')


import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (0.547002262443439,	48.2830910845477,	0.165009618094931,	0.0380501748975315,	7.70619833362317,	1.99308965200501,	-0.202560799509997,	32.8636936936937,	0.0328229739342443,	-0.000941909441909418,	0.0995475113122172,		27,	175,	80)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is amused')
elif(prediction[0] == 1):
  print('The person is neutral')
else:
  print('The person is stressed')