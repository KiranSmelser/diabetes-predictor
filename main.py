"""Predicts whether a person will have diabetes or not using KNN.
"""

import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def main():
    dataset = pd.read_csv("diabetes.csv")
    print(dataset.head())
    # Replace zeros in the dataset.
    zero_not_accepted = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]
    for column in zero_not_accepted:
        dataset[column] = dataset[column].replace(0, np.NaN)
        mean = int(dataset[column].mean(skipna=True))
        dataset[column] = dataset[column].replace(np.NaN, mean)
    # Split dataset into train and test.
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
main()