from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Load the dataset

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Load the dataset
    data = pd.read_csv('/Users/animesh/Desktop/python/HousePricePrediction/USA_Housing.csv')
    # Preprocess the data
    data = data.drop(['Address'], axis=1)
    # Split the data into features and target variable
    X = data.drop(['Price'], axis=1)
    Y = data['Price']       
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Create a linear regression model
    model = LinearRegression()
    # Train the model
    model.fit(X_train, Y_train)

    var1 = float(request.GET.get('n1', 0))
    var2 = float(request.GET.get('n2', 0))
    var3 = float(request.GET.get('n3', 0))
    var4 = float(request.GET.get('n4', 0))
    var5 = float(request.GET.get('n5', 0))

    pred = model.predict(np.array([[var1, var2, var3, var4, var5]]))
    pred = round(pred[0], 2)
    # Display the predicted price
    

    price = f"Predicted House Price: ${pred:,.2f}"
    try:
        val1 = float(request.GET.get('n1', 0))
        val2 = float(request.GET.get('n2', 0))
        val3 = float(request.GET.get('n3', 0))
        val4 = float(request.GET.get('n4', 0))
        val5 = float(request.GET.get('n5', 0))

        # Dummy logic for prediction - replace this with your model prediction
        predicted_price = val1 + val2 + val3 + val4 + val5

        result_text = f"Predicted House Price: ${predicted_price:,.2f}"

    except Exception as e:
        result_text = f"Error in input: {e}"

    return render(request, 'predict.html', {'result2': price})
