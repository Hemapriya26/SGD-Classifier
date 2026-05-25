# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries: scikit-learn modules for dataset loading, preprocessing, classification, train-test split, and evaluation. 

2.Load the Iris dataset and separate input features X and target labels Y. 

3.Normalize the feature values using StandardScaler. 

4.Split the dataset into training and testing sets. 

5.Create and train the SGD Classifier model using the training data. 

6.Predict the test data results, calculate accuracy, and display the classification report.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Hemapriya P
RegisterNumber:  212225040126

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Species:", iris.target_names[prediction][0])

plt.scatter(X[:,0], X[:,1], c=y)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification")

plt.show()
```

## Output:

<img width="1205" height="721" alt="image" src="https://github.com/user-attachments/assets/73063889-24ab-4dca-a5fd-1729ba1b281d" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
