from csv import reader
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from math import exp

dataset = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
print(dataset.shape)




data = dataset.values
# y = data.iloc[0:, 0]
#
#
# for index in range(0, y.size):
#     if (y[index] == 8):
#         y[index] = 1
#     else:
#         y[index] = 0



Xtrain = data[0:700, 0:8]
Ytrain = data[0:700, 8]

Xtest = data[700:, 0:8]
Ytest = data[700:, 8]

learningRate = 1e-6
# 7
print(Xtrain.shape)
print(Ytrain)
X0 = np.ones((700 , 1))
X1 = np.ones((67 , 1))


Xtrain = np.hstack((X0, Xtrain))
Xtest = np.hstack((X1, Xtest))
print(Xtrain)
print(Xtrain.shape)
b = np.zeros((9,))
# b = np.zeros((784,))


# Find the min and max values for each column


def likelihood(b):
    first = 0
    for i in range(0, 700):
        first -= np.log10(1 + np.exp(np.matmul(Xtrain[i, :], b)))
    second = 0
    for i in range(0, 700):
        second += Ytrain[i] * (np.matmul(Xtrain[i, :], b))
    likelihood = first + second
    return likelihood


def derivative(j):
    first = 0
    for i in range(0, 700):
        first -= (1 / (1 + (np.exp(np.matmul(Xtrain[i, :], b))))) * (np.exp(np.matmul(Xtrain[i, :], b))) * Xtrain[i, j]
    second = 0
    for i in range(0, 700):
        second += Ytrain[i]*Xtrain[i, j]
    derivative = first + second
    return derivative


def predict(b):
    for x in range(0, 67):
        y[x] += (np.exp(np.matmul(Xtest[x, :], b)))/(1 + np.exp(np.matmul(Xtest[x, :], b)))
    print(y)
    for x in range(0, 67):
        if y[x] >= 0.4:
            y[x] = 1
        else:
            y[x] = 0
    print(y)
    return y


def accuracyMetric(predict, actual):
    total = len(actual)
    count = 0
    ones = 0
    # for i in range(0, 67):
    #     if actual[i] == 1:
    #         ones += 1
    for i in range(0, 67):
        if predict[i] == actual[i]:
            count += 1
    accuracy = (count/total)*100
    # for i in range(0, 67):
    #     if predict[i] == actual[i] == 1:
    #         count += 1
    # accuracy = (count/ones)*100
    return accuracy



print(likelihood(b))
list = []
list.append(likelihood(b))
print("$")
for num in range(0, 300):
    for j in range(0, 9):
        b[j] += learningRate*derivative(j)
        print(b[j])
    list.append(likelihood(b))
    print("$")
    print(likelihood(b))
    print("$")
plt.plot(list)
plt.show()
print(b)

y = np.zeros((67, 1))
y = predict(b)

print(accuracyMetric(y, Ytest))



