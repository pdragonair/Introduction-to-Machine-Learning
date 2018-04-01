from csv import reader
import pandas as pd
import numpy

train = pd.read_csv(r"C:\Users\pranav.khorana\Documents\Internship 2017-2018\MNIST_training.csv")
test = pd.read_csv(r"C:\Users\pranav.khorana\Documents\Internship 2017-2018\MNIST_test.csv")

Xtrain = train.iloc[:, 1:]
Ytrain = train.iloc[:, 0]



b_opt = numpy.matmul(numpy.linalg.pinv((Xtrain).transpose().dot(Xtrain)), Xtrain.transpose().dot(Ytrain))
#(X'X)^-1 * X'y

Xtest = test.iloc[:, 1:]
Ytest = test.iloc[:, 0]


y_pred = Xtest.dot(b_opt)
for i in range(0, y_pred.size):
    if (y_pred[i] > 0.5):
        y_pred[i] = 1
    else:
        y_pred[i] = 0

def accuracyMetric(predict, actual):
    total = len(actual)
    count = 0
    for i in range(0, predict.size-1):
        if (predict[i] == actual[i]):
            count+=1
    accuracy = (count/total)*100
    return accuracy



print(accuracyMetric(y_pred, Ytest))

print(b_opt.shape)