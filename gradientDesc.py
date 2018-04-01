from csv import reader
import pandas as pd
import numpy

train = pd.read_csv(r"C:\Users\pranav.khorana\Documents\Internship 2017-2018\MNIST_training.csv")
test = pd.read_csv(r"C:\Users\pranav.khorana\Documents\Internship 2017-2018\MNIST_test.csv")


Xtrain = train.iloc[:, 1:]
Ytrain = train.iloc[:, 0]


# learningRate = 1e-9
learningRate = 1e-9
Ycomp = numpy.zeros((190,))
for i in range(0 , 190):
    Ycomp[i] = Ycomp[i] + 0.08


b_opt = numpy.matmul(numpy.linalg.pinv((Xtrain).transpose().dot(Xtrain)), Xtrain.transpose().dot(Ytrain))
b_est = numpy.zeros((784, ))
cost = numpy.square(Ytrain-Xtrain.dot(b_est))

print(cost)

# iter = 15

# for num in range (0,iter):
#     b_est = b_est - learningRate*(-2*Xtrain.transpose().dot(Ytrain) + 2*Xtrain.transpose().dot(Xtrain).dot(b_est))

#

# for i in range(0,190):
#     while (abs(cost[i]) > abs(Ycomp[i])):
#         b_est = b_est - learningRate*(-2*Xtrain.transpose().dot(Ytrain) + 2*Xtrain.transpose().dot(Xtrain).dot(b_est))
#         cost = numpy.square(Ytrain - Xtrain.dot(b_est))


# while (abs(cost[145]) > abs(Ycomp[145])):
#     b_est = b_est - learningRate*(-2*Xtrain.transpose().dot(Ytrain) + 2*Xtrain.transpose().dot(Xtrain).dot(b_est))
#     cost = numpy.square(Ytrain-Xtrain.dot(b_est))


#99.5 % accuracy!
while (abs(cost[147]) > abs(Ycomp[147])):
    b_est = b_est - learningRate * (-2 * Xtrain.transpose().dot(Ytrain) + 2 * Xtrain.transpose().dot(Xtrain).dot(b_est))
    cost = numpy.square(Ytrain - Xtrain.dot(b_est))



Xtest = test.iloc[:, 1:]
Ytest = test.iloc[:, 0]


y_pred = Xtest.dot(b_est)

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
            count += 1
    accuracy = (count/total) * 100
    return accuracy

print(sum(abs(b_opt-b_est)))

print(accuracyMetric(y_pred, Ytest))

