# multivariate multi-step encoder-decoder lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd

import numpy as np
np.random.seed(1234)



def split_dataset(data):
    # split into standard hours of day
    train,val,test = data[1:8713], data[8713:8737],data[8737:]
    print(train.shape)
    print(test.shape)
    print('==============================')
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 24))
    val = array(split(val, len(val) / 24))
    test = array(split(test, len(test) / 24))
    print(train)
    print(test)
    print(val)
    print('==============================')
    print(train.shape)
    print(val.shape)
    print(test.shape)
    return train, val,test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=24):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    print('data is',data)
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    print('==============================')
    print(array(X))
    print('==============================')
    print(array(y))
    print('input,Output',array(X), array(y))
    return array(X), array(y)


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    print("train x data",train_x)
    print("train y data", train_y)
    # define parameters
    verbose, epochs, batch_size = 2, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# make a forecast
def forecast(model, history, n_input,avg_power)  :
    # flatten data
    data = array(history)
    print('History data is /%n',data)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, ]
    print('input data',input_x)
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=2)
    # we only want the vector forecast
    yhat = yhat[0]
    yhat = [x*avg_power for x in yhat]
    fore1 = pd.DataFrame(yhat)
    fore1.to_csv('forecasted.csv')

    print('model prediction is',fore1)
    test_data = test[:, :, 0]
    test_data = [x*avg_power for x in test_data]
    test_data = pd.DataFrame(test_data)
    fore2= test_data.T
    print(fore2)
    fore2.to_csv('actual.csv')

    aa = [x for x in range(24)]
    fig = plt.figure()
    plt.plot(aa, fore2[:24], marker='.', label="actual")
    plt.plot(aa, fore1[:24], 'r', marker='.', label="prediction")
    plt.ylabel('Load', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.show()
    fig.savefig('prediction with (t).png')
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input,val,avg_power):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in val]
    print("History data is",history)
    # walk-forward validation over each hour
    predictions = list()
    for i in range(len(test)):
        # predict the load for 24 hour
        yhat_sequence = forecast(model, history, n_input,avg_power)
        print(yhat_sequence)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next 24 hour
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    test_data= test[:, :, 0]
    print(test_data)
    return score, scores


# load the new file
dataset = pd.read_csv('2007(1).csv',sep=',', low_memory=False,na_values=['nan','?'],
                 parse_dates={'dt' : ['Date', 'TIME']}, infer_datetime_format=True, index_col='dt')
print(dataset)
avg=dataset.mean(axis = 0, skipna = True)
print(avg)
print(avg[0])
dataset= (dataset.div(dataset.mean(axis=0), axis=1))
print(dataset)

# split into train and test
train, val, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 24
score, scores = evaluate_model(train, test, n_input,val,avg[0])

# summarize scores
summarize_scores('lstm', score, scores)
# plot scores

