import os
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime as dt
from collections import UserDict

from TimeSeriesTensor import TimeSeriesTensor

# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# http://scikit-learn.org/stable/install.html
from sklearn.preprocessing import MinMaxScaler

# this does not work. skip it at first
# https://blog.csdn.net/liangzuojiayi/article/details/78183783
# https://ipython.org/ipython-doc/3/interactive/tutorial.html
# https://docs.microsoft.com/en-us/visualstudio/python/interactive-repl-ipython
# need ipython to support ?
# skip it at first
# % %matplotlib inline

pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)



# load_data.py
def load_data():

    # read GEFCom2014 load data

    data_dir = 'data/'

    energy = pd.read_csv(os.path.join(data_dir, 'energy.csv'), parse_dates=['timestamp'])

    # Reindex the dataframe such that the dataframe has a record for every time point
    # between the minimum and maximum timestamp in the time series. This helps to 
    # identify missing time periods in the data (there are none in this dataset).

    energy.index = energy['timestamp']
    energy = energy.reindex(pd.date_range(min(energy['timestamp']),
                                          max(energy['timestamp']),
                                          freq='H'))
    energy = energy.drop('timestamp', axis=1)

    return energy

# mape.py
def mape(predictions, actuals):
    return ((predictions - actuals).abs() / actuals).mean()

# create_evalatuion_df.py
def create_evaluation_df(predictions, test_inputs, H, scaler):
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, H+1)])
    eval_df['timestamp'] = test_inputs.dataframe.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target']).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df



if __name__ == "__main__":

    # step 1:
    # download data folders
    # set extract_data.py as start file and generate  energy.csv


    # step2: use energy.csv to extra data
    energy = load_data()
    print (energy.head())

    valid_start_dt = '2014-09-01 00:00:00'
    test_start_dt = '2014-11-01 00:00:00'

    T = 6
    HORIZON = 3


    # Create training set containing only the model features
    train = energy.copy()[energy.index < valid_start_dt][['load']]


    # Scale data to be in range (0, 1). This transformation should be calibrated on the training set only.
    # normalization  only applies to training data. This is to prevent information from the validation or test sets leaking into the training data. 
    y_scaler = MinMaxScaler()
    y_scaler.fit(train[['load']])

    X_scaler = MinMaxScaler()
    train[['load']] = X_scaler.fit_transform(train)


    # using time seriese tensor class to parse data
    # Shift the values of the time series to create a Pandas dataframe containing all the data for a single training example
    # Discard any samples with missing values
    # Transform this Pandas dataframe into a numpy array of shape (samples, time steps, features) for input into Keras

    tensor_structure = {'encoder_input':(range(-T+1, 1), ['load']), 'decoder_input':(range(0, HORIZON), ['load'])}

    # dataset: original time series
    # H: the forecast horizon
    # tensor_structure: a dictionary discribing the tensor structure in the form { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
    # freq: time series frequency
    # drop_incomplete: (Boolean) whether to drop incomplete samples, default is true
    #  so here does drops incomplete examples

    # https://stackoverflow.com/questions/4534438/typeerror-module-object-is-not-callable
    # calling a module

    train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)
    print (train_inputs.dataframe.head())

