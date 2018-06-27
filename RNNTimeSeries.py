import os
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime as dt
from collections import UserDict

from TimeSeriesTensor import TimeSeriesTensor

# Using TensorFlow backend.
# so need to pip install tensorflow
from keras.models import Model, Sequential
from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten, Input
from keras.callbacks import EarlyStopping

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



    # extra data as valid inputs i guess
    look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load']]
    valid[['load']] = X_scaler.transform(valid)
    valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)
    print (valid_inputs.dataframe.head())

    # parameter
    #  not sure how this gets seup, might be by hyper parameter tune up
    BATCH_SIZE = 32
    LATENT_DIM = 5
    EPOCHS = 50

    # define training encoder
    # really different from lecture's example
    # https://keras.io/getting-started/sequential-model-guide/#specifying-the-input-shape
    # specify input with None
    encoder_input = Input(shape=(None, 1))
    # using GUR is differnt from RNN
    # https://keras.io/layers/recurrent/
    # return_state: Boolean. Whether to return the last state in addition to the output.
    encoder = GRU(LATENT_DIM, return_state=True)
    # proviate input and extra state different from output
    encoder_output, state_h = encoder(encoder_input)
    # ? how does this line work ? it looks like accessing dictionary by key
    # https://openhome.cc/Gossip/CodeData/PythonTutorial/ContainerFlowComprehensionPy3.html
    # it should be initial as list dadastrucutre i guees 
    # this will output an object
    print (state_h)
    encoder_states = [state_h]
    print (encoder_states)


    # define training decoder
    decoder_input = Input(shape=(None, 1))
    # why setup return sequences ture at the second layer?
    # yes, based on coding it does make sense
    # why latent_dim is 6 . i think it should be 3
    # latent_dim is the dimention of GUR , notinput
    decoder_GRU = GRU(LATENT_DIM, return_state=True, return_sequences=True)
    # ise _ to get output states
    decoder_output, _ = decoder_GRU(decoder_input, initial_state=encoder_states)


    # https://keras.io/getting-started/functional-api-guide/#all-models-are-callable-just-like-layers
    # https://keras.io/layers/wrappers/
    # https://blog.csdn.net/u012193416/article/details/79477220
    # https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
    # ? not very understand about syntax 
    # ? might be following lecture 4 to recode it better
    decoder_dense = TimeDistributed(Dense(1))
    decoder_output = decoder_dense(decoder_output)


    # also GRU
    # ? not do sequentail and model add here
    # ? why go on thos way
    # basically same as model add as lecture 4
    # https://github.com/say543/RNNForTimeSeriesForecastTutorial/blob/master/4_multi_step_encoder_decoder_simple.ipynb
    model = Model([encoder_input, decoder_input], decoder_output)


    # optimization function
    # https://github.com/say543/RNNForTimeSeriesForecastTutorial/blob/master/slides/RNN%20For%20Time%20Series%20Forecasting%20Tutorial.pdf
    # mse : Mean-squared-error
    # how to select optimization ?
    # i used to learn that is it provided by Maximal likelihood optimization
    # https://keras.io/getting-started/sequential-model-guide/#compilation
    # having adagrad, studying in th future
    model.compile(optimizer='RMSprop', loss='mse')

    # output model
    # i borrow from 
    # https://github.com/say543/RNNForTimeSeriesForecastTutorial/blob/master/4_multi_step_encoder_decoder_simple.ipynb
    # ? not sure how to read it
    # https://keras.io/models/about-keras-models/#about-keras-models
    model.summary()


    # introducing early stop
    # ? why need to have early stop
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

    # get label data for train data and valid data
    # no user dictionary so change it 
    train_target = train_inputs['target'].reshape(train_inputs['target'].shape[0], train_inputs['target'].shape[1], 1)
    valid_target = valid_inputs['target'].reshape(valid_inputs['target'].shape[0], valid_inputs['target'].shape[1], 1)
    #train_target = train_inputs.dataframe.target.reshape(train_inputs.dataframe.target.shape[0], train_inputs.dataframe.target.shape[1], 1)
    #valid_target = valid_inputs.dataframe.target.reshape(valid_inputs.dataframe.target.shape[0], valid_inputs.dataframe.target.shape[1], 1)


    # train
    # why needs to have both decoder inpput and encoder input as inputs
    # ? i think having decoder input is good enough
    # no user dictionary so change it 
    model.fit([train_inputs['encoder_input'], train_inputs['decoder_input']],
          train_target,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=([valid_inputs['encoder_input'], valid_inputs['decoder_input']], valid_target),
          callbacks=[earlystop],
          verbose=1)
    #model.fit([train_inputs.dataframe.encoder_input, train_inputs.dataframe.decoder_input],
    #      train_target,
    #      batch_size=BATCH_SIZE,
    #      epochs=EPOCHS,
    #      validation_data=([valid_inputs.dataframe.encoder_input, valid_inputs.dataframe.decoder_input], valid_target),
    #      callbacks=[earlystop],
    #      verbose=1)



