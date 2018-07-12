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



# Define the funtion to make single sequence prediction 
# based on scoring encoder-decoder
def predict_single_sequence(single_input_seq, horizon, n_features, encoder_model, decoder_model):
    # apply encoder model to the input_seq to get state
    states_value = encoder_model.predict(single_input_seq)
    
    # get input for decoder's first time step (which is encoder input at time t)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    dec_input = np.zeros((1, 1, n_features))

    # https://stackoverflow.com/questions/15535205/what-does-1-mean-do-in-python
    # https://www.digitalocean.com/community/tutorials/how-to-index-and-slice-strings-in-python-3
    # normal
    # proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
    # [:5]
    # output 0 to 5(excluding)
    # [-3]
    # reverse index is minus and it starts with -1 -2 -3...
    # -3 is to output the character with index = -3
    # [-4:-1]
    # output character with index = -4 to index = -1
    # [::-1]
    # stride = -1, output total string reversely
    # [::-2]
    # stride = -2

    # [:-1] 
    # upper bound = -1, so remove the last charaacter
    # [-1:]
    # low bound = -1, 
    # output the last character, same as [-1]
    # [-1:1:-1]
    # reverse string
    #  index -1 to index 1(excluding)
    # ex: input = 'abcedfg'
    # cedfg
    # string is reverse output becaasue striide is -1
    # so output 'gfedc'
    # [:]
    # means nothing constraint
    # just output original string


    # https://www.zhihu.com/question/22686450
    # https://stackoverflow.com/questions/11367902/negative-list-index/11367936
    # ?more examples
    # ? not fully understand

    # https://stackoverflow.com/questions/31061625/accessing-slice-of-3d-numpy-array
    # python slice
    
    # for debug
    # this is the first row for encoding input data
    #print (single_input_seq)


    # access the kast element of the vector
    # why starting from the last one
    # it is becasue this is time T
    dec_input[0, 0, 0] = single_input_seq[0, -1, :]


    # for debug
    #print (dec_input[0, 0, 0])
    
    # create final output placeholder
    # ? only one elements in single_input_seq is used to predict three outputs for three timestamps
    # becasue the previous output will be used as next time input 
    output = list()
    # collect predictions
    for t in range(horizon):
        # predict next value
        yhat, h = decoder_model.predict([dec_input] + [states_value])
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h]
        # update decoder input to be used as input for next prediction
        dec_input[0, 0, 0] = yhat
        
    return np.array(output)

# Define the funtion to make multiple sequence prediction 
# based on scoring encoder-decoder
def predict_multi_sequence(input_seq_multi, horizon, n_features):
    # create output placeholder
    predictions_all = list()
    for seq_index in range(input_seq_multi.shape[0]):       
        # Take one sequence for decoding
        input_seq = input_seq_multi[seq_index: seq_index + 1]
        # Generate prediction for the single sequence
        predictions = predict_single_sequence(input_seq, horizon, n_features)
        # store all the sequence prediction
        predictions_all.append(predictions)
        
    return np.array(predictions_all)


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
    # Compute the minimum and maximum to be used for later scaling.
    y_scaler.fit(train[['load']])

    X_scaler = MinMaxScaler()
    # fit to data and transform it
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    # ? accroding to exampke, why needs to have two scalers
    # ? why traing data is using  0-1 to transform 
    # ? but y_scaler is used for testing data by maximum and minimum
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

    # for debug
    # dictionary tyoe
    # 'target'
    # 'encoder_input'
    # 'decoder_input'
    # as key to access different data
    print (valid_inputs)
    

    print (valid_inputs.dataframe.head())

    # parameter
    #  not sure how this gets seup, might be by hyper parameter tune up
    BATCH_SIZE = 32
    LATENT_DIM = 5
    EPOCHS = 50

    # define training encoder
    # really different from lecture's example
    # https://keras.io/getting-started/sequential-model-guide/#specifying-the-input-shape
    # specify input with None None indicates that any positive integer may be expected).
    #  ? encoder input is a tuple
    # (seqence_length , input kength)
    # ? no using  rolling feature of normal flow
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
    # https://keras.io/layers/recurrent/
    # initial_state should be a list of tensors
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
    # ? why go on thos way, why not basically going same as model add as lecture 4
    # https://github.com/say543/RNNForTimeSeriesForecastTutorial/blob/master/4_multi_step_encoder_decoder_simple.ipynb

    # model class API
    # https://keras.io/models/model/#model-class-api
    # https://keras.io/getting-started/functional-api-guide/
    # here using  multiple input, sinlge output (providing output is for dense)
    # Note that by calling a model you aren't just reusing the architecture of the model, you are also reusing its weights.
    # [] should be forming a list
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



    # implement inference model
    # ? not sure why this is different from training model, should it use previous model to do

    # build ingerence encoder model
    # ? why using encoder input / encoder states again
    # encoder_input seems only providng dimention
    # ? how about encoder_states, is it output from training output for reuse or just dimention

    # for debug 
    # cannot tell from output, only know they are two objects
    # https://keras.io/layers/recurrent/
    # initial_state should be a list of tensors
    ## print(encoder_input)
    ## print(encoder_states)

    # https://keras.io/layers/recurrent/
    # initial_state should be a list of tensors
    encoder_model = Model(encoder_input, encoder_states)

    # build ingerence decoder model
    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    # form a list since initial_state needs a list
    decoder_states_input = [decoder_state_input_h]

    # reuse decoder_GRU's archutecture 
    decoder_output, state_h = decoder_GRU(decoder_input, initial_state=decoder_states_input)
    decoder_states = [state_h]
    decoder_output = decoder_dense(decoder_output)

    # ? why adding here, do not understand  also using +
    # looks like list operation
    # for debug
    # it will be a list having two tensors as elements
    # print ([decoder_input] + decoder_states_input)
    # basically the same as this i guess ?
    # ? wrong this will say unhashable list becasue decoder_states_input / decoder_states are list already
    # decoder_model = Model([decoder_input, decoder_states_input], [decoder_output, decoder_states])
    # so using list combine operation
    # https://stackoverflow.com/questions/1720421/how-to-concatenate-two-lists-in-python
    decoder_model = Model([decoder_input] + decoder_states_input, [decoder_output] + decoder_states)


    #################################################################
    # example of single sequence prediction
    ##################################################################
    #  predict_single_sequence will use encoder_model as global parameter
    #  so passing it 
    #print(predict_single_sequence(valid_inputs['encoder_input'][0:1], HORIZON, 1))
    # ? does 1 mean input size
    print(predict_single_sequence(valid_inputs['encoder_input'][0:1], HORIZON, 1, encoder_model, decoder_model))

    #################################################################
    # example of output sequence prediction
    # using  single sequence prediction as a subroutine
    ##################################################################
    look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
    # energy is a input file
    test = energy.copy()[test_start_dt:][['load']]
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    # ? y_scaler search this code above. why it goes this way?
    test[['load']] = y_scaler.transform(test)
    test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)


