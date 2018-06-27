

import numpy as np
import pandas as pd

#from collections.abc import Mapping
# https://docs.python.org/3/library/collections.html#collections.UserDict
from collections import UserDict

# this is for class inheritated
# cannot find userDict so comment it 
# https://stackoverflow.com/questions/1392396/advantages-of-userdict-class-in-python
# http://www.kr41.net/2016/03-23-dont_inherit_python_builtin_dict_type.html
class TimeSeriesTensor(UserDict):
#class TimeSeriesTensor(dict):
#class TimeSeriesTensor(Mapping):

    
    # A dictionary of tensors for input into the RNN model
    
    # Use this class to:
    #   1. Shift the values of the time series to create a Pandas dataframe containing all the data
    #      for a single training example
    #   2. Discard any samples with missing values
    #   3. Transform this Pandas dataframe into a numpy array of shape 
    #      (samples, time steps, features) for input into Keras

    # The class takes the following parameters:
    #    - **dataset**: original time series
    #    - **H**: the forecast horizon
    #    - **tensor_structures**: a dictionary discribing the tensor structure of the form
    #          { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
    #          if features are non-sequential and should not be shifted, use the form
    #          { 'tensor_name' : (None, [feature, feature, ...])}
    #    - **freq**: time series frequency
    #    - **drop_incomplete**: (Boolean) whether to drop incomplete samples
    
    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())
        
        self.dataframe = self._shift_data(H, freq, drop_incomplete)
        self.data = self._df2tensors(self.dataframe)
    
    # for support Mapping glass
    '''
    # http://www.kr41.net/2016/03-23-dont_inherit_python_builtin_dict_type.html
    def __getitem__(self, key):
        return self.dataframe[key]
    def __iter__(self):
        return iter(self.dataframe)    # ``ghost`` is invisible
    def __len__(self):
        return len(self.dataframe)
    '''
    
    def _shift_data(self, H, freq, drop_incomplete):
        
        # Use the tensor_structures definitions to shift the features in the original dataset.
        # The result is a Pandas dataframe with multi-index columns in the hierarchy
        #     tensor - the name of the input tensor
        #     feature - the input feature to be shifted
        #     time step - the time step for the RNN in which the data is input. These labels
        #         are centred on time t. the forecast creation time
        df = self.dataset.copy()
        
        idx_tuples = []
        for t in range(1, H+1):
            df['t+'+str(t)] = df[self.target].shift(t*-1, freq=freq)
            idx_tuples.append(('target', 'y', 't+'+str(t)))

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            dataset_cols = structure[1]
            
            for col in dataset_cols:
            
            # do not shift non-sequential 'static' features
                if rng is None:
                    df['context_'+col] = df[col]
                    idx_tuples.append((name, col, 'static'))

                else:
                    for t in rng:
                        sign = '+' if t > 0 else ''
                        shift = str(t) if t != 0 else ''
                        period = 't'+sign+shift
                        shifted_col = name+'_'+col+'_'+period
                        df[shifted_col] = df[col].shift(t*-1, freq=freq)
                        idx_tuples.append((name, col, period))
                
        df = df.drop(self.dataset.columns, axis=1)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
        df.columns = idx

        if drop_incomplete:
            df = df.dropna(how='any')

        return df
    
    
    def _df2tensors(self, dataframe):
        
        # Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These
        # arrays can be used to input into the keras model and can be accessed by tensor name.
        # For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named
        # "target", the input tensor can be acccessed with model_inputs['target']
    
        inputs = {}
        y = dataframe['target']
        y = y.as_matrix()
        inputs['target'] = y

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            cols = structure[1]
            tensor = dataframe[name][cols].as_matrix()
            if rng is None:
                tensor = tensor.reshape(tensor.shape[0], len(cols))
            else:
                tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
                tensor = np.transpose(tensor, axes=[0, 2, 1])
            inputs[name] = tensor

        return inputs
    
    
    def subset_data(self, new_dataframe):
        
        # Use this function to recreate the input tensors if the shifted dataframe
        # has been filtered.
        
        self.dataframe = new_dataframe
        self.data = self._df2tensors(self.dataframe)