# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:14:15 2020

@author: mh
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Masking
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as k



class WTTE:
    
    def __init__(self, max_time, n_features):
        self.model = None
        self.max_time = max_time
        self.n_features = n_features
        
        
    def __call__(self):
        return self.model
    
    def weibull_loglik_discrete(self, y_true, ab_pred, name=None):
        y_ = y_true[:, 0]
        u_ = y_true[:, 1]
        a_ = ab_pred[:, 0]
        b_ = ab_pred[:, 1]
    
        hazard0 = k.pow((y_ + 1e-35) / a_, b_)
        hazard1 = k.pow((y_ + 1) / a_, b_)
    
        return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)
    
    def activate(self, ab):
        a = k.exp(ab[:, 0])
        b = k.softplus(ab[:, 1])
    
        a = k.reshape(a, (k.shape(a)[0], 1))
        b = k.reshape(b, (k.shape(b)[0], 1))
    
        return k.concatenate((a, b), axis=1)
    
    def compile_model(self):
        self.model = Sequential()
        # Mask parts of the lookback period that are all zeros (i.e., unobserved) so they don't skew the model
        self.model.add(Masking(mask_value=0., input_shape=(self.max_time, self.n_features)))
        # LSTM is just a common type of RNN. You could also try anything else (e.g., GRU).
        self.model.add(LSTM(20, input_dim=self.n_features))
        # We need 2 neurons to output Alpha and Beta parameters for our Weibull distribution
        self.model.add(Dense(2))
        # Apply the custom activation function mentioned above
        self.model.add(Activation(self.activate))
        # Use the discrete log-likelihood for Weibull survival data as our loss function
        self.model.compile(loss=self.weibull_loglik_discrete, optimizer=RMSprop(lr=.001))

    def get_model(self):
        return self.model

    def fit(self, x, y, epochs=10, batch_size=64, verbose=2, validation_data=None):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=validation_data)
