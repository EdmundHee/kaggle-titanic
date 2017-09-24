
# SOOURCE https://www.kaggle.com/cstahl12/titanic-with-keras

from __future__ import print_function
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout


raw_data = pd.read_csv('../data/train.csv', index_col=0)
raw_data['is_test'] = 0

train_data = pd.read_csv('../data/test.csv', index_col=0)
train_data['is_test'] = 0

all_data = pd.concat((raw_data, train_data), axis=0)
print(all_data)
