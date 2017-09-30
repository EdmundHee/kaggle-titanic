
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

def get_title_last_name(name):
    full_name = name.str.split(', ', n=0, expand=True)
    titles = full_name[1].str.split('.', n=0, expand=True)
    titles = titles[0]
    return titles

def get_titles_from_names(df):
    df['Title'] = get_title_last_name(df['Name'])
    df = df.drop(['Name'], axis=1)
    return df

def get_dummy_cats(df):
    return(pd.get_dummies(df, columns=['Title', 'Pclass', 'Sex', 'Embarked','Cabin', 'Cabin_letter']))

def get_cabin_letter(df):
    df['Cabin'].fillna('Z', inplace=True)
    df['Cabin_letter'] = df['Cabin'].str[0]
    return df

def process_data(df):
    # preprocess titles, cabin, embarked
    df = get_titles_from_names(df)
    df['Embarked'].fillna('S', inplace=True)
    df = get_cabin_letter(df)

    # drop remaining features
    df = df.drop(['Ticket', 'Fare'], axis=1)

    # create dummies for categorial features
    df = get_dummy_cats(df)
    return(df)

proc_data = process_data(all_data)
proc_train = proc_data[proc_data['is_test'] == 0]
proc_test = proc_data[proc_data['is_test'] == 1]

proc_data.head()

for_age_train = proc_data.drop(['Survived', 'is_test'], axis=1).dropna(axis=0)
x_train_age = for_age_train.drop('Age', axis=1)
y_train_age = for_age_train['Age']

# create model
tmodel = Sequential()
tmodel.add(Dense(input_dim=x_train_age.shape[1], units=128,
                 kernel_initializer='normal', bias_initializer='zeros'))
tmodel.add(Activation('relu'))

for i in range(0, 8):
    tmodel.add(Dense(units=64, kernel_initializer='normal',
                     bias_initializer='zeros'))
    tmodel.add(Activation('relu'))
    tmodel.add(Dropout(.25))

tmodel.add(Dense(units=1))
tmodel.add(Activation('linear'))
tmodel.compile(loss='mean_squared_error', optimizer='rmsprop')
tmodel.fit(x_train_age.values, y_train_age.values, epochs=800, verbose=2)

train_data = proc_train
train_data.loc[train_data['Age'].isnull()]

to_pred = train_data.loc[train_data['Age'].isnull()].drop(['Age', 'Survived', 'is_test'], axis=1)
p = tmodel.predict(to_pred.values)
train_data['Age'].loc[train_data['Age'].isnull()] = p
