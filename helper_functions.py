import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import svm

from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, CuDNNLSTM
from keras.optimizers import Adam


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from keras.layers import Dropout


def import_data(filename: str):
    data = pd.read_csv(filename, delimiter=";", decimal=",", thousands=".", index_col="date")
    data.index = pd.to_datetime(data.index, format='%d/%m/%Y')
    return data


def clean_data(df: pd.DataFrame):
    #data_cleaned = data[~(data['production'] >= 10000)]
    #data_cleaned = data_cleaned[~(data_cleaned['radiation'] >= 400)]
    #data_cleaned["production"] = data_cleaned["radiation"] / 10

    # Outlier detection and deletion
    q_low = df["production"].quantile(0.0125)
    q_hi = df["production"].quantile(0.99)
    df.loc[df['production'] > q_hi, 'production'] = np.nan
    df.loc[df['production'] < q_low, 'production'] = np.nan

    q_hi = df["radiation"].quantile(0.995)
    df.loc[df['radiation'] > q_hi, 'radiation'] = np.nan

    df['production'] = df['production'].interpolate(option='spline')
    df['radiation'] = df['radiation'].interpolate(option='spline')
    #df.fillna(method="bfill", inplace=True)

    return df


def cyclical_date(df):
    df["day"] = df.index.dayofyear

    df['day_sin'] = np.sin(2 * np.pi * (df['day'] - 1) / 365)
    df['day_cos'] = np.cos(2 * np.pi * (df['day'] - 1) / 365)

    return df


def normalize_data(data: pd.DataFrame):
    prod = np.array(data["production"])
    rad = np.array(data["radiation"])


    scaler_prod = preprocessing.MinMaxScaler()
    #scaler_prod = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaler_rad = preprocessing.MinMaxScaler()
    #scaler_rad = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    prod_scaled = scaler_prod.fit_transform(prod.reshape(-1, 1))
    rad_scaled = scaler_rad.fit_transform(rad.reshape(-1, 1))
    df = data
    df["production"] = prod_scaled
    df["radiation"] = rad_scaled

    return df, scaler_prod, scaler_rad


def divide_data(train_p, test_p, valid_p, data):
    train_data = data[:int(len(data)*train_p)]
    test_data = data[int(len(data)*train_p): len(data) - int(len(data) * valid_p)]
    valid_data = data[-int(len(data)*valid_p):]

    return train_data, test_data, valid_data


def build_lin_reg_model(X, y):
    model = LinearRegression().fit(X, y)

    return model


def build_poly_reg_model(X, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)


def build_svm_model(X, y):
    model = svm.SVR().fit(X, y)

    return model


# def to build decision tree model
def build_dt_model(X, y):
    model = DecisionTreeRegressor().fit(X, y)

    return model


# def to build random forest model
def build_rf_model(X, y):
    model = RandomForestRegressor().fit(X, y)

    return model


def prepare_windowed_data(df, win_size):
    X_train_windowed = []
    y_train_windowed = []
    for i in range(len(df)):
        if i+win_size < len(df):
            X_train_windowed.append(np.array(df.iloc[i:i+win_size]))
            y_train_windowed.append(np.array(df.iloc[i+win_size]["production"]))

    return np.array(X_train_windowed), np.array(y_train_windowed)


def build_lstm_model(learn_rate, activation, win_size):
    # activations = "relu", "tanh", "sigmoid"
    model = Sequential([Input((win_size, 2)),
                        CuDNNLSTM(8),
                        #Dense(2, activation=activation),
                        Dense(1)])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=learn_rate))
                  
    return model


def build_lstm_model_sincosattr(learn_rate, activation, win_size):
    # activations = "relu", "tanh", "sigmoid"
    # build LSTM model with multiple hidden layers

    model = Sequential([Input((win_size, 4)),
                        #Dense(8, activation=activation),
                        #Dropout(0.2),
                        CuDNNLSTM(8),
                        #Dropout(0.2),
                        #Dense(4, activation=activation),
                        #Dense(4, activation=activation),
                        Dense(1, activation=activation)])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=learn_rate))

    return model
