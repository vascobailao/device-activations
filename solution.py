#!/usr/bin/env python3

"""
This is a sample program for the data science challenge and can be used as a
starting point for a solution.

It will be run as follows;
    solution.py <current time> <input file name> <output file name>

Current time is the current hour and input file is all measured values from
the activation detector in each room for the past few hours.
"""

import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import sys

import warnings
warnings.filterwarnings("ignore")


def get_missing_ts(df, devices_names, all_ts):
    """
    get_missing_ts get missing timestamps per device

    :param df: device aggregated data by hour
    :param device_names: unique devices
    :param all_ts: timestamps of the aggregated data
    
    :return: data with the missing timestamps
    """ 
    
    df_all_ts = df.copy()
    df_dict = dict()
    
    for dev in devices_names:
        
        df_dev = df[df["device"] == dev]
        df_dev["time"] = pd.to_datetime(df_dev["time"])
        df_dev_ts_values = df_dev["time"].tolist()
        
        df_dict[f"{dev}"] = {}
        df_dict[f"{dev}"]["diff"] = list(set(all_ts).difference(set(df_dev_ts_values)))
        df_dict[f"{dev}"]["len_diff"] = len(df_dict[f"{dev}"]["diff"])
        df_dict[f"{dev}"]["data"] = pd.DataFrame(data=zip([dev]*df_dict[f"{dev}"]["len_diff"], df_dict[f"{dev}"]["diff"]), 
                                                 columns=["device", "time"]).assign(device_activated = 0)

        df_all_ts = pd.concat([df_all_ts, df_dict[f"{dev}"]["data"]])
        
    return df_all_ts


def create_features(df, create_target=False):
    """
    create_features create temporal features and encode categorical ones

    :param df: device aggregated data by hour including missing ts
    :param create_target: flag to create target for previous data
    """ 
    
    b = [0,4,8,12,16,20,24]
    l = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
    
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df['time'].dt.hour
    df["dow"] = df['time'].dt.weekday
    df["is_weekend"] = df['time'].dt.day_name().isin(['Saturday', 'Sunday'])
    df["woy"] = df['time'].dt.isocalendar().week
    
    df['session'] = pd.cut(df['hour'], bins=b, labels=l, include_lowest=True)
    df_ohe = pd.get_dummies(df[["device", "session"]])
    
    if (create_target):
        df["label"] = np.where(df["device_activated"] > 0, 1, 0)
        df = pd.concat([df.drop(columns=["device", "session", "device_activated"]), df_ohe], axis = 1)
        
    else:
        df = pd.concat([df.drop(columns=["device", "session"]), df_ohe], axis = 1)
    
    return df

def temporal_train_test_split(df, train_end='2016-08-01', test_start='2016-08-07'):
    """
    split_transform_previous temporal split data into train/test 

    :param df: data with all the created features
    :param train_end: end of training data
    :param test_start: beginning of test data
    """ 
    
    df["time"] = pd.to_datetime(df["time"])
    
    df_ord = df.sort_values(by=["time"])
    
    _df = df_ord.set_index('time')
    
    df_train = _df[:train_end].reset_index()
    df_test = _df[test_start:].reset_index()
    
    
    return (df_train.drop(columns=["time", "label"]), 
            df_test.drop(columns=["time", "label"]),
            df_train[["label"]], df_test[["label"]])

def split_transform_previous(df):
    """
    split_transform_previous split data into train/test 
                             standardize data

    :param df: data with all the created features
    """ 
    
    X_train, X_test, y_train, y_test = temporal_train_test_split(df)
    
    std_cols = X_train.columns[~X_train.columns.str.contains('device_|is_weekend|session_')].tolist()
    
    scaler = StandardScaler()

    X_train[std_cols] = scaler.fit_transform(X_train[std_cols])
    X_test[std_cols] = scaler.transform(X_test[std_cols])
    
    return scaler, X_train, X_test, y_train, y_test


def transform_next(df, scaler):
    """
    transform_next standardize data for the next 24 slots

    :param df: data with 24 hourly slots per device
    :param scaler: standard scaler fitted with train data
    """ 
    
    df_aux = df.drop(columns=["time"])
    
    std_cols = df_aux.columns[~df_aux.columns.str.contains('device_|is_weekend|session_')].tolist()

    df_aux[std_cols] = scaler.transform(df_aux[std_cols])
    
    return df_aux

def train_lr(X_train, X_test, y_train, y_test):
    """
    train_lr train logistic regression model

    :param X_train: train features
    :param X_test: test features
    :param y_train: train target
    :param y_test: test target
    """ 
    
    lr = LogisticRegression()
    lr = lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    
    print(classification_report(pred, y_test))
    
    return lr
                    
                   

def predict_future_activation(current_time, previous_readings):
    """This function predicts future hourly activation given previous sensings.

    It's probably not the best implementation as it just returns a random
    guess.
    """
    # Aggregate data at the hour level by device
    dfagg = previous_readings.set_index('time').groupby([pd.Grouper(freq='H'), 'device']).sum().reset_index()
    dfagg["time"] = pd.to_datetime(dfagg["time"])

    # Make 24 predictions for each hour starting at the next full hour
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')
    
    device_names = sorted(previous_readings.device.unique())
    
    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, device_names))
    predictions = pd.DataFrame(xproduct, columns=['time', 'device'])
    
    # Get previous timestamps
    ts_previous = pd.date_range("2016-07-01 00:00:00", "2016-08-31 23:00:00", freq="H")
    
    # For previous data - get misssing timestamps and create features
    df_previous = get_missing_ts(dfagg, device_names, ts_previous)
    df_previous = create_features(df_previous, create_target=True)
    
    # For next data - create features
    df_next = create_features(predictions, create_target=False)
    
    # Split into train/test and standardize
    scaler, X_train, X_test, y_train, y_test = split_transform_previous(df_previous)
    
    # Standardize next 24 hourly slots
    df_next_std = transform_next(df_next, scaler)
    
    # Train LR model and evalute test performance
    lr = train_lr(X_train, X_test, y_train, y_test)
    
    predictions = predictions[["time", "device"]]
    predictions.set_index('time', inplace=True)
    predictions['activation_predicted'] = lr.predict(df_next_std)
    
    
    return predictions


if __name__ == '__main__':

    current_time, in_file, out_file = sys.argv[1:]

    previous_readings = pd.read_csv(in_file, parse_dates=["time"])
    
    result = predict_future_activation(current_time, previous_readings)
    result.to_csv(out_file)
