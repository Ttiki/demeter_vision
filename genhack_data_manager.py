import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():
    directory = os.path.abspath(__file__)
    enclosingfolder = os.path.dirname(directory)
    enclosingfolder = os.path.dirname(enclosingfolder)
    noise_data_path_elhcs = os.path.join(enclosingfolder, 'data', 'noise.npy')

    # Load data from CSV files
    station_40_path = os.path.join(enclosingfolder, 'data', 'station_40.csv')
    station_49_path = os.path.join(enclosingfolder, 'data', 'station_49.csv')
    station_63_path = os.path.join(enclosingfolder, 'data', 'station_63.csv')
    station_80_path = os.path.join(enclosingfolder, 'data', 'station_80.csv')

    station_40 = pd.read_csv(station_40_path)
    station_49 = pd.read_csv(station_49_path)
    station_63 = pd.read_csv(station_63_path)
    station_80 = pd.read_csv(station_80_path)

    # Combine, preprocess, and filter data
    station_40_features = station_40.iloc[0:, :-1]
    station_40_yield = station_40.iloc[:, -1]
    station_49_features = station_49.iloc[:, :-1]
    station_49_yield = station_49.iloc[:, -1]
    station_63_features = station_63.iloc[:, :-1]
    station_63_yield = station_63.iloc[:, -1]
    station_80_features = station_80.iloc[:, :-1]
    station_80_yield = station_80.iloc[:, -1]

    # Combine data from all stations
    combined_data_features = pd.concat(
        [station_40_features, station_49_features, station_63_features, station_80_features], axis=1)
    combined_data_yield = pd.concat([station_40_yield, station_49_yield, station_63_yield, station_80_yield], axis=1)
    return pd.concat([combined_data_features, combined_data_yield], axis=1)


# Remove highly correlated features
def remove_highly_correlated_features(features):
    # Remove highly correlated features
    correlation_threshold = 0.8
    correlation_matrix = np.corrcoef(features, rowvar=False)
    correlated_features = np.where(np.abs(correlation_matrix) > correlation_threshold)
    unique_correlated_features = {tuple(sorted([i, j])) for i, j in zip(*correlated_features)}
    to_delete = [item for sublist in unique_correlated_features for item in sublist]
    return features.drop(features.columns[to_delete], axis=1)


# Standardize data using StandardScaler
def standardize_data(features):
    # Standardize data using StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(features)


# Split data into training and testing sets
def split_data(features, targets):
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
