# ds_loading.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dataset_station40 = pd.read_csv("data/station_40.csv")
dataset_station49 = pd.read_csv("data/station_49.csv")
dataset_station63 = pd.read_csv("data/station_63.csv")
dataset_station80 = pd.read_csv("data/station_80.csv")

# Normalize the datasets
dataset_station40 = scaler.fit_transform(dataset_station40)
dataset_station49 = scaler.fit_transform(dataset_station49)
dataset_station63 = scaler.fit_transform(dataset_station63)
dataset_station80 = scaler.fit_transform(dataset_station80)