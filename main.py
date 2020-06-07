import numpy as np
from sklearn import neural_network
import json
from state import state
from utils import *
import time
from data import *
import sys
from joblib import dump, load
import datetime
import numpy as np
import pandas as pd

def fit(model, x, y):
    return model.fit(x, y)

def train(model, filename):
    # Open pandas dataset
    dtype = {
        'total': np.float64,
        'temperature': np.int32,
        'humidity': np.float64,
        'solar': np.float64,
        'car_connected': np.int32,
        'car_energy': np.int32,
        'battery_energy': np.int32,
        'current_temperature': np.int32,
        'b': np.int32,
        'c': np.int32,
        'air': np.int32,
        'cost': np.int32
    }
    x = pd.read_csv('train_data.csv', dtype=dtype)

    y = x.pop('cost').values
    fit(model, x, y)


if __name__ == "__main__":
    solver = sys.argv[1]
    mlp = neural_network.MLPRegressor(hidden_layer_sizes=(16,), solver=solver)
    train(mlp, 'train_data.csv')
    print("Saving model")
    dt_now= datetime.datetime.now().isoformat()
    dump(mlp, f'mlp-{dt_now}.joblib')
    print("Finished")
