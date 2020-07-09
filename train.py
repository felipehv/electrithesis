import numpy as np
from sklearn import neural_network
import json
import time
import sys
from joblib import dump, load
import datetime
import numpy as np
import pandas as pd
import joblib

from .utils import *
from .state import state

def fit(model, x, y):
    return model.fit(x, y)

def train(model, x, y):
    # Open pandas dataset
    print("Iniciando entrenamiento")
    model.fit(x, y)
    print("Fin del entrenamiento")


if __name__ == "__main__":
    print("Iniciando")
    solver = sys.argv[1]
    # client = Client(processes=False, threads_per_worker=4,
    #             n_workers=2, memory_limit='3GB')
    # print(client)
    mlp = neural_network.MLPRegressor(
        hidden_layer_sizes=(16,), 
        solver=solver, 
        verbose=10,
        activation='relu',
        batch_size=32,
        learning_rate_init=0.01, # funciona mejor
        tol=1e-3,
        early_stopping=False,
        epsilon=1e-4,
        n_iter_no_change=3)
    # mlp = load('mlp.joblib')
    initial_time = time.now()
    train(mlp, 'train_data.csv')
    end_time = time.now()
    print("Time training:")
    print("Saving model")
    dt_now= datetime.datetime.now().isoformat()
    dump(mlp, f'mlp.joblib')
    print("Finished")
