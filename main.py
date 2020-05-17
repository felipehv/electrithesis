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

def fit(model, x, y):
    return model.fit(x, y)

def train(model, data, data_percentage = 100, epochs = 1):
    dataset_size = len(data)//100*data_percentage
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        X = []
        Y = []
        for t in range(24*dataset_size-1):
            initial_time = time.time()
            print(f'Etapa: {t}/{24*dataset_size-1}\n')
            current_hour = data[t // 24][t % 24]
            next_hour = data[(t+1) // 24][(t+1) % 24]

            """Iterate over states (battery status, car...)"""
            for car_connected in [False, True]:
                for car_energy in range(30):
                    for battery_energy in range(0, 10):
                        lower_temp = current_hour['temperature'] - 10
                        upper_temp = current_hour['temperature'] + 10
                        for current_temperature in range(lower_temp, upper_temp + 1): # Cambiar rango a +-10 de temperatura de afuera
                            """Iterate over actions"""
                            for b in range(2):
                            # battery actions: 0: CHARGE, 1: USE
                                for c in range(2):
                                    # car actions 0: CHARGE, 1: USE
                                    for air in range(5):
                                        x,y = state(
                                                    current_hour,
                                                    next_hour,
                                                    car_connected,
                                                    car_energy,
                                                    battery_energy,
                                                    current_temperature,
                                                    b, c, air
                                                )
                                        X.append(x)
                                        Y.append(y)
            end_time = time.time()
            print(f'\nTiempo: {end_time - initial_time}')
        fit(mlp, X, Y)

if __name__ == "__main__":
    percentage = int(sys.argv[1])
    epochs = int(sys.argv[2])
    mlp = neural_network.MLPRegressor(hidden_layer_sizes=(16,), solver='sgd', max_iter=1)
    train(mlp, data_2016, data_percentage=percentage, epochs=epochs)
    print("Saving model")
    dt_now= datetime.datetime.now().isoformat()
    dump(mlp, f'mlp-{dt_now}.joblib')
