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
    return model.partial_fit(x, y)

def train(model, data, data_percentage = 100, epochs = 1):
    dataset_size = 24*len(data)*data_percentage//100 - 1 # -1 for the last item
    X = []
    Y = []
    for epoch in range(epochs):
        print(f'Epoch {epoch}', dataset_size)
        for t in range(dataset_size-1):
            initial_time = time.time()
            print(f'Etapa: {t}/{dataset_size-1}\n')
            current_hour = data[t // 24][t % 24]
            next_hour = data[(t+1) // 24][(t+1) % 24]

            """Iterate over states (battery status, car...)"""
            for battery_energy in range(0, 10):
                print(f'\r battery_energy: {battery_energy}', end='')
                lower_temp = current_hour['temperature'] - 10
                upper_temp = current_hour['temperature'] + 10
                for current_temperature in range(lower_temp, upper_temp + 1): # Cambiar rango a +-10 de temperatura de afuera
                    for car_connected in [False, True]:
                        for car_energy in range(0, car_connected * 30 + 1): # Cambiar rango a +-10 de temperatura de afuera
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
                                        Y.append(round(y))
            end_time = time.time()
            print(f'\nTiempo: {end_time - initial_time}')
    print('Training')
    fit(model, X, Y)
    print('End training')

if __name__ == "__main__":
    percentage = int(sys.argv[1])
    epochs = int(sys.argv[2])
    mlp = neural_network.MLPRegressor(hidden_layer_sizes=(16,), solver='adam')
    train(mlp, data_2016, data_percentage=percentage)
    print("Saving model")
    dt_now= datetime.datetime.now().isoformat()
    dump(mlp, f'mlp-{dt_now}.joblib')
    print("Finished")
