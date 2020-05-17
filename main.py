import numpy as np
from sklearn import neural_network
import json
from state import state
from utils import *
import time
from data import *

EPOCHS = 1

mlp = neural_network.MLPRegressor(hidden_layer_sizes=(16,), solver='sgd', max_iter=1)

def fit(model, x, y):
    return model.fit(x, y)

for _ in range(EPOCHS):
    X = []
    Y = []
    for t in range(24*len(data_2016)-1):
        initial_time = time.time()
        print(f'Etapa: {t}/{4*len(data_2016)-1}\n')
        current_hour = data_2016[t // 24][t % 24]
        next_hour = data_2016[(t+1) // 24][(t+1) % 24]

        """Iterate over states (battery status, car...)"""
        for car_connected in [False, True]:
            for car_energy in range(30):
                for battery_energy in range(0, 10):
                    for current_temperature in range(0, 35): # Cambiar rango a +-10 de temperatura de afuera
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