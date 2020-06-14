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

def test(model, data, data_percentage = 100):
    X=[]
    Y=[]
    dataset_size = 24*len(data)*data_percentage//100 - 1 # -1 for the last item
    corrects = 0
    incorrects = 0
    total_tests = 0
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
            for current_temperature in [lower_temp, upper_temp]: # Cambiar rango a +-10 de temperatura de afuera
                for car_connected in [False, True]:
                    for car_energy in range(0, car_connected * 30 + 1, 5):
                        """Iterate over actions"""
                        min_cost = 10000000
                        min_action = [0,0,0]
                        min_pred_cost = 10000000
                        min_pred_action = [0,0,0]
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
                                    y_pred = model.predict([x])
                                    # for real cost
                                    if y < min_cost:
                                        min_action = [b,c,air]
                                        min_cost = y
                                    # for NN cost
                                    if y_pred < min_pred_cost:
                                        min_pred_action = [b,c,air]
                                        min_pred_cost = y
                    # Do the shit
                    if min_action == min_pred_action:
                        corrects += 1
                    else:
                        incorrects += 1
                    total_tests += 1
        end_time = time.time()
        print(f'\nTiempo: {end_time - initial_time}')
        print(f'{corrects}, {incorrects}')
        print(f'Total choices accuracy: {100*corrects/total_tests}%')
    return X, Y

if __name__ == "__main__":
    filename = sys.argv[1]
    percentage = int(sys.argv[2])
    mlp_saved = load(filename)
    X,Y = test(mlp_saved, data_2018, data_percentage=percentage)
    # Caluclar MSE
    r2 = mlp_saved.score(X,Y)
    print(f'R squared: {r2}')
    predicted = mlp_saved.predict(X)
    real = Y
    mse = sum([ (y_pred - y_real)**2 for y_pred, y_real in zip(predicted, real)]) / len(predicted)
    print(f'Mean Squared Error: {mse}')
    for i in range(20):
        print(predicted[i], real[i])
    print("The end")
