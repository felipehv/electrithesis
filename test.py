import numpy as np
from sklearn import neural_network
import json
import time
import sys
from joblib import dump, load
import datetime

from .data import *
from .utils import *
from .state import state

def test(model, data, data_percentage = 100):
    X=[]
    Y=[]
    dataset_size = 24*len(data)*data_percentage//100 - 1 # -1 for the last item
    corrects = 0
    incorrects = 0
    total_tests = 0
    places = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 'rest': 0}
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
            for current_temperature in range(lower_temp, upper_temp, 2): # Cambiar rango a +-10 de temperatura de afuera
                for car_connected in [False, True]:
                    for car_energy in range(0, car_connected * 30 + 1, 5):
                        """Iterate over actions"""
                        min_cost = 10000000
                        min_action = [0,0,0]
                        real_costs = []
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
                                    real_costs.append(y)
                                    # for real cost
                                    if y < min_cost:
                                        min_action = [b,c,air]
                                        min_cost = y
                                    # for NN cost
                                    if y_pred < min_pred_cost:
                                        min_pred_action = [b,c,air]
                                        min_pred_cost = y_pred

                        # Calcular lugar de predicción
                        pred_cost_index = 10 * min_pred_action[0] + 5 * min_pred_action[1] + min_pred_action[2]
                        pred_cost = real_costs[pred_cost_index]
                        real_costs.sort()
                        place = real_costs.index(pred_cost) + 1
    
                        if place not in places:
                            places['rest'] += 1
                        else:
                            places[place] += 1
                        total_tests += 1
        end_time = time.time()
        print(f'\nTiempo: {end_time - initial_time}')
        print(f'{places[1]}, {total_tests - places[1]}')
        print(f'1st: {places[1]} - 2nd: {places[2]} - 3rd: {places[3]} - 4th: {places[4]} - rest: {places["rest"]}')
        print(f'Total choices accuracy: {100*places[1]/total_tests}%')
    return X, Y

if __name__ == "__main__":
    filename = sys.argv[1]
    percentage = int(sys.argv[2])
    mlp_saved = load(filename)
    data_2018 = load_data('.', which='2018')
    X,Y = test(mlp_saved, data_2018, data_percentage=percentage)
    # Caluclar MSE
    r2 = mlp_saved.score(X,Y)
    print(f'R squared: {r2}')
    predicted = mlp_saved.predict(X)
    real = Y
    mse = sum([ (y_pred - y_real)**2 for y_pred, y_real in zip(predicted, real)]) / len(predicted)
    print(f'Mean Squared Error: {mse}')
    for i in range(30):
        print(predicted[i], real[i])
    print("The end")
