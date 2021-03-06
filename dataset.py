import numpy as np
from sklearn import neural_network
import json
import time
import sys
from joblib import dump, load
import datetime
import pandas
from functools import reduce

from .utils import *
from .data import *
from .state import state

def export_csv(data, path, data_percentage=100):
    dataset_size = 24*len(data)
    for t in range(dataset_size-1):
        file = open(path, 'a')
        data_rows=[]
        initial_time = time.time()
        print(f'Etapa: {t}/{dataset_size-1}\n')
        current_hour = data[t // 24][t % 24]
        next_hour = data[(t+1) // 24][(t+1) % 24]

        """Iterate over states (battery status, car...)"""
        for battery_energy in range(0, 5): # battery energy uses only 5 states (2 units each.)
            lower_temp = current_hour['temperature'] - 5
            upper_temp = current_hour['temperature'] + 5
            current_temp = current_hour['temperature']
            for current_temperature in [lower_temp, current_temp, upper_temp]: # Cambiar rango a +-10 de temperatura de afuera
                for car_connected in [False, True]:
                    for car_energy in range(0, car_connected * 30 + 1, 6): # each unit is 3 kwh
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
                                    # Add row x,y
                                    data_rows.append(reduce(lambda v, w: str(v) + ',' + str(w), x) + ',' + str(round(y)) + '\n')
        file.writelines(data_rows)
        file.close()
        end_time = time.time()
        print(f'\nTiempo: {end_time - initial_time}')

if __name__ == "__main__":
    data_2016 = load_data('.', which='2016')
    with open('train_data.csv', 'w') as writer:
        writer.write('total,temperature,humidity,solar,car_connected,car_energy,battery_energy,current_temperature,b,c,air,cost\n')
    export_csv(data_2016)
    print("Saving...")
    print("Finished")
