from utils import *

TEMP_COMFORT = 20
CAR_BATTERY_MAX_CAP = 30
CHARGING_SPEED = 8

def state(current_hour, next_hour, car_connected, car_energy, battery_energy, current_temperature, b, c, air):
    # print(f'\r State: {car_connected},{car_energy},{battery_energy},{current_temperature},{b},{c},{air}', end='')
    '''Temperature A/C'''
    ptt = nextTemp(current_hour['temperature'], current_temperature, air) # power to temp
    air_use, next_temp = ptt # power of A/C

    '''Non controllable energy (fixed)'''
    nc_use = next_hour['total']

    '''Car charging cost'''
    # Car energy is discretized (one unit means 3)
    car_energy = car_energy * 3
    if car_connected and c == 0:
        car_use = min([CHARGING_SPEED, (CAR_BATTERY_MAX_CAP - car_energy)])
    else:
        car_use = 0
    total_use = air_use + nc_use + car_use
    energy_from_battery = b * min([total_use, battery_energy + current_hour['solar']])
    energy_from_car = car_connected * (c * max( [min([total_use-energy_from_battery, car_energy]), 0] ))


    '''Temperature Discomfort cost'''
    temp_cost = abs(next_temp - TEMP_COMFORT) * 105/4


    total_cost = temp_cost + price(total_use - energy_from_battery - energy_from_car, hour=None)

    x = [
        current_hour['total'], current_hour['temperature'], 
        current_hour['humidity'], current_hour['solar'],
        int(car_connected), car_energy, battery_energy, current_temperature,
        b, c, air
    ]

    y = total_cost
    return x, y