alpha = 1/3
def nextTemp(outside_temp, current_temp, power_opt):
    """
    power: 0, 0.2, 0.6, 0.8
    temps: outside, 27, 23, 19
    """
    powerToTemp = {
        4: {
            "power": 0,
            "temp": None
        },
        3: {
            "power": 0.2,
            "temp": 27
        },
        2: {
            "power": 0.5,
            "temp": 23
        },
        1: {
            "power": 0.8,
            "temp": 19
        },
        0: {
            "power": 1.1,
            "temp": 15
        }
    }

    if power_opt == 4:
        return powerToTemp[power_opt]['power'], current_temp*(1/2) + outside_temp*(1/2)

    return powerToTemp[power_opt]['power'], (current_temp*(alpha) + outside_temp*(alpha) + powerToTemp[power_opt]['temp']*(alpha))

def price(e, hour):
    return 105 * e