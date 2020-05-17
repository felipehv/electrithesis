import json
from utils import *

PATH='.'

def celsius(f):
    return round((f-32)*5/9)

with open(f'{PATH}/data_2016_dict.dump', 'r') as reader:
    data_2016 = json.loads(reader.read())
for day in data_2016:
    for data in day:
        data['temperature'] = celsius(data['temperature'])
        data['total'] = round(data['total'], 2)

# with open(f'{PATH}/data_2018_dict.dump', 'r') as reader:
#     data_2018 = json.loads(reader.read())
# for day in data_2018:
#     for data in day:
#         data['temperature'] = celsius(data['temperature'])
#         data['total'] = round(data['total'], 2)