from dask.distributed import Client, progress
import numpy as np
import dask.array as da
import pandas as pd
from dask_ml.wrappers import ParallelPostFit
from sklearn import neural_network
# from train import *
import numpy as np
# Scale up: connect to your own cluster with bmore resources
# see http://dask.pydata.org/en/latest/setup.html
client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='2GB')

print(client)
dtype = {
        'total': np.float64,
        'temperature': np.int32,
        'humidity': np.float64,
        'solar': np.float64,
        'car_connected': np.int32,
        'car_energy': np.int32,
        'battery_energy': np.int32,
        'current_temperature': np.int32,
        'b': np.int32,
        'c': np.int32,
        'air': np.int32,
        'cost': np.int32
    }
x = pd.read_csv('train_data.csv', dtype=dtype)

y = x.pop('cost').values

mlp = ParallelPostFit(neural_network.MLPRegressor(hidden_layer_sizes=(16,), solver='adam'), scoring="r2")

print('Training')
mlp.fit(x, y)
print('Finished')

