# %%
import os

# Data management
import pandas as pd
import numpy as np
from datetime import datetime

# Plotting and colormapping
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Machine learning processing and CC
import tensorflow as tf
import tensorflow_addons as tfa
import tflite_runtime.interpreter as tflite
from scipy import integrate # integration with trapizoid
from AutoFeedBack import AutoFeedBack

# Fancy bar for loops
from tqdm import trange

# mpl.rcParams['figure.figsize'] = (32, 16)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'Bender'

my_cmap = cm.get_cmap('jet_r')

output_loc  : str = 'Data/BMS_data/July09-FUDS1/'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
GPU=0
if physical_devices:
    #! With /device/GPU:1 the output was faster.
    #! need to research more why.
    tf.config.experimental.set_visible_devices(
                            physical_devices[GPU], 'GPU')

    #if GPU == 1:
    tf.config.experimental.set_memory_growth(
                            physical_devices[GPU], True)
    print("GPU found and memory growth enabled") 
    
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    print("GPU found") 
    print(f"\nPhysical GPUs: {len(physical_devices)}"
                  f"\nLogical GPUs: {len(logical_devices)}")
#! For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float32')

# %%
BMSsData = []
for i in range(0, 6):
    try:
        #* Voltages
        BMSsData.append(
                pd.read_csv(filepath_or_buffer=f'{output_loc}Filt_CANid_{i}.csv',
                            sep=',', verbose=True)
            )
        print(f'Cycle samples of V at BMS{i} - {BMSsData[i].shape} and T -{BMSsData[i].shape}')
    except Exception as e:
        print(e)
        print(f'============Failed to extract Cycle data of BMS {i}==============')
# %%
try:
    VIT : tf.keras.models.Sequential = tf.keras.models.load_model(
            filepath='Models/VIT', compile=False,
            custom_objects={"RSquare": tfa.metrics.RSquare}
        )
    print('VIT model loaded')
    VITpSOC : AutoFeedBack = AutoFeedBack(units=510,
            out_steps=30, num_features=1
        )
    VITpSOC.load_weights('Models/VITpSoC/12')
    print('VITpSoC model loaded')
except:
    print('One of the models failed to load.')
MEAN = np.array([-0.35640615,  3.2060466 , 30.660755  ], dtype=np.float32)
STD  = np.array([ 0.9579658 ,  0.22374259, 13.653275  ], dtype=np.float32)

MEAN2= np.array([-0.35726398,  3.198611  , 26.348543 , 0 ], dtype=np.float32)
STD2 = np.array([ 1.0102334 ,  0.2357838 ,  4.8134885, 1 ], dtype=np.float32)    
# %%
test_data : np.ndarray = np.zeros(shape=(1, 500, 3), dtype=np.float32)
BMSsCharges = []
for BMSid in range(0, 1):
    print(f'BMS - {BMSid}', flush=True)
    BMSsCharges.append(BMSsData[BMSid].loc[:,['Date_Time(ms)', 'Cycle_Time(s)']])
    sample_len = BMSsData[BMSid].shape[0]
    for cell in range(1, 11):
        chargeP = np.zeros(shape=(sample_len, ), dtype=np.float32)
        #* Repmated prediction
        for i in trange(0,500):   #* Reverse order
            current = np.resize(a=BMSsData[BMSid].loc[:i, 'Current(A)'].to_numpy(), new_shape=(500,))
            test_data[:, :, 0] = np.concatenate([current[i:], current[:i]], axis=0)
            voltage = np.resize(a=BMSsData[BMSid].loc[:i, f'6-Cell_{cell}'].to_numpy(), new_shape=(500,))
            test_data[:, :, 1] = np.concatenate([voltage[i:], voltage[:i]], axis=0)
            temperature = np.resize(a=BMSsData[BMSid].loc[:i, f'Sns_{cell}'].to_numpy(), new_shape=(500,))
            test_data[:, :, 2] = np.concatenate([temperature[i:], temperature[:i]], axis=0)
            chargeP[i] = np.round(VIT.predict(x=np.divide(
                                    np.subtract(
                                            np.copy(a=test_data[:, :, :]),
                                            MEAN
                                        ),
                                    STD
                                ), batch_size=1)[0][0], decimals=2)
        #* Normal prediction
        for i in trange(0,sample_len-500):
            test_data[:, :, 0] = BMSsData[BMSid].loc[i:499+i, 'Current(A)'].to_numpy()
            test_data[:, :, 1] = BMSsData[BMSid].loc[i:499+i, f'6-Cell_{cell}'].to_numpy()
            test_data[:, :, 2] = BMSsData[BMSid].loc[i:499+i, f'Sns_{cell}'].to_numpy()
            chargeP[500+i] = np.round(VIT.predict(x=np.divide(
                                    np.subtract(
                                            np.copy(a=test_data[:, :, :]),
                                            MEAN
                                        ),
                                    STD
                                ), batch_size=1)[0][0], decimals=2)
        BMSsCharges[BMSid][f'VIT_{cell}(%)'] = chargeP.copy()
# BMSsCharges.iloc[:,2:12].plot(subplots=True)
if not os.path.exists(output_loc+'Chemali2017'):
    print('First time, making dirs')
    os.makedirs(output_loc+'Chemali2017')
for BMSid in range(0, 1):
    BMSsCharges[BMSid].to_csv(output_loc+f'Chemali2017/VIT_CANid_{BMSid}')

test_data : np.ndarray = np.zeros(shape=(1, 500, 4), dtype=np.float32)
BMSsCharges = []
for BMSid in range(0, 1):
    print(f'BMS - {BMSid}', flush=True)
    BMSsCharges.append(BMSsData[BMSid].loc[:,['Date_Time(ms)', 'Cycle_Time(s)']])
    sample_len = BMSsData[BMSid].shape[0]
    for cell in range(1, 11):
        chargeP = np.zeros(shape=(sample_len, ), dtype=np.float32)
        #* Repmated prediction
        for i in trange(0,500):   #* Reverse order
            current = np.resize(a=BMSsData[BMSid].loc[:i, 'Current(A)'].to_numpy(), new_shape=(500,))
            test_data[:, :, 0] = np.concatenate([current[i:], current[:i]], axis=0)
            voltage = np.resize(a=BMSsData[BMSid].loc[:i, f'6-Cell_{cell}'].to_numpy(), new_shape=(500,))
            test_data[:, :, 1] = np.concatenate([voltage[i:], voltage[:i]], axis=0)
            temperature = np.resize(a=BMSsData[BMSid].loc[:i, f'Sns_{cell}'].to_numpy(), new_shape=(500,))
            test_data[:, :, 2] = np.concatenate([temperature[i:], temperature[:i]], axis=0)
            chargeP[i] = np.round(VIT.predict(x=np.divide(
                                    np.subtract(
                                            np.copy(a=test_data[:, :, :3]),
                                            MEAN
                                        ),
                                    STD
                                ), batch_size=1)[0][0], decimals=2)
        #* Normal prediction
        # chargeP[:500] = 0.98
        for i in trange(0,sample_len-500):
            test_data[:, :, 0] = BMSsData[BMSid].loc[i:499+i, 'Current(A)'].to_numpy()
            test_data[:, :, 1] = BMSsData[BMSid].loc[i:499+i, f'6-Cell_{cell}'].to_numpy()
            test_data[:, :, 2] = BMSsData[BMSid].loc[i:499+i, f'Sns_{cell}'].to_numpy()
            test_data[:, :, 3] = chargeP[i:500+i]
            chargeP[500+i] = np.round(VITpSOC.predict(x=np.divide(
                                    np.subtract(
                                            np.copy(a=test_data[:, :, :]),
                                            MEAN2
                                        ),
                                    STD2
                                ), batch_size=1)[0], decimals=2)
        BMSsCharges[BMSid][f'VIT_{cell}(%)'] = chargeP.copy()

if not os.path.exists(output_loc+'Sadykov2021'):
    print('First time, making dirs')
    os.makedirs(output_loc+'Sadykov2021')
for BMSid in range(0, 1):
    BMSsCharges[BMSid].to_csv(output_loc+f'Sadykov2021/VITpSoC_CANid_{BMSid}')