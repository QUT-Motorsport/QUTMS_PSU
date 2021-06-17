# %%
import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import matplotlib.cm as cm

mpl.rcParams['figure.figsize'] = (32, 16)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'Bender'

my_cmap = cm.get_cmap('jet_r')

# output_loc  : str = '/mnt/LibrarySM/SHARED/Data/BMS_data/June11/'
output_loc  : str = '/home/xana/tmp/June11/'
# %%
# Current data and plot
BMSCurrent  : pd.DataFrame = pd.read_csv(
    filepath_or_buffer=f'{output_loc}Current.csv', sep=",", delimiter=None,
    # Column and Index Locations and Names
    header="infer", names=None, index_col=None, usecols=None, squeeze=False,
    prefix=None, mangle_dupe_cols=True,
    # General Parsing Configuration
    dtype=None, engine=None, converters=None, true_values=None,
    false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0,
    nrows=None,
    # NA and Missing Data Handling
    na_values=None, keep_default_na=True, na_filter=True, verbose=True,
    skip_blank_lines=True,
    # Datetime Handling
    parse_dates=False, infer_datetime_format=False, keep_date_col=False,
    date_parser=None, dayfirst=False, cache_dates=True,
    # Iteration
    iterator=False, chunksize=None,
    # Quoting, Compression, and File Format
    compression="infer", thousands=None, decimal = ".", lineterminator=None,
    quotechar='"', doublequote=True, escapechar=None, comment=None,
    encoding=None, dialect=None,
    # Error Handling
    error_bad_lines=True, warn_bad_lines=True,
    # Internal
    delim_whitespace=False, memory_map=False, float_precision=None)
print(BMSCurrent.head())
# Use Date converted to ints up to seconds ONLY
BMSCurrent['TSeconds']=(pd.to_datetime(BMSCurrent['Date_Time'].iloc[:].str.slice(stop=-4),
               format='%d/%m/%Y %H:%M:%S')- pd.to_datetime('1970-01-01')).dt.total_seconds()

start_cycling = pd.to_datetime(BMSCurrent['Date_Time'].iloc[0][:-4])
BMSCurrent['Cycle_Time(s)']=(
        pd.to_datetime(BMSCurrent['Date_Time'].iloc[:].str.slice(stop=-4)) \
        - start_cycling
    ).dt.total_seconds().astype(int)
time_minutes = np.linspace(0,int(BMSCurrent['Cycle_Time(s)'].iloc[-1]/60)+1,BMSCurrent['Cycle_Time(s)'].shape[0])
BMSCurrent['Cycle_Time(m)'] = time_minutes
# BMSCurrent['TSeconds'][(BMSCurrent['TSeconds'][:] == int(BMSCurrent['TSeconds'][5]-1))]

# Plot current applied
fig, ax = plt.subplots()
ax.plot(BMSCurrent['Cycle_Time(s)'], BMSCurrent['Current(A)'], label='Current Profile')
ax.set_title('FUDS Cycling profile', fontsize=36)
ax.tick_params(axis='both', labelsize=24)
ax.set_ylabel("Discharge/Charge", fontsize=32)
ax.set_xlabel('Time(s)', fontsize=32)
fig.tight_layout()
fig.show()
# %%
# Extracting Voltages
BMS_id  : int = 0   #! ID 3 is worthless to look at

BMSsVoltages = []
BMSsFUDS_V = []

BMSsTemperatures = []
BMSsFUDS_T = []
for i in range(0, 6):
    try:
        #* Voltages
        BMSsVoltages.append(
                pd.read_csv(filepath_or_buffer=f'{output_loc}Voltages/CANid_{i}.csv',
                            sep=',', verbose=True)
            )
        BMSsFUDS_V.append(
                BMSsVoltages[i][ BMSsVoltages[i]['Date_Time'] >= BMSCurrent['Date_Time'].iloc[0] ].copy()
            )
        print(f'BMS {i} Voltage Shape - {BMSsVoltages[i].shape}')
        BMSsVoltages[i]['TSeconds'] = (pd.to_datetime(BMSsVoltages[i]['Date_Time'].iloc[:].str.slice(stop=-4),
                format='%d/%m/%Y %H:%M:%S')- pd.to_datetime('1970-01-01')).dt.total_seconds()
        # BMSsFUDS_V[-1]['Cycle_Time(s)'] = BMSsFUDS_V[-1]['Step_Time(s)']-BMSsFUDS_V[-1]['Step_Time(s)'].iloc[0]
        
        
        #* Temperature
        BMSsTemperatures.append(
                pd.read_csv(filepath_or_buffer=f'{output_loc}Temperatures/CANid_{i}.csv',
                            sep=',', verbose=True)
            )
        BMSsFUDS_T.append(
                BMSsTemperatures[i][ BMSsTemperatures[i]['Date_Time'] >= BMSCurrent['Date_Time'].iloc[0] ].copy()
            )
        print(f'BMS {i} Temperature Shape - {BMSsTemperatures[i].shape}\n')
        BMSsTemperatures[i]['TSeconds'] = (pd.to_datetime(BMSsTemperatures[i]['Date_Time'].iloc[:].str.slice(stop=-4),
                format='%d/%m/%Y %H:%M:%S')- pd.to_datetime('1970-01-01')).dt.total_seconds()
        # BMSsFUDS_T[-1]['Cycle_Time(s)'] = BMSsFUDS_T[-1]['Step_Time(s)']-BMSsFUDS_T[-1]['Step_Time(s)'].iloc[0]
        
    except Exception as e:
        print(e)
        print(f'============Failed to extract Cycle data of BMS {i}==============')
# cycling = BMSsVoltages[i][ BMSsVoltages[i]['Date_Time'] >= BMSCurrent['Date_Time'].iloc[0] ]
# plt.plot(cycling['4Cell_1'])
for i in range(0, 6):
    try:
        BMSsFUDS_V[i]['Cycle_Time(s)'] = BMSsFUDS_V[i]['Step_Time(s)']-BMSsFUDS_V[i]['Step_Time(s)'].iloc[0]
        BMSsFUDS_T[i]['Cycle_Time(s)'] = BMSsFUDS_T[i]['Step_Time(s)']-BMSsFUDS_T[i]['Step_Time(s)'].iloc[0]
        print(f'Cycle samples of V at BMS{i} - {BMSsFUDS_V[i].shape} and T -{BMSsFUDS_T[i].shape}')
    except Exception as e:
        print(f'{e} + BMS {i}')
# %%
#! Working only with BMS IDs:
IDs = [1, 2]
fig, axs = plt.subplots(3,2)
fig.suptitle('Data for model over BMSid 1 and 2', fontsize=36)
axs[0][0].plot(BMSsFUDS_V[0]['Cycle_Time(s)'], BMSsFUDS_V[0].iloc[:,2:12])
axs[0][0].set_title('CANid_1', fontsize=32)
axs[0][0].tick_params(axis='both', labelsize=24)
axs[0][0].set_ylabel("Voltage(V)", fontsize=32)

axs[0][1].plot(BMSsFUDS_V[1]['Cycle_Time(s)'], BMSsFUDS_V[1].iloc[:,2:12])
axs[0][1].set_title('CANid_2', fontsize=32)
axs[0][1].tick_params(axis='both', labelsize=24)
axs[0][1].set_ylabel("Voltage(V)", fontsize=32)
axs[0][1].set_ylim([3.0, 3.5])

axs[1][0].plot(BMSsFUDS_T[0]['Cycle_Time(s)'], BMSsFUDS_T[0].iloc[:,2:16])
axs[1][0].tick_params(axis='both', labelsize=24)
axs[1][0].set_ylabel("Temperature(T)", fontsize=32)

axs[1][1].plot(BMSsFUDS_T[0]['Cycle_Time(s)'], BMSsFUDS_T[0].iloc[:,2:16])
axs[1][1].tick_params(axis='both', labelsize=24)
axs[1][1].set_ylabel("Temperature(T)", fontsize=32)

axs[2][0].plot(BMSCurrent['Cycle_Time(s)'], BMSCurrent['Current(A)']/18)
axs[2][0].tick_params(axis='both', labelsize=24)
axs[2][0].set_ylabel("Current(A)", fontsize=32)
axs[2][0].set_xlabel('Time(s)', fontsize=32)

axs[2][1].plot(BMSCurrent['Cycle_Time(s)'], BMSCurrent['Current(A)']/18)
axs[2][1].tick_params(axis='both', labelsize=24)
axs[2][1].set_ylabel("Current(A)", fontsize=32)
axs[2][1].set_xlabel('Time(s)', fontsize=32)
fig.tight_layout()
fig.show()

#! This proves that we need an Embeddewd solution to the BMS. As per NVIDIAs goals
#!of integration of ML into any structure, we were not ahead of them, but following along
#!The Coral or TPU solution has to be the next goal for the QUEV-4 and if milestone
#!of our successes is going to be submitted - this is the prove that we need to
#!continue to push research to stay close to the market grow.

# %%
# Running Machine learning
import tensorflow as tf
import tensorflow_addons as tfa
# import tflite_runtime.interpreter as tflite

from AutoFeedBack import AutoFeedBack
from tqdm import trange
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
    # model_file  : str ='Models/Model-№1-FUDS-48.tflite'
    # model_file, *device = model_file.split('@')
    # interpreter = tflite.Interpreter(
    #         model_path=model_file,
    #         experimental_delegates=[
    #             tflite.load_delegate('libedgetpu.so.1', {'device': device[0]} if device else {})
    #         ]
    #     )
    # interpreter.allocate_tensors()
except:
    print('One of the models failed to load.')

MEAN = np.array([-0.35640615,  3.2060466 , 30.660755  ], dtype=np.float32)
STD  = np.array([ 0.9579658 ,  0.22374259, 13.653275  ], dtype=np.float32)

def SoC(V : pd.DataFrame, I : pd.DataFrame, T : pd.DataFrame,
        gSoC : np.ndarray
        ) -> tuple[np.float32, np.float32]:
    """ Determine state of charge based on 3-4 feature model
    """
    test_data : np.ndarray = np.zeros(shape=(1, 500, 4), dtype=np.float32)
    test_data[:, :, 0] = I
    test_data[:, :, 1] = V
    test_data[:, :, 2] = T
    test_data[:, :,:3]= np.divide(
                    np.subtract(
                            np.copy(a=test_data[:, :,:3]),
                            MEAN
                        ),
                    STD
                )
    # Standad LSTM
    VIT_charge = VIT.predict(test_data[:,:,:3], batch_size=1)[0][0]
    
    test_data[:, :, 3] = gSoC
    # # Custom LSTM
    VITpSoC_charge = VITpSOC.predict(test_data[:,:,:], batch_size=1)[0]
    # if(interpreter):
    #     # TFLite
    #     interpreter.set_tensor(
    #             interpreter.get_input_details()['index'],
    #             np.expand_dims(test_data[:,:3], axis=0)
    #         )
    #     interpreter.invoke()
    #     VITLite_charge = interpreter.get_tensor(
    #             interpreter.get_output_details()[0]['index']
    #         )
    #     return VIT_charge, VITpSoC_charge, VITLite_charge
    # else:
    return VIT_charge, VITpSoC_charge, 0.0

#! Cycking time based on amount of current samples
period = BMSCurrent.shape[0]
BMS_Volts = np.zeros(shape=(1,10), dtype=np.float32)
for i in range(0, int(np.round(BMSsFUDS_V[1]['Cycle_Time(s)']).iloc[-1])):
    BMS_Volts = np.append(BMS_Volts,
            np.expand_dims(BMSsFUDS_V[1][(np.round(BMSsFUDS_V[1]['Cycle_Time(s)']) == i)].iloc[0, 2:12].to_numpy(), axis=0),
            axis=0

        )
BMS_Volts = BMS_Volts[1:period+1,:]

BMS_Temps = np.zeros(shape=(1,14), dtype=np.float32)
for i in range(0, int(np.round(BMSsFUDS_T[1]['Cycle_Time(s)']).iloc[-1])):
    try:
        BMS_Temps = np.append(BMS_Temps,
                np.expand_dims(BMSsFUDS_T[1][(np.round(BMSsFUDS_T[1]['Cycle_Time(s)']) == i)].iloc[0, 2:16].to_numpy(), axis=0),
                axis=0

            )
    except:
        print(f'Time {i} missing')
        BMS_Temps = np.append(BMS_Temps,
                BMS_Temps[-1:,:],
                axis=0
            )
BMS_Temps = BMS_Temps[1:period+1,:]
BMS_Current = (BMSCurrent['Current(A)']/18).to_numpy()
print(f'Data Shapes: Cr-{BMSCurrent.shape}, Vt-{BMS_Volts.shape}, Tp-{BMS_Temps.shape}')
#! 3 Hours of data to process on sincgle cell. That will take a while
# %%
charge = np.zeros(shape=(period-500, 10), dtype=np.float32)
chargeP = np.zeros(shape=(period, 10), dtype=np.float32)
chargeP[:500,:] = 0.8
for cell in range(0, 1):
    for i in trange(0,period-500):
        charge[i,cell], chargeP[501+i,cell], _ = SoC(
                BMS_Volts[i:500+i,cell],
                BMS_Current[i:500+i],
                BMS_Temps[i:500+i,cell],
                chargeP[i:500+i,cell]
            )

plt.plot(chargeP[:,5])
# %%
def smooth(y, box_pts: int) -> np.array:
    """ Smoothing data using numpy convolve. Based on the size of the
    averaging box, data gets smoothed.
    Here it used in following form:
    y = V/(maxV-minV)
    box_pts = 500

    Args:
        y (pd.Series): A data which requires to be soothed.
        box_pts (int): Number of points to move averaging box

    Returns:
        np.array: Smoothed data array
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
# 0 and 5
fig, axs = plt.subplots()
axs.plot(range(501,len(charge[:,0])+501), charge[:,0]*100)
axs.set_title('CANid_1 Cell0-VIT model', fontsize=32)
axs.tick_params(axis='both', labelsize=24)
axs.set_ylabel("Charge(%)", fontsize=32)
axs.set_xlabel("Time(s)", fontsize=32)
axs.set_ylim(-5,105)
fig.show()
fig.savefig(output_loc+'CANid_1 Cell0-VIT.png', facecolor='white', transparent=False)
# fig, axs = plt.subplots()
# axs.plot(range(501,len(chargeP[:,0])+501), chargeP[:,0]*100)
# axs.set_title('CANid_1 Cell0 - VITpSoC model', fontsize=32)
# axs.tick_params(axis='both', labelsize=24)
# axs.set_ylabel("Charge(%)", fontsize=32)
# axs.set_xlabel("Time(s)", fontsize=32)
# axs.set_ylim(-5,105)
# fig.show()

fig, axs = plt.subplots()
axs.plot(range(501,len(charge[:,5])+501), charge[:,5]*100)
axs.set_title('CANid_1 Cell5- VIT model', fontsize=32)
axs.tick_params(axis='both', labelsize=24)
axs.set_ylabel("Charge(%)", fontsize=32)
axs.set_xlabel("Time(s)", fontsize=32)
axs.set_ylim(-5,105)
fig.show()
fig.savefig(output_loc+'CANid_1 Cell5-VIT.png', facecolor='white', transparent=False)

fig, axs = plt.subplots()
axs.plot(range(501,len(chargeP[:,5])+501), chargeP[:,5]*100)
axs.plot(range(501,len(chargeP[:,5])+501), smooth(chargeP[:,5], 500)*100, linewidth=10)
axs.set_title('CANid_1 Cell5- VITpSoC model', fontsize=32)
axs.tick_params(axis='both', labelsize=24)
axs.set_ylabel("Charge(%)", fontsize=32)
axs.set_xlabel("Time(s)", fontsize=32)
axs.set_ylim(-5,105)
fig.show()
fig.savefig(output_loc+'CANid_1 Cell5-VITpSoC.png', facecolor='white', transparent=False)
