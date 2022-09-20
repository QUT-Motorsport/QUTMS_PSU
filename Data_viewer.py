# %%
import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import tflite_runtime.interpreter as tflite

# from scipy import integrate # integration with trapizoid

mpl.rcParams['figure.figsize'] = (32, 16)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'Bender'

my_cmap = cm.get_cmap('jet_r')

# output_loc  : str = '/mnt/LibrarySM/SHARED/Data/BMS_data/June11/'
output_loc  : str = '/home/user/tmp/November16/'
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
                pd.read_csv(filepath_or_buffer=f'{output_loc}VoltageInfo/CANid_{i}.csv',
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
                pd.read_csv(filepath_or_buffer=f'{output_loc}TemperatureInfo/CANid_{i}.csv',
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
# Determine initial voltages before the discharge
BMSsFUDS_prevV = []
BMSsFUDS_prevT = []
for i in IDs:
    BMSsFUDS_prevV.append(
            BMSsVoltages[i][ BMSsVoltages[i]['Date_Time'] <= BMSCurrent['Date_Time'].iloc[500] ].iloc[-2500:,:].copy()
        )

    BMSsFUDS_prevT.append(
            BMSsTemperatures[i][ BMSsTemperatures[i]['Date_Time'] <= BMSCurrent['Date_Time'].iloc[500] ].iloc[-2000:,:].copy()
        )

# %%
# Running Machine learning
import tensorflow as tf
import tensorflow_addons as tfa
import tflite_runtime.interpreter as tflite

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
except:
    print('One of the models failed to load.')

try:
    model_file  : str ='Models/Model-№1-FUDS-48.tflite'
    model_file, *device = model_file.split('@')
    interpreter = tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate('libedgetpu.so.1', {'device': device[0]} if device else {})
            ]
        )
    interpreter.allocate_tensors()
except:
    print('Failed to load Lite model. Verefy connection between TPU and PC')

MEAN = np.array([-0.35640615,  3.2060466 , 30.660755  ], dtype=np.float32)
STD  = np.array([ 0.9579658 ,  0.22374259, 13.653275  ], dtype=np.float32)

MEAN2= np.array([-0.35726398,  3.198611  , 26.348543 , 0 ], dtype=np.float32)
STD2 = np.array([ 1.0102334 ,  0.2357838 ,  4.8134885, 1 ], dtype=np.float32)
# %%
def SoC(V : pd.DataFrame, I : pd.DataFrame, T : pd.DataFrame,
        gSoC : np.ndarray
        ) -> tuple[np.float32, np.float32]:
    """ Determine state of charge based on 3-4 feature model
    """
    test_data : np.ndarray = np.zeros(shape=(1, 500, 4), dtype=np.float32)
    test_data[:, :, 0] = I
    test_data[:, :, 1] = V
    test_data[:, :, 2] = T
    test_data[:, :, 3] = gSoC

    # Standad LSTM
    VIT_charge = VIT.predict(x=np.divide(
                                    np.subtract(
                                            np.copy(a=test_data[:, :,:3]),
                                            MEAN
                                        ),
                                    STD
                                ), batch_size=1)[0][0]
    
    # # Custom LSTM
    # VITpSoC_charge = VITpSOC.predict(x=np.divide(
    #                                 np.subtract(
    #                                         np.copy(a=test_data[:, :,:3]),
    #                                         MEAN2
    #                                     ),
    #                                 STD2
    #                             ), batch_size=1)[0]
    # TFLite
    interpreter.set_tensor(
            tensor_index=interpreter.get_input_details()[0]['index'],
            value=np.divide(
                        np.subtract(
                                np.copy(a=test_data[:, :,:3]),
                                MEAN
                            ),
                        STD
                    )
        )
    interpreter.invoke()
    VITLite_charge = interpreter.get_tensor(
            interpreter.get_output_details()[0]['index']
        )
    
    return VIT_charge, 0.0, VITLite_charge
    

#! Cycking time based on amount of current samples
period = BMSCurrent.shape[0]
BMS_Volts = np.zeros(shape=(1,10), dtype=np.float32)
for i in range(0, int(np.round(BMSsFUDS_V[2]['Cycle_Time(s)']).iloc[-1])):
    BMS_Volts = np.append(BMS_Volts,
            np.expand_dims(BMSsFUDS_V[2][(np.round(BMSsFUDS_V[2]['Cycle_Time(s)']) == i)].iloc[0, 2:12].to_numpy(), axis=0),
            axis=0

        )
BMS_Volts = BMS_Volts[1:period+1,:]

BMS_Temps = np.zeros(shape=(1,14), dtype=np.float32)
for i in range(0, int(np.round(BMSsFUDS_T[2]['Cycle_Time(s)']).iloc[-1])):
    try:
        BMS_Temps = np.append(BMS_Temps,
                np.expand_dims(BMSsFUDS_T[2][(np.round(BMSsFUDS_T[2]['Cycle_Time(s)']) == i)].iloc[0, 2:16].to_numpy(), axis=0),
                axis=0

            )
    except:
        print(f'Time {i} missing')
        BMS_Temps = np.append(BMS_Temps,
                BMS_Temps[-1:,:],
                axis=0
            )
BMS_Temps = BMS_Temps[1:period+1,:]
BMS_Current = np.append(np.zeros(shape=(500,), dtype=np.float32),
                        (BMSCurrent['Current(A)']/18).to_numpy())
print(f'Data Shapes: Cr-{BMSCurrent.shape}, Vt-{BMS_Volts.shape}, Tp-{BMS_Temps.shape}')
#! 3 Hours of data to process on sincgle cell. That will take a while
# %%
# Determine the initial set of charges
#* Current is zero
#* BMSsFUDS_prevV
#* BMSsFUDS_prevT
test_data : np.ndarray = np.zeros(shape=(1, 500, 3), dtype=np.float32)
chargeP = np.zeros(shape=(len(IDs), period+500, 10), dtype=np.float32)
chargeC = np.zeros(shape=(len(IDs), period, 10), dtype=np.float32)
j_list = 0
for bms in IDs:
    print(f'BMS - {bms}', flush=True)
    for cell in range(0, 10):
        print(f'Cell - {cell}', flush=True)
        for i in trange(0,500):
            test_data[:, :, 1] = BMSsFUDS_prevV[j_list].iloc[i:2000+i:4, 2+cell].to_numpy()
            test_data[:, :, 2] = BMSsFUDS_prevT[j_list].iloc[i:1500+i:3, 2+cell].to_numpy()
            chargeP[j_list, i, cell] = np.round(VIT.predict(x=np.divide(
                                    np.subtract(
                                            np.copy(a=test_data[:, :, :]),
                                            MEAN
                                        ),
                                    STD
                                ), batch_size=1)[0][0], decimals=2)
    j_list +=1
BMS_Volts = np.append(BMSsFUDS_prevV[1].iloc[-2000::4, 2:12].to_numpy(),
                        BMS_Volts, axis=0
                    )
BMS_Temps = np.append(BMSsFUDS_prevT[1].iloc[-1500::3, 2:16].to_numpy(),
                        BMS_Temps, axis=0
                    )
# %%
# charge = np.zeros(shape=(period-500, 10), dtype=np.float32)
# chargeT = np.zeros(shape=(period-500, 10), dtype=np.float32)
test_data : np.ndarray = np.zeros(shape=(1, 500, 4), dtype=np.float32)
samples = BMS_Current[BMS_Volts[::,cell] >= 2.0].shape[0]
for cell in range(1, 10):
    for i in trange(0,samples-500):
        # charge[i,cell], chargeP[501+i,cell], chargeT[i,cell], = SoC(
        #         BMS_Volts[i:500+i,cell],
        #         BMS_Current[i:500+i],
        #         BMS_Temps[i:500+i,cell],
        #         chargeP[i:500+i,cell]
        #     )
        test_data[:, :, 0] = BMS_Current[BMS_Volts[::,cell] >= 2.0][i:500+i]
        test_data[:, :, 1] = BMS_Volts[BMS_Volts[::,cell] >= 2.0][i:500+i:,cell]
        test_data[:, :, 2] = BMS_Temps[BMS_Volts[::,cell] >= 2.0][i:500+i:,cell]
        test_data[:, :, 3] = chargeP[0, i:500+i,cell]
        chargeP[0, 500+i, cell] = np.round(VITpSOC.predict(x=np.divide(
                                np.subtract(
                                        np.copy(a=test_data[:, :, :]),
                                        MEAN2
                                    ),
                                STD2
                            ), batch_size=1)[0], decimals=2)
        chargeC[0, i, cell] = np.round(VIT.predict(x=np.divide(
                                np.subtract(
                                        np.copy(a=test_data[:, :, :3]),
                                        MEAN
                                    ),
                                STD
                            ), batch_size=1)[0][0], decimals=2)
# plt.plot(chargeP[:,5])
# %%
# s0 = 100;               % Initial SOC estimate
# Q = 45;                 % battery Capacity (Ah)
# eta = 1/(3600*Q);       % Coulombic Efficiency
# i = batteryData.i;      % Load current Data
# v = batteryData.vt;     % Load simulated battery voltage
# dataSize = length(i);   % dataSize will be sued a lot

# SOCcc = zeros(1,dataSize); % Storage vector for SOC estimates 
# t = (1:dataSize);        % Make time vector

# for n = 1:dataSize-1
#     SOCcc(n) = s0 -(eta *trapz(t(n:n+1),i(n:n+1)))*100;
#     s0 = SOCcc(n);
# end  
df_BMSCurrent = pd.read_csv('FUDS_Current.csv')
SoCnorm = pd.read_csv('Data/BMS_data/June11/BMS1_chargeC.csv').iloc[:,1:]
SoCplus = pd.read_csv('Data/BMS_data/June11/BMS1_chargeP.csv').iloc[:,1:]
SoCcc = np.zeros(shape=(chargeP[0,499,:].shape[0], df_BMSCurrent.shape[0]))
SoCcc[:,0] = chargeP[0,499,:]#+0.25

fig, axs = plt.subplots(2,1)
axs[0].plot(chargeP[0,:499,:])
axs[0].set_title('Initial charge over BMS 1', fontsize=32)
axs[0].tick_params(axis='both', labelsize=24)
axs[0].set_ylabel("Charge(%)", fontsize=32)
axs[0].set_xlabel("Time(s)", fontsize=32)
axs[0].set_ylim(0,1.0)
axs[1].set_title('Steady State Voltages over BMS 1', fontsize=32)
axs[1].plot(BMSsFUDS_prevV[0].iloc[i:2000+i:4, 2:12].to_numpy())
axs[1].tick_params(axis='both', labelsize=24)
axs[1].set_ylabel("Voltage(V)", fontsize=32)
axs[1].set_xlabel("Time(s)", fontsize=32)
axs[1].set_ylim(3.1,3.4)

def ccSoC(current   : pd.Series,
          time_s    : pd.Series,
          n_capacity: float = 2.5 ) -> pd.Series:
    """ Return SoC based on Couloumb Counter.
    @ 25deg I said it was 2.5

    Args:
        chargeData (pd.Series): Charge Data Series
        discargeData (pd.Series): Discharge Data Series

    Raises:
        ValueError: If any of data has negative
        ValueError: If the data trend is negative. (end-beg)<0.

    Returns:
        pd.Series: Ceil data with 2 decimal places only.
    """
    return (1/(3600*n_capacity))*(
            integrate.cumtrapz(
                    y=current, x=time_s,
                    dx=1, axis=-1, initial=0
                )
        )
for bms in range(0,10):
    for n in range(1, df_BMSCurrent.shape[0]):
        SoCcc[bms, n] = SoCcc[bms, n-1] + ccSoC(df_BMSCurrent['Current(A)'].iloc[n-1:n+1].to_numpy()/18,
                            df_BMSCurrent['Cycle_Time(s)'].iloc[n-1:n+1].to_numpy())[1]
num = 10
fig, axs = plt.subplots(num, 1, figsize=(24,96))
for i in range(num):
    axs[i].plot(np.arange(SoCcc.shape[1]), SoCcc[i,:], label='CC')
    axs[i].plot(np.arange(SoCcc.shape[1]), SoCnorm.iloc[:,i], label='Published')
    axs[i].plot(np.arange(SoCcc.shape[1]), SoCplus.iloc[500:,i], label='QUTMS')
    axs[i].set_title(f'Coulub Counting over Cell {i}', fontsize=32)
    axs[i].tick_params(axis='both', labelsize=24)
    axs[i].set_ylabel("Charge(%)", fontsize=32)
    axs[i].set_xlabel("Time(s)", fontsize=32)
    axs[i].legend()
# fig.savefig('ChargeTest.png', facecolor='white')
# plt.plot(ccSoC(df_BMSCurrent['Current(A)']/18, df_BMSCurrent['Cycle_Time(s)']))
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

# %%
# Timimg
#? CPU - 0.076 - 4x faster
#? GPU - 0.057 - 6x faster
#? TPU - 0.305
# import time
# tic = time.perf_counter()
# test_data : np.ndarray = np.zeros(shape=(1, 500, 3), dtype=np.float32)
# test_data[:, :, 0] = BMS_Current[i:500+i]
# test_data[:, :, 1] = BMS_Volts[i:500+i,cell]
# test_data[:, :, 2] = BMS_Temps[i:500+i,cell]
# VIT.predict(x=np.divide(
#                 np.subtract(
#                         np.copy(a=test_data[:, :,:3]),
#                         MEAN
#                     ),
#                 STD
#             ), batch_size=1)
# print(f'GPU model: {time.perf_counter()-tic}')

# tic = time.perf_counter()
# test_data : np.ndarray = np.zeros(shape=(1, 500, 3), dtype=np.float32)
# test_data[:, :, 0] = BMS_Current[i:500+i]
# test_data[:, :, 1] = BMS_Volts[i:500+i,cell]
# test_data[:, :, 2] = BMS_Temps[i:500+i,cell]
# interpreter.set_tensor(
#         tensor_index=interpreter.get_input_details()[0]['index'],
#         value=np.divide(
#                     np.subtract(
#                             np.copy(a=test_data[:, :,:3]),
#                             MEAN
#                         ),
#                     STD
#                 )
#     )
# interpreter.invoke()
# interpreter.get_tensor(
#         interpreter.get_output_details()[0]['index']
#     )
# print(f'TPU model: {time.perf_counter()-tic}')