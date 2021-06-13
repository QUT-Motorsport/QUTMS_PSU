#!/usr/bin/python
# %% [markdown]
# # # AMS State of Charge plotting
# Script intended to compate dirrefent State of Charge predictions incuding:
#   *   Coulumb Counting
#   *   SoC with VIT
#   *   SoC with VITpSoc
# %%
import time

# Data management
import pandas as pd
import numpy as np

# Plotting animated 
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# Machine learning processing
import tensorflow as tf
from tensorflow.python.eager.context import device
import tensorflow_addons as tfa
import tflite_runtime.interpreter as tflite

from scipy import integrate # integration with trapizoid

from AutoFeedBack import AutoFeedBack

from datetime import datetime
import getpass

mpl.rcParams['figure.figsize'] = (32, 16)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'Bender'

my_cmap = cm.get_cmap('jet_r')
output_loc = f'/home/{getpass.getuser()}/tmp/{datetime.now().strftime("%B%d")}/'

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

# def get_demo_data():
#     BMSv_DataFrames = []
#     BMSt_DataFrames = []
#     for i in range(0, h_shape):
#         BMSv_DataFrames.append(pd.read_csv(f'demo/voltages/CANid_{i}.csv'))
#         BMSt_DataFrames.append(pd.read_csv(f'demo/temperatures/CANid_{i}.csv'))
#     current = pd.read_csv(f'demo/current.csv')
#     return BMSv_DataFrames, BMSt_DataFrames, current

def SoC(V : pd.DataFrame, I : pd.DataFrame, T : pd.DataFrame,
        gSoC : np.ndarray, cell : int, interpreter=None
        ) -> tuple[np.float32, np.float32]:
    """ Determine state of charge based on 3-4 feature model
    """
    test_data : np.ndarray = np.zeros(shape=(500,4), dtype=np.float32)
    test_data[:, 0] =-I.iloc[::fs,0].to_numpy()
    test_data[:, 1] = V.iloc[::fs,cell].to_numpy()
    test_data[:, 2] = T.iloc[::,cell].to_numpy()
    test_data[:,:3]= np.divide(
                    np.subtract(
                            np.copy(a=test_data[:,:3]),
                            MEAN
                        ),
                    STD
                )
    # Standad LSTM
    VIT_charge = model.predict(np.expand_dims(test_data[:,:3], axis=0),
                            batch_size=1)[0][0]
    
    test_data[:,3] = gSoC[:,cell]
    # Custom LSTM
    VITpSoC_charge = VITpSOC.predict(np.expand_dims(test_data[:,:], axis=0),
                            batch_size=1)[0]
    if(interpreter):
        # TFLite
        interpreter.set_tensor(
                interpreter.get_input_details()['index'],
                np.expand_dims(test_data[:,:3], axis=0)
            )
        interpreter.invoke()
        VITLite_charge = interpreter.get_tensor(
                interpreter.get_output_details()[0]['index']
            )
        return VIT_charge, VITpSoC_charge, VITLite_charge
    else:
        return VIT_charge, VITpSoC_charge, 0.0
def ccSoC(current   : pd.Series,
          time_s    : pd.Series,
          n_capacity: float = 2.5 ) -> pd.Series:
    """ Return SoC based on Couloumb Counter.
    TODO: Ceil the data and reduce to 2 decimal places.

    Args:
        chargeData (pd.Series): Charge Data Series
        discargeData (pd.Series): Discharge Data Series

    Raises:
        ValueError: If any of data has negative
        ValueError: If the data trend is negative. (end-beg)<0.

    Returns:
        pd.Series: Ceil data with 2 decimal places only.
    """
    # Raise error
    # if(any(time_s) < 0):
    #     raise ValueError("Parser: Time cannot be negative.")
    # if(n_capacity == 0):
    #     raise ZeroDivisionError("Nominal capacity cannot be zero.")
    # Integration with trapezoid    
    # Nominal Capacity 2.5Ah. Double for that
    #uni_data_multi["CC"] = uni_data_multi["trapz(I)"]/3600*2.5
    #DoD = DsgCap - ChgCap
    #SOC = 1-Dod/Q
    #@ 25deg I said it was 2.5
    return (integrate.cumtrapz(current, time_s, initial=0)/abs(n_capacity*2))
# %%
if __name__ == '__main__':
    #! Setup all plots for animations
    # BMSv_DataFrames, BMSt_DataFrames, current = get_demo_data()

    labels_v = ['cell-1','cell-2','cell-3','cell-4','cell-5','cell-6','cell-7','cell-8','cell-9','cell-10']
    labels_t = ['CC-1','CC-2','CC-3','CC-4','CC-5','CC-6','CC-7','CC-8','CC-9','CC-10','CC-11','CC-12']
    v_shape : int = 6
    h_shape : int = 6

    fig = plt.figure(num = 1, figsize=(32,16))
    fig.suptitle("AMS here                          CC && SoC w/ VIT  && SoC w/ VITpSoC                   AMS here",
                fontsize=12)
    # SoC w/ VIT bar
    ax0  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(0, 2), colspan=1)
    ax1  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(0, 1), colspan=1)
    ax2  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(0, 0), colspan=1)
    ax3  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(1, 2), colspan=1)
    ax4  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(1, 1), colspan=1)
    ax5  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(1, 0), colspan=1)
    axsSoCb = [ax0, ax1, ax2, ax3, ax4, ax5]
    # SoC w/ VIT plots
    ax6  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(0, 5), colspan=1)
    ax7  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(0, 4), colspan=1)
    ax8  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(0, 3), colspan=1)
    ax9  = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(1, 5), colspan=1)
    ax10 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(1, 4), colspan=1)
    ax11 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(1, 3), colspan=1)
    axsSoCp = [ax6, ax7, ax8, ax9, ax10, ax11]

    # SoC w/ VITpSoC bar
    ax12 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(2, 2), colspan=1)
    ax13 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(2, 1), colspan=1)
    ax14 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(2, 0), colspan=1)
    ax15 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(3, 2), colspan=1)
    ax16 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(3, 1), colspan=1)
    ax17 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(3, 0), colspan=1)
    axsSoCplusb = [ax12, ax13, ax14, ax15, ax16, ax17]
    # SoC w/ VITpSoC plots
    ax18 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(2, 5), colspan=1)
    ax19 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(2, 4), colspan=1)
    ax20 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(2, 3), colspan=1)
    ax21 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(3, 5), colspan=1)
    ax22 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(3, 4), colspan=1)
    ax23 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(3, 3), colspan=1)
    axsSoCplusp = [ax18, ax19, ax20, ax21, ax22, ax23]

    # CC plot
    ax24 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(4, 0), colspan=1)
    ax25 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(4, 1), colspan=1)
    ax26 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(4, 2), colspan=1)
    ax27 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(4, 3), colspan=1)
    ax28 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(4, 4), colspan=1)
    ax29 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(4, 5), colspan=1)
    axsCC = [ax24, ax25, ax26, ax27, ax28, ax29]
    # Current plot
    ax30 = plt.subplot2grid(fig = fig, shape=(v_shape, h_shape), loc=(5, 0), colspan=6)

    # SoC
    linesSoCb = []
    id = 0
    for ax in axsSoCb:
        ax.grid(b=True, axis='y', linestyle='-', linewidth=2)
        ax.set_title(f'CANid {id}')
        ax.set_ylim([0, 100])
        linesSoCb.append(
                ax.bar(range(10),
                       [50]*10,
                    )
            )
        id+=1
    
    linesSoCp = []
    id = 0
    for ax in axsSoCp:
        ax.grid(b=True, axis='both', linestyle='-', linewidth=1)
        ax.set_title(f'CANid {id}')
        ax.set_ylim([0, 100])
        ax.set_xlim([0, 60])
        linesSoCp.append(
                ax.plot(range(60),
                        [[50]*10]*60,
                    )
            )
        ax.legend(labels_v, loc='center left')
        id+=1

    # SoC plus
    linesSoCplusb = []
    id = 0
    for ax in axsSoCplusb:
        ax.grid(b=True, axis='y', linestyle='-', linewidth=2)
        ax.set_title(f'CANid {id}')
        ax.set_ylim([0, 100])
        linesSoCplusb.append(
                ax.bar(range(10),
                       [50]*10,
                    )
            )
        id+=1
    
    linesSoCplusp = []
    id = 0
    for ax in axsSoCplusp:
        ax.grid(b=True, axis='both', linestyle='-', linewidth=1)
        ax.set_title(f'CANid {id}')
        ax.set_ylim([0, 100])
        ax.set_xlim([0, 60])
        linesSoCplusp.append(
                ax.plot(range(60),
                        [[50]*10]*60,
                    )
            )
        ax.legend(labels_v, loc='center left')
        id+=1
    
    # CC
    linesCC = []
    id = 0
    for ax in axsCC:
        ax.grid(b=True, axis='both', linestyle='-', linewidth=1)
        ax.set_title(f'CANid {id}')
        ax.set_ylim([0, 100])
        ax.set_xlim([0, 120])
        linesCC.append(
                ax.plot(range(120),
                       [50]*120,
                    #    label=labels

                )
            )
        ax.legend(labels_t, loc='center left')
        id+=1
    
    # Total provided current
    linesCurrent = []
    ax30.grid(b=True, axis='both', linestyle='-', linewidth=1)
    ax30.set_title(f'Current applied')
    ax30.set_ylim([-50, 50])
    ax30.set_xlim([0, 360])
    linesCurrent.append(
            ax30.plot(range(360),
                    [0]*360
                )
        )
# %%
    try:
        model : tf.keras.models.Sequential = tf.keras.models.load_model(
                filepath='Models/VIT', compile=False,
                # filepath=f'/mnt/WORK/QUT/TF/Battery_SoCv4/Models/Chemali2017/FUDS-models/48', compile=False,
                custom_objects={"RSquare": tfa.metrics.RSquare}
            )
        print('VIT model loaded')
        VITpSOC : AutoFeedBack = AutoFeedBack(units=510,
                out_steps=30, num_features=1
            )
        VITpSOC.load_weights('Models/VITpSoC/12')
        print('VITpSoC model loaded')
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
        print('One of the models failed to load.')

    fs : int = 4
    iSoC = 0.78
    MEAN = np.array([-0.35640615,  3.2060466 , 30.660755  ], dtype=np.float32)
    STD  = np.array([ 0.9579658 ,  0.22374259, 13.653275  ], dtype=np.float32)
    cSoC = [np.zeros(shape=(500, 10), dtype=np.float32)]*6
    pSoC = [np.ones(shape=(500, 10), dtype=np.float32)*iSoC]*6
    CC = np.ones(shape=(120, 6), dtype=np.float32)*iSoC

# %%
    i = 0
    tick = time.perf_counter()
    def animate(self):
        global i
        # TotI : pd.DataFrame = pd.read_csv(
        #         f'demo/current.csv',
        #         # skiprows=11000
        #     ).tail(2000+i)[:2000]/18
        TotI = pd.read_csv(f'{output_loc}Current.csv').tail(500)
        for bms in range(0, len(linesSoCb)):
            # BMSv : pd.DataFrame = pd.read_csv(
            #         f'demo/voltages/CANid_{bms}.csv',
            #         # skiprows=10000
            #     ).tail(2000+i)[:2000]
            # BMSt : pd.DataFrame = pd.read_csv(
            #         f'demo/temperatures/CANid_{bms}.csv',
            #         # skiprows=6000
            #     ).tail(2000+i)[:500]
            BMSv = pd.read_csv(f'{output_loc}Voltages/CANid_{bms}.csv').tail(60)
            # BMSb = pd.read_csv(f'{output_loc}BalanceInfo/CANid_{bms}.csv').tail(1)
            BMSt = pd.read_csv(f'{output_loc}Temperatures/CANid_{bms}.csv').tail(120)
        
            for cell in range(0,len(linesSoCb[bms])):
                VIT, VITpSoC, lite = SoC(BMSv, TotI, BMSt, pSoC[bms], cell, interpreter)
                #?  SoC plots
                #* Bar plot SoC
                linesSoCb[bms][cell].set_height(
                       VIT*100
                    )
                linesSoCb[bms][cell].set_color(
                       my_cmap(VIT)
                    )
                #* Line plot SoC
                cSoC[bms][:,cell] = np.concatenate(
                        (cSoC[bms][:,cell], np.expand_dims(VIT, axis=0)),
                        axis=0
                    )[1:]
                linesSoCp[bms][cell].set_ydata(
                        cSoC[bms][-60:,cell]*100
                    )
                #?  SoC+ plots
                #* Bar plot SoC+
                linesSoCplusb[bms][cell].set_height(
                       VITpSoC*100
                    )
                linesSoCplusb[bms][cell].set_color(
                       my_cmap(VITpSoC)
                    )
                #* Line plot SoC+
                pSoC[bms][:,cell] = np.concatenate(
                        (pSoC[bms][:,cell], np.expand_dims(VITpSoC, axis=0)),
                        axis=0
                    )[1:]
                linesSoCplusp[bms][cell].set_ydata(
                        pSoC[bms][-60:,cell]*100
                    )
                # print(cSoC[bms][:,cell].shape)
                # print(pSoC[bms][:,cell].shape)
            #?  Coulumb Counter
            newCC = CC[-1,bms] + ccSoC(current=TotI[-i:].to_numpy()[0]*6, time_s=np.expand_dims(i, axis=0))
            CC[:,bms] = np.concatenate(
                        (CC[:,bms], newCC),
                        axis=0
                    )[1:]
            linesCC[bms][0].set_ydata(
                    CC[:,bms]*100
                )
        #?  Current plot
        linesCurrent[0][0].set_ydata(
                TotI.to_numpy()[-360-i:-i]*18
            )
        tuples_return = tuple(linesCurrent[0])
        for bms in range(0, len(linesSoCb)):
            tuples_return += tuple(linesSoCb[bms])
            tuples_return += tuple(linesSoCp[bms])
            tuples_return += tuple(linesSoCplusb[bms])
            tuples_return += tuple(linesSoCplusp[bms])
        i += 1
        print(time.perf_counter()-tick)
        return tuples_return

    ani = animation.FuncAnimation(      #blit syncs the animate as it completes
            fig, animate, interval=0, blit=True)
    fig.show()
    input('Exiting?')
#    color=my_cmap(SoC[-1,:, id])
# %%