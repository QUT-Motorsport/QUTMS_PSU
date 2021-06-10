#!/usr/bin/python
# %% [markdown]
# # # AMS Heavy plotting
# Script intended to ptoduce multiplots of Voltages and temperature over time
#csv files. State of Charge monitor is a matter of another script.
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

from datetime import datetime
import getpass

mpl.rcParams['figure.figsize'] = (32, 16)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'Bender'

my_cmap = cm.get_cmap('jet_r')

output_loc = f'/home/{getpass.getuser()}/tmp/{datetime.now().strftime("%B%d")}/'
# def get_demo_data():
#     BMSv_DataFrames = []
#     BMSt_DataFrames = []
#     for i in range(0, 6):
#         BMSv_DataFrames.append(pd.read_csv(f'demo/voltages/CANid_{i}.csv'))
#         BMSt_DataFrames.append(pd.read_csv(f'demo/temperatures/CANid_{i}.csv'))
#     current = pd.read_csv(f'demo/current.csv')
#     return BMSv_DataFrames, BMSt_DataFrames, current
# %%
if __name__ == '__main__':
    #! Setup all plots for animations
    # BMSv_DataFrames, BMSt_DataFrames, current = get_demo_data()
    # fig, axs = plt.subplots(2, 1, figsize=(28,12))


    labels_v = ['cell-1','cell-2','cell-3','cell-4','cell-5','cell-6','cell-7','cell-8','cell-9','cell-10']
    labels_t = ['sns-1','sns-2','sns-3','sns-4','sns-5','sns-6','sns-7','sns-8','sns-9','sns-10','sns-11','sns-12','sns-13','sns-14']
    fig = plt.figure(num = 0, figsize=(32,16))
    fig.suptitle("AMS here                          VOLTAGES && TEMPERATURES                            AMS here",
                fontsize=12)
    # Voltages Bars
    ax0  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(0, 2), colspan=1)
    ax1  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(0, 1), colspan=1)
    ax2  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(0, 0), colspan=1)
    ax3  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(1, 2), colspan=1)
    ax4  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(1, 1), colspan=1)
    ax5  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(1, 0), colspan=1)
    axsVb = [ax0, ax1, ax2, ax3, ax4, ax5]
    # Voltages Plots
    ax6  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(0, 5), colspan=1)
    ax7  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(0, 4), colspan=1)
    ax8  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(0, 3), colspan=1)
    ax9  = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(1, 5), colspan=1)
    ax10 = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(1, 4), colspan=1)
    ax11 = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(1, 3), colspan=1)
    axsVp = [ax6, ax7, ax8, ax9, ax10, ax11]
    # Temperature Plot
    ax12 = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(2, 0), colspan=1)
    ax13 = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(2, 1), colspan=1)
    ax14 = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(2, 2), colspan=1)
    ax15 = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(2, 3), colspan=1)
    ax16 = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(2, 4), colspan=1)
    ax17 = plt.subplot2grid(fig = fig, shape=(3, 6), loc=(2, 5), colspan=1)
    axsTp = [ax12, ax13, ax14, ax15, ax16, ax17]

    linesVb = []
    id = 0
    for ax in axsVb:
        ax.grid(b=True, axis='y', linestyle='-', linewidth=2)
        ax.set_title(f'CANid {id}')
        ax.set_ylim([2.8, 3.7])
        linesVb.append(
                ax.bar(range(10),
                       [3.0]*10,
                    )
            )
        id+=1
    
    linesVp = []
    id = 0
    for ax in axsVp:
        ax.grid(b=True, axis='both', linestyle='-', linewidth=1)
        ax.set_title(f'CANid {id}')
        ax.set_ylim([2.8, 3.7])
        ax.set_xlim([0, 60])
        linesVp.append(
                ax.plot(range(60),
                        [[0.0]*10]*60,
                    )
            )
        ax.legend(labels_v, loc='center left')
        id+=1
    
    linesTp = []
    id = 0
    for ax in axsTp:
        ax.grid(b=True, axis='both', linestyle='-', linewidth=1)
        ax.set_title(f'CANid {id}')
        ax.set_ylim([15, 50])
        ax.set_xlim([0, 120])
        linesTp.append(
                ax.plot(range(120),
                       [[0]*14]*120,
                    #    label=labels

                )
            )
        ax.legend(labels_t, loc='center left')
        id+=1
# %%
    # i = 0
    tick = time.perf_counter()
    def animate(self):        
        # global i
        for bms in range(0, len(linesVp)):
            # BMSv_DataFrames.append(pd.read_csv(f'demo/voltages/CANid_{i}.csv'))
            # BMSt_DataFrames.append(pd.read_csv(f'demo/temperatures/CANid_{i}.csv'))
            BMSv = pd.read_csv(f'{output_loc}Voltages/CANid_{bms}.csv').tail(60)
            BMSb = pd.read_csv(f'{output_loc}BalanceInfo/CANid_{bms}.csv').tail(1)
            BMSt = pd.read_csv(f'{output_loc}Temperatures/CANid_{bms}.csv').tail(120)
        
            for cell in range(0,len(linesVp[bms])):
                #* Voltage Bar
                linesVb[bms][cell].set_height(
                    BMSv.iloc[-1, cell+2]
                    )
                if(BMSb.iloc[-1, cell+3] == 1):
                    linesVb[bms][cell].set_color('r')
                else:
                    linesVb[bms][cell].set_color('b')
                #* Voltage Plot
                linesVp[bms][cell].set_xdata(
                        range(len(BMSv.iloc[:,cell+2].to_numpy()))
                    )
                linesVp[bms][cell].set_ydata(
                        BMSv.iloc[:,cell+2].to_numpy()
                    )
            for sns in range(0,len(linesTp[bms])):
                #* Tempearature Bar
                linesTp[bms][sns].set_xdata(
                        range(len(BMSt.iloc[:,sns+2].to_numpy()))
                    )
                linesTp[bms][sns].set_ydata(
                        BMSt.iloc[:,sns+2].to_numpy()
                    )
            # if(any(BMSb.iloc[-1, 3:-1])):
            #     print(f"BV for BMS:{bms} - {BMSb.iloc[-1, 2]}V")
        tuples_return = tuple()
        for bms in range(0, len(linesVb)):
            tuples_return += tuple(linesVb[bms])
            tuples_return += tuple(linesVp[bms])
        for sns in range(0,len(linesTp[bms])):
            tuples_return += tuple(linesTp[bms])
        # i += 1
        print(time.perf_counter()-tick)
        return tuples_return
        # return tuple(linesVb[0]) + tuple(linesVb[1]) + tuple(linesVb[2]) +\
        #        tuple(linesVb[3]) + tuple(linesVb[4]) + tuple(linesVb[5]) +\
        #        tuple(linesVp[0]) + tuple(linesVp[1]) + tuple(linesVp[2]) +\
        #        tuple(linesVp[3]) + tuple(linesVp[4]) + tuple(linesVp[5]) +\
        #        tuple(linesTp[0]) + tuple(linesTp[1]) + tuple(linesTp[2]) +\
        #        tuple(linesTp[3]) + tuple(linesTp[4]) + tuple(linesTp[5])

    # ani = animation.FuncAnimation(
    #         fig, animate, interval=1000, blit=True, save_count=50)
    ani = animation.FuncAnimation(
            fig, animate, interval=0, blit=True)
    fig.show()
    input('Exiting?')