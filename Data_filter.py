# %%
import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import tflite_runtime.interpreter as tflite

from scipy import integrate # integration with trapizoid
from sys import platform

# mpl.rcParams['figure.figsize'] = (32, 16)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'Bender'

my_cmap = cm.get_cmap('jet_r')

if (platform == 'win32'):
    output_loc  : str = 'DataWin\\BMS_data\\July14\\'
else:
    output_loc  : str = 'Data/BMS_data/July14/'
# %%
# Current data and plot
BMSCurrent  : pd.DataFrame = pd.read_csv(
    filepath_or_buffer=f'{output_loc}Current.csv', sep=",")
print(BMSCurrent.head())
# Use Date converted to ints up to seconds ONLY
BMSCurrent['TSeconds']=(pd.to_datetime(BMSCurrent['Date_Time'].iloc[:].str.slice(stop=-4),
               format='%d/%m/%Y %H:%M:%S')- pd.to_datetime('1970-01-01')).dt.total_seconds()

start_cycling = pd.to_datetime(BMSCurrent['Date_Time'].iloc[0][:-4])
BMSCurrent['Cycle_Time(s)']=(
        pd.to_datetime(BMSCurrent['Date_Time'].iloc[:].str.slice(stop=-4)) \
        - start_cycling
    ).dt.total_seconds().astype(int)
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
        BMSsFUDS_V[i]['Cycle_Time(round)'] = np.round(BMSsFUDS_V[i]['Cycle_Time(s)'])
        BMSsFUDS_T[i]['Cycle_Time(round)'] = np.round(BMSsFUDS_T[i]['Cycle_Time(s)'])
        print(f'Cycle samples of V at BMS{i} - {BMSsFUDS_V[i].shape} and T -{BMSsFUDS_T[i].shape}')
    except Exception as e:
        print(f'{e} + BMS {i}')

# %%
# def check
# bmsID   : int = 0
# pd.set_option('mode.chained_assignment', None)
for bmsID in range(0, 6): #! BMS 2 is a problem with temps
    record = BMSsFUDS_V[bmsID][BMSsFUDS_V[bmsID]['Cycle_Time(s)']==BMSCurrent['Cycle_Time(s)'][0]].copy()
    record_temps = BMSsFUDS_T[bmsID][BMSsFUDS_T[bmsID]['Cycle_Time(round)']==BMSCurrent['Cycle_Time(s)'][0]][:1].copy()
    dictin = {
            'Date_Time(ms)' : record['Date_Time'].values,
            'Cycle_Time(s)' : record['Cycle_Time(s)'].values,
            'Current(A)'    : BMSCurrent['Current(A)'][0]/18,
            '6-Cell_1'      : record[record.columns[2]].values,
            '6-Cell_2'      : record[record.columns[3]].values,
            '6-Cell_3'      : record[record.columns[4]].values,
            '6-Cell_4'      : record[record.columns[5]].values,
            '6-Cell_5'      : record[record.columns[6]].values,
            '6-Cell_6'      : record[record.columns[7]].values,
            '6-Cell_7'      : record[record.columns[8]].values,
            '6-Cell_8'      : record[record.columns[9]].values,
            '6-Cell_9'      : record[record.columns[10]].values,
            '6-Cell_10'     : record[record.columns[11]].values,
            'Sns_1'         : record_temps[record_temps.columns[2]].values,
            'Sns_2'         : record_temps[record_temps.columns[3]].values,
            'Sns_3'         : record_temps[record_temps.columns[4]].values,
            'Sns_4'         : record_temps[record_temps.columns[5]].values,
            'Sns_5'         : record_temps[record_temps.columns[6]].values,
            'Sns_6'         : record_temps[record_temps.columns[7]].values,
            'Sns_7'         : record_temps[record_temps.columns[8]].values,
            'Sns_8'         : record_temps[record_temps.columns[9]].values,
            'Sns_9'         : record_temps[record_temps.columns[10]].values,
            'Sns_10'        : record_temps[record_temps.columns[11]].values
        }    
    # record[record.columns[2:12]].values[0]
    test = pd.DataFrame.from_dict(data=dictin)
    prev_record = record
    prev_record_temps = record_temps
    for i in range(1, len(BMSCurrent['Cycle_Time(s)'])):
        record_fs = BMSsFUDS_V[bmsID][BMSsFUDS_V[bmsID]['Cycle_Time(round)']==BMSCurrent['Cycle_Time(s)'][i]].copy()
        record = record_fs[:1].copy()
        record_temps_fs = BMSsFUDS_T[bmsID][BMSsFUDS_T[bmsID]['Cycle_Time(round)']==BMSCurrent['Cycle_Time(s)'][i]].copy()
        record_temps = record_temps_fs[:1].copy()

        #!Check for outliers
        j=1
        while( (record[record.columns[2:12]].values < 2.2).any() ):
            # print(f'underVoltage outliers at {i} with {j}fs')
            if j >= len(record_fs):
                record = record_fs.loc[j-3:j+1-3,:].copy()
            else:
                indexes = np.arange(2,12)[(record[record.columns[2:12]].values < 2.2)[0]]
                record.loc[0,record.columns[indexes]] = record_fs.loc[:,record_fs.columns[indexes]][j:j+1].values[0].copy()
                # new_val = record_fs.loc[:,record_fs.columns[indexes]][j:j+1].values[0].copy()
                # record.loc[:,record.columns[indexes]] = new_val.copy()
            
            j+=1
            
        j=1
        while( (record[record.columns[2:12]].values > 4.0).any()):
            # print(f'OverVoltage outliers at {i} with {j}fs')
            if j >= len(record_fs):
                record = record_fs.loc[j-3:j+1-3,:].copy()
            else:
                indexes = np.arange(2,12)[(record[record.columns[2:12]].values > 4.0)[0]]
                record.loc[0, record.columns[indexes]] = record_fs.loc[:,record_fs.columns[indexes]][j:j+1].values[0].copy()
                # record.loc[:, record.columns[indexes]] = record_fs[record_fs.columns[indexes]].iloc[j:j+1]
            
            j+=1

        j=1
        while( (record_temps[record_temps.columns[2:12]].values < 17).any() ):
            # print(f'underTemperature outliers at {i} with {j}fs')
            if j >= len(record_temps_fs):
                record_temps = record_temps_fs.loc[j-2:j+1-2,:].copy()
            else:
                indexes = np.arange(2,12)[(record_temps[record_temps.columns[2:12]].values < 5)[0]]
                record_temps.loc[0,record_temps.columns[indexes]] = record_temps_fs.loc[:,record_temps_fs.columns[indexes]][j:j+1].values[0].copy()
                # record_temps.loc[:,record_temps.columns[indexes]] = record_temps_fs[record_temps_fs.columns[indexes]].iloc[j:j+1]

            j+=1

        j=1
        while( (record_temps[record_temps.columns[2:12]].values > 57).any()):
            # print(f'overTemperature outliers at {i} with {j}fs')
            if j >= len(record_temps_fs):
                record_temps = record_temps_fs.loc[j-2:j+1-2,:].copy()
            else:
                indexes = np.arange(2,12)[(record_temps[record_temps.columns[2:12]].values > 55)[0]]
                record_temps.loc[:,record_temps.columns[indexes]] = record_temps_fs[record_temps_fs.columns[indexes]].iloc[j:j+1]
            
            j+=1

        #! Voltage missing but because of one sensor, entire array gets replaced with previos one
        if len(record) == 0:
            # print(f'Voltage Mising at {i} time')
            record = prev_record
        else:
            prev_record = record
        if len(record_temps) == 0:
            # print(f'Tempearature Mising at {i} time')
            record_temps = prev_record_temps
        else:
            prev_record_temps = record_temps
        
        #!If ANY start cleaning
        dictin = {
                'Date_Time(ms)' : record['Date_Time'].values,
                'Cycle_Time(s)' : record['Cycle_Time(s)'].values,
                'Current(A)'    : BMSCurrent['Current(A)'][i]/18,  #! Divide by 18
                '6-Cell_1'      : record[record.columns[2]].values,
                '6-Cell_2'      : record[record.columns[3]].values,
                '6-Cell_3'      : record[record.columns[4]].values,
                '6-Cell_4'      : record[record.columns[5]].values,
                '6-Cell_5'      : record[record.columns[6]].values,
                '6-Cell_6'      : record[record.columns[7]].values,
                '6-Cell_7'      : record[record.columns[8]].values,
                '6-Cell_8'      : record[record.columns[9]].values,
                '6-Cell_9'      : record[record.columns[10]].values,
                '6-Cell_10'     : record[record.columns[11]].values,
                'Sns_1'         : record_temps[record_temps.columns[2]].values,
                'Sns_2'         : record_temps[record_temps.columns[3]].values,
                'Sns_3'         : record_temps[record_temps.columns[4]].values,
                'Sns_4'         : record_temps[record_temps.columns[5]].values,
                'Sns_5'         : record_temps[record_temps.columns[6]].values,
                'Sns_6'         : record_temps[record_temps.columns[7]].values,
                'Sns_7'         : record_temps[record_temps.columns[8]].values,
                'Sns_8'         : record_temps[record_temps.columns[9]].values,
                'Sns_9'         : record_temps[record_temps.columns[10]].values,
                'Sns_10'        : record_temps[record_temps.columns[11]].values
            }    
        test = test.append(pd.DataFrame.from_dict(data=dictin), ignore_index=False)
    test = test.reset_index()
    test = test.drop(columns=['index'], axis=1)
    test.to_csv(f'{output_loc}Filt_CANid_{bmsID}.csv')
    plt.figure()
    plt.plot(test['Cycle_Time(s)'], test.iloc[:,3:13])
    plt.title(f'Voltages of BMS {bmsID}')
    plt.show()
    plt.figure()
    plt.plot(test['Cycle_Time(s)'], test.iloc[:,13:23])
    plt.title(f'Temperatues of BMS {bmsID}')
    plt.show()
# %%