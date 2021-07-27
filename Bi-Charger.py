#!/usr/bin/python
# %% [markdown]
# # # Bi-polar charger client
# # 
#
# %%
import os, sys
import socket
import struct
import matplotlib as mpl  # Plot functionality
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time, getpass

from datetime import datetime
from time import perf_counter
from sys import platform

# sys.path.append(os.getcwd() + '/..')
if (platform == 'win32'):
    from py_modulesWin.parse_excel import ParseExcelData
    train_dir	  : str ='..\\Data\\A123_Matt_Val_2nd'
    output_loc = (f'C:\\Users\\{getpass.getuser()}\\'
                 f'tmp\\{datetime.now().strftime("%B%d")}\\')
else:
    from py_modules.parse_excel import ParseExcelData
    train_dir	  : str ='Data/A123_Matt_Val_2nd'
    output_loc = (f'/home/{getpass.getuser()}'
                  f'/tmp/{datetime.now().strftime("%B%d")}/')

from datetime import datetime
import getpass

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['font.family'] = 'Bender'

try:
    # If ipykernel make plots interactive
    # %matplotlib qt
    print('Set to QT plots')
except:
    print('Interactive plots')
# >>> d = 1.234
# >>> b = struct.pack('d', d)
# >>> b
# b'X9\xb4\xc8v\xbe\xf3?'
# >>> d2, = struct.unpack('d', b)
# >>> 1.234
# %%
#! Messages strings
#* Engage remote lock: 0 - lock off, 1 - lock on
REMOTE_LOCK : list[str] = ['SYST:LOCK OFF\n', 'SYST:LOCK ON\n']

#* Run the charger: 0 - turn off, 1 - turn on
DC_START    : list[str] = ['OUTP:STAT OFF\n', 'OUTP:STAT ON\n']

#* Set inverse: 0 - Normal, 1 - Inverted
REMSV_INV   : list[str] = ['SYST:CONF:ANAL:REMSB:LEV NORM\n',
                           'SYST:CONF:ANAL:REMSB:LEV INV\n' ]

#* Get Error messages: 0 - all, 1 - single
ERROR_GET   : list[str] = ['SYST:ERR:ALL?\n', 'SYST:ERR:NEXT?\n']

#! SOURCE commands
#* Source GET values: 0 - Voltage, 1 - Current, 2 - Power
SOUR_GET  : list[str] = ['SOUR:VOLT?\n',
                         'SOUR:CURR?\n',
                         'SOUR:POW?\n' ]

#* Source SET: 0 - Voltage, 1 - Current, 2 - Power
SOUR_SET  : list[str] = ['SOUR:VOLT {0:.2f}\n',
                         'SOUR:CURR {0:.2f}\n',
                         'SOUR:POW {0:.2f}\n' ]
#* Source get limits: 0 - Voltage[L,H], 1 - Current[L,H], 2 - Power[H,H]
SOUR_LIM  : list[str, str] = [('SOUR:VOLT:LIM:LOW?\n','SOUR:VOLT:LIM:HIGH?\n'),
                              ('SOUR:CURR:LIM:LOW?\n','SOUR:CURR:LIM:HIGH?\n'),
                              ('SOUR:POW:LIM:HIGH?\n', 'SOUR:POW:LIM:HIGH?\n')]

#! SINK commands
#* Sink GET values: 0 - Voltage, 1 - Current, 2 - Power
SINK_GET    : list[str]  = ['SOUR:VOLT?\n',
                            'SINK:CURR?\n',
                            'SINK:POW?\n' ]

#* Sink SET: 0 - Voltage, 1 - Current, 2 - Power
SINK_SET    : list[str]  = ['SOUR:VOLT {0:.2f}\n',
                            'SINK:CURR {0:.2f}\n',
                            'SINK:POW {0:.2f}\n' ]
#* Sink get limits: 0 - Voltage[L,H], 1 - Current[L,H], 2 - Power[H,H]
SINK_LIM    : list[str, str] = [('SOUR:VOLT:LIM:LOW?\n','SOUR:VOLT:LIM:HIGH?\n'),
                                ('SINK:CURR:LIM:LOW?\n','SINK:CURR:LIM:HIGH?\n'),
                                ('SINK:POW:LIM:HIGH?\n', 'SINK:POW:LIM:HIGH?\n')]

#! Measurments commands
#* Get sensor measurments: 0 - Voltage, 1 - Current, 2 - Power
MEASURMENTS : list[str] = ['MEAS:VOLT:DC?\n',
                           'MEAS:CURR:DC?\n',
                           'MEAS:POW:DC?\n']
# %%
# Get data for profiling
columns = ['Test_Time(s)', 'Current(A)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']
tr_ls_df, _ = ParseExcelData(train_dir,
                                    #range(4 ,12), # DST
                                    range(22,25),   # FUDS
                                    columns)
#! 1 hour 40 minuts should be good for testing
step_time = (tr_ls_df[0]['Test_Time(s)'].iloc[:]-tr_ls_df[0]['Test_Time(s)'].iloc[0]).to_numpy() # 2000:8000 and 2000
current = (tr_ls_df[0]['Current(A)'].iloc[:]).to_numpy()
# current = np.append(current, (tr_ls_df[0]['Current(A)'].iloc[:]).to_numpy(), axis=0)
# current = np.append(current, (tr_ls_df[0]['Current(A)'].iloc[:]).to_numpy(), axis=0)
#! 3 times more

# %%
#? Socket transmit function
terminator = bytes('\n', 'ascii')
def writeline(s : socket, message : str) -> None:
    s.sendall(bytes(message,'ascii'))

def writeread(s : socket, message : str) -> str:
    s.send(bytes(message,'ascii'))
    buffer = b''
    while(buffer[-1:] != terminator):
        buffer += s.recv(1024)
    return buffer[:-1]

def get_float(s : socket, message : str) -> str:
    s.send(bytes(message,'ascii'))
    buffer = b''
    while(buffer[-1:] != terminator):
        buffer += s.recv(1024)
    return float(buffer[:-2])

def get_int(s : socket, message : str) -> str:
    s.send(bytes(message,'ascii'))
    buffer = b''
    while(buffer[-1:] != terminator):
        buffer += s.recv(1024)
    return int(buffer[:-2])
# %%
#? Configuration of the socket
PSB_IP      : str = '192.168.0.102'
PSB_PORT    : str = 5025

V_SOUR  : int = 72      # Volts charging
V_SINK  : int = 45      # Volts depletion

I_SOUR  : float = 2.0   # Amps charging
I_SINK  : float = 2.0   # Amps sinking

# curentCutOff : int = 30
powerMax     : int = 30000
# %%
# create the socket
#*AF_INET == ipv4
#*SOCK_STREAM == TCP
pcb = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
pcb.connect((PSB_IP, PSB_PORT))

# %%
#TODO: Determine timeout between messages. See what I can come up.

writeline(pcb, REMOTE_LOCK[1])    #!Engaging lock

#? First always charge, but this time it has to be discharge
# writeline(pcb, SOURCE_SET[0].format(V_SOUR))
writeline(pcb, REMSV_INV[1])

writeline(pcb, SOUR_SET[1].format(0))
writeline(pcb, SOUR_SET[2].format(powerMax))

writeline(pcb, SINK_SET[0].format(V_SINK))
writeline(pcb, SINK_SET[1].format(0))
writeline(pcb, SINK_SET[2].format(powerMax))

#TODO: Verefy && wait for input
var_order = ['Voltages', 'Current', 'Power']
for i in range(0,3):
    print(var_order[i])
    print('SOURce :: ', writeread(pcb, SOUR_GET[i]))
    print('SINK :: ', writeread(pcb, SINK_GET[i]))
    print('SOUrce:LIMit:LOW :: ', writeread(pcb, SOUR_LIM[i][0]))
    print('SOUrce:LIMit:HIGH :: ',writeread(pcb, SINK_LIM[i][1]))
    print('\n\n')
# input('Run tests')
# %%
# writeline(pcb, SOUR_SET[1].format(2))
# writeline(pcb, DC_START[1])
# c_time = 0
# tic = perf_counter()
# while(get_float(pcb, MEASURMENTS[0]) != V_SINK):
#     c_time = perf_counter() - tic
#     if(c_time > 10):
#         break
# writeline(pcb, DC_START[0])
# writeline(pcb, SOUR_SET[1].format(0))
# print(f'Time for Cap charge: {c_time}')
# #! Time the discharge capacitor
# # input('Check VOltage and Current respinse...\n\nConnect battery.')
# %%
# input('Press any to proceed...')
header_voltage = ['Date_Time', 'Current(A)', 'Voltage(V)', '\n']
if not os.path.exists(output_loc+'Current.csv'):
    print('First time, making dirs')
    with open(output_loc+'Current.csv', 'w+') as f:
        f.write(','.join(header_voltage))
else:
    print('Directories exists')

#!Add 5 minutes worth zeroes and high pulse
init = np.zeros(shape=(5*60), dtype=np.float32)
current = current[:int(len(current)/2)]
current = np.append(init, current)

writeline(pcb, SOUR_SET[1].format(0))
writeline(pcb, SINK_SET[1].format(0))
writeline(pcb, DC_START[1])
try:
    for i in range(0, len(current)): #! CHange back to zero
        targetCurrent = current[i]*40
        if(targetCurrent > 0 and targetCurrent < 200):
            writeline(pcb, SOUR_SET[1].format(targetCurrent))
            writeline(pcb, SOUR_SET[0].format(V_SOUR))
            writeline(pcb, SINK_SET[1].format(0))
        else:
            writeline(pcb, SINK_SET[1].format(abs(targetCurrent)))
            writeline(pcb, SINK_SET[0].format(V_SINK))
            writeline(pcb, SOUR_SET[1].format(0))
        record = [
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]}',
                f'{get_float(pcb, MEASURMENTS[1])}',
                f'{get_float(pcb, MEASURMENTS[0])}',
                '\n'
            ]
        with open(output_loc+'Current.csv', 'a') as f:
            f.write(','.join(record))

        print(f'Voltage :: {writeread(pcb, MEASURMENTS[0])}'
              f'  Current :: {writeread(pcb, MEASURMENTS[1])}' 
              f'  targetCurrent :: {targetCurrent}')
        
        time.sleep(1)
except:
    writeline(pcb, SOUR_SET[1].format(0))
    writeline(pcb, SINK_SET[1].format(0))
    writeline(pcb, SOUR_SET[0].format(V_SINK))
    writeline(pcb, DC_START[0])
writeline(pcb, SOUR_SET[1].format(0))
writeline(pcb, SINK_SET[1].format(0))
writeline(pcb, SOUR_SET[0].format(V_SINK))
writeline(pcb, DC_START[0])
print("IT IS OVER!!!")
# %%
#! Make a plot
#! Add IR model
#! Modefy alrorith until it reaches
#!-90A - 5 minutes
header_voltage = ['Date_Time', 'Current(A)', 'Voltage(V)', '\n']
if not os.path.exists(output_loc+'Current.csv'):
    print('First time, making dirs')
    with open(output_loc+'Current.csv', 'w+') as f:
        f.write(','.join(header_voltage))
else:
    print('Directories exists')

try:
    while(1):
        record = [
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]}',
                f'{get_float(pcb, MEASURMENTS[1])}',
                f'{get_float(pcb, MEASURMENTS[0])}',
                '\n'
            ]
        with open(output_loc+'Current.csv', 'a') as f:
            f.write(','.join(record))

        print(f'Voltage :: {writeread(pcb, MEASURMENTS[0])}'
              f'  Current :: {writeread(pcb, MEASURMENTS[1])}')
        
        time.sleep(1)
except:
    print('Reading failed or interupted')

# %%

header_voltage = ['Date_Time', 'Current(A)', 'Voltage(V)', '\n']
if not os.path.exists(output_loc+'Current.csv'):
    print('First time, making dirs')
    with open(output_loc+'Current.csv', 'w+') as f:
        f.write(','.join(header_voltage))
else:
    print('Directories exists')

# writeline(pcb, SOUR_SET[1].format(0))
# writeline(pcb, SINK_SET[1].format(0))
# writeline(pcb, DC_START[1])

# writeline(pcb, SOUR_SET[1].format(45))
# writeline(pcb, SOUR_SET[0].format(V_SOUR))
# writeline(pcb, SINK_SET[1].format(0))

try:
    while(1):
        record = [
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]}',
                f'{get_float(pcb, MEASURMENTS[1])}',
                f'{get_float(pcb, MEASURMENTS[0])}',
                '\n'
            ]
        with open(output_loc+'Current.csv', 'a') as f:
            f.write(','.join(record))

        print(f'Voltage :: {writeread(pcb, MEASURMENTS[0])}'
              f'  Current :: {writeread(pcb, MEASURMENTS[1])}' 
              f'  targetCurrent :: {45}')
        
        time.sleep(1)
except:
    writeline(pcb, SOUR_SET[1].format(0))
    writeline(pcb, SINK_SET[1].format(0))
    writeline(pcb, SOUR_SET[0].format(V_SINK))
    # writeline(pcb, DC_START[0])
writeline(pcb, SOUR_SET[1].format(0))
writeline(pcb, SINK_SET[1].format(0))
writeline(pcb, SOUR_SET[0].format(V_SINK))
writeline(pcb, DC_START[0])
# %%
# targetCurrent = 3.6 # 2.5*18 #! Balance current: 20-> 3.6A
# # 3.6 * 10 = 36, 36*2 = 72
# writeline(pcb, SOUR_SET[1].format(targetCurrent))
# writeline(pcb, SINK_SET[1].format(0))
# writeline(pcb, SOUR_SET[0].format(V_SOUR))
# writeline(pcb, DC_START[1])
# tic = perf_counter()
# try:
#     while(1):
#         print(f'Time charging: {perf_counter()-tic}'
#             f'  Voltage :: {writeread(pcb, MEASURMENTS[0])}'
#             f'  Current :: {writeread(pcb, MEASURMENTS[1])}'
#             f'  delay - 5')
#         time.sleep(5)
# except:
#     writeline(pcb, SOUR_SET[1].format(0))
#     writeline(pcb, SINK_SET[1].format(0))
#     writeline(pcb, SOUR_SET[0].format(V_SINK))
#     writeline(pcb, DC_START[0])
# %%
#TODO: Beatiful plot
#?float(str(value).split(' ')[1])
try:
    # fig, axs = plt.subplots(2, 1, figsize=(28,12))
    fig, axs = plt.subplots(figsize=(28,12))
    test_time = range(0, 60)
    list_current = [0]*60
    list_voltage = [0]*60
    # ax1 = axs[0]
    ax1 = axs
    line, = ax1.plot(test_time, list_current, label="Current", color='#0000ff')
    ax1.grid(b=True, axis='both', linestyle='-', linewidth=1)
    ax1.set_xlabel("Time Slice (60s)", fontsize=32)
    ax1.set_title(f'Plotting at time ...')
    ax1.set_ylim([-10.3,10.3])
    ax1.set_xlim([0.00,60.0])

    t_Start = perf_counter()
    global targetCurrent 
    targetCurrent = current[0]

    def animate(i):    
        # Measure and format current
        value = writeread(pcb, MEASURMENTS[0])
        currentCurrent = float(str(value).split(' ')[2])
        list_current.append(currentCurrent)
        value = writeread(pcb, MEASURMENTS[1])
        currentVoltage = float(str(value).split(' ')[2])
        list_voltage.append(currentVoltage)

        #! Round that value
        if (currentCurrent == targetCurrent):
            print('Set new current')
            c_time = int(perf_counter()-t_Start)
            index = np.where(np.round(step_time) == c_time)[0][0]
            targetCurrent = current[index]
            if(targetCurrent < 0):
                writeline(pcb, SINK_SET[1].format((targetCurrent)))
                #? Wait?
                writeline(pcb, SINK_SET[0].format(V_SINK))

            else:
                writeline(pcb, SOUR_SET[0].format(V_SOUR))
                writeline(pcb, SOUR_SET[1].format(targetCurrent))
        else:
            print('Target current not reached: {int(perf_counter()-t_Start)}')

        line.set_ydata(list_current[-60:])
        line.set_title(f'Plotting at time {c_time}')    
        return line,


    writeline(pcb, DC_START[1])
    ani = animation.FuncAnimation(
        fig, animate, interval=1000, blit=True, save_count=50)
    fig.show()
except Exception as e:
    print(e)
    writeline(pcb, SOUR_SET[1].format(0.0))
    writeline(pcb, SINK_SET[1].format(0.0))
    writeline(pcb, DC_START[0])
    pcb.close()
input('Ended???')
writeline(pcb, REMOTE_LOCK[0])    #!Release the lock
pcb.close()
# %%
# while True:
#     full_msg = ''
#     while True:
#         msg = s.recv(8)
#         if len(msg) <= 0:
#             break
#         full_msg += msg.decode("utf-8")

#     if len(full_msg) > 0:
#         print(full_msg)
# %%