#!/usr/bin/python
# %% [markdown]
# # # AMS Monitoring and logging script
# Ensure this script is started before the AMS is powered on, as it must be 
# ready to respond to the AMS's charging check request
# %%
import sys, glob
import os, io
from numpy.core.fromnumeric import repeat
import serial as sr
import numpy as np

import json
from datetime import datetime
import getpass

DEFAULT_BAUD    : int = 115200
# %%
def serial_ports() -> list[str]:
    """ Lists serial port names
    Taken from: https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = sr.Serial(port)
            s.close()
            result.append(port)
        except (OSError, sr.SerialException):
            pass
    return result

# %%
if __name__ == '__main__':
    print("AMS Monitor Starting\n")
    aports  : list[str] = serial_ports()
    print("====================\n")
    for i in range(0, len(aports)):
        print("({}) - {}\n".format(i+1, aports[i]))
    
    port    : int = int(input("Please select a serial port: "))-1
    ser     : sr  = sr.Serial(aports[port], DEFAULT_BAUD)
    
    # # Configurate Line terminator - CR/LF (0x0D 0x0A) - EOL
    s = io.TextIOWrapper(io.BufferedRWPair(ser, ser),newline="\r\n")
    print("Connected to serial port, awaiting AMS wakeup sequence...\n")
    print("Press Stop Loop button on figure to terminate\n")


    # jlfid = open('jsonLog2.json', 'r')
    
    # Chooce the quick read/write location
    output_loc = f'/home/{getpass.getuser()}/tmp/{datetime.now().strftime("%B%d")}/'
    header_voltage = ['Date_Time', 'Step_Time(s)']
    header_voltage += [f'4Cell_{i}' for i in range(1,11)]
    header_voltage += '\n'
    header_temp = ['Date_Time', 'Step_Time(s)']
    header_temp += [f'Sns_{i}' for i in range(1,15)]
    header_temp += '\n'
    header_balance = ['Date_Time', 'Step_Time(s)', 'BalanceVoltage']
    header_balance += [f'4Cell_{i}' for i in range(1,11)]
    header_balance += '\n'

    N_BMSs : int = 6
    if not os.path.exists(output_loc):
        print('First time, making dirs')
        os.makedirs(output_loc)
        
        os.makedirs(output_loc+'VoltageInfo')
        os.makedirs(output_loc+'TemperatureInfo')
        os.makedirs(output_loc+'BalanceInfo')
        for i in range(0,N_BMSs):
            with open(output_loc+'VoltageInfo/'+f'CANid_{i}.csv', 'w+') as f:
                f.write(','.join(header_voltage))
            with open(output_loc+'TemperatureInfo/'+f'CANid_{i}.csv', 'w+') as f:
                f.write(','.join(header_temp))
            with open(output_loc+'BalanceInfo/'+f'CANid_{i}.csv', 'w+') as f:
                f.write(','.join(header_balance))
    else:
        print('Directories exists')
# %%
    # on_sequence = str(0x69006901)
    on_sequence = '69006901' # ^M is the problem. On WIndows, it send us \r\n terminator. Workout the replacer
    jsonlog = open(f'{output_loc}/jsonLog.txt', 'a')
    try:
        while(1):
            while ser.inWaiting():
                data = s.readline()
                jsonlog.write(data)
                # data = jlfid.readline()
                # while(data):
                # else:
                # jlfid.write(data + ',')
                # record = pd.read_json(data)
                # data = record[record.keys()[0]]    # Gets entire record info
                # # data.index[2]   # Gets the name
                # # data['BMS']
                # # data['RT']      # Try getting them by seconds
                # # data[2]         # Gets the values
                # iType, values = list(json.loads(data.replace('},','}')).items())[0]
                try:
                    # print(data)
                    iType, values = list(json.loads(data).items())[0]
                    record = [
                        f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]}',
                        f'{values["RT"]}',
                        #f'{values["BalanceVoltage"]/1000}'
                        ]
                    if(iType == 'BalanceInfo'):
                        record += [f'{values["BalanceVoltage"]/1000}']
                        record += [f'{v}' if v != ' ' else '0' for v in format(values["BalanceState"], '10b')[::-1]]
                    else:
                        key = list(values.keys())[2]
                        record += [f'{v}' for v in values[key]]
                    record += '\n'
                    with open(f'{output_loc}{iType}/CANid_{values["BMS"]}.csv', 'a') as f:
                        f.write(','.join(record))
                    # if(iType == 'BalanceInfo'):
                    #     key = list(values.keys())[2]
                    #     #TODO That part quite equivalent, see how can it be merged. Remove
                    #     record = [
                    #         f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]}',
                    #         f'{values["RT"]}',
                    #         f'{values["BalanceVoltage"]/1000}'
                    #         ]
                    #     # record += [f'{v}' for v in str(bin(values["BalanceState"]))[2:]]
                    #     record += [f'{v}' for v in str(bin(values["BalanceState"]))[:2:-1]]
                    #     record += '\n'
                    #     with open(f'{output_loc}{iType}/CANid_{values["BMS"]}.csv', 'a') as f:
                    #         f.write(','.join(record))
                    # else:
                    #     key = list(values.keys())[2]
                    #     record = [
                    #         f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]}',
                    #         f'{values["RT"]}'
                    #         ]
                    #     record += [f'{v}' for v in values[key]]
                    #     record += '\n'
                    #     with open(f'{output_loc}{key}/CANid_{values["BMS"]}.csv', 'a') as f:
                    #         f.write(','.join(record))

                except:
                    # print(data)
        #             break
        # break
                    if(data[:-2] == str(0x69FF69FE)):
                        print('AMS charge check')
                        # ser.write(bytes(on_sequence, "asci")) # "uint8"
                        s.write(on_sequence)
                        s.flush()
                        # ser.write(0x69006901)
                        # ser.write(np.array(0x69006901, dtype=np.uint8)[0])
                        # ser.write(bytes('\r\n', "utf-8")) # "uint8"
                    pass
                # data = jlfid.readline()
                # time.sleep(1)
                # print("Data out")
    except KeyboardInterrupt:
        print('Exiting ...')
        # jlfid.close()
        jsonlog.close()
        ser.close()
        # s.close()
    except Exception as e:
        print('Unhanled exception')
        print(e)
        jsonlog.close()
        ser.close()
# %%
