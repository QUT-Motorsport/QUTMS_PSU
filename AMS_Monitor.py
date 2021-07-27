#!/usr/bin/python
# %% [markdown]
# # # AMS Monitoring and logging script
# Ensure this script is started before the AMS is powered on, as it must be 
# ready to respond to the AMS's charging check request
# %%
import sys, glob, os, io

import serial as sr
import numpy as np

import json, getpass, time

from multiprocessing import Process, Queue
import concurrent.futures

from datetime import datetime

from sys import platform

DEFAULT_BAUD    : int = 115200

#! Choose the quick read/write location
if (platform == 'win32'):
    output_loc = (f'C:\\Users\\{getpass.getuser()}\\'
                 f'tmp\\{datetime.now().strftime("%B%d")}\\')
else:
    output_loc = (f'/home/{getpass.getuser()}'
                  f'/tmp/{datetime.now().strftime("%B%d")}/')

on_sequence = '1761634561' # ^M is the problem. On WIndows, it send us \r\n
                         #terminator. Workout the replacer

AWAITING : bool  = False
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

def worker_UART(queue : Queue, serial : sr, stream : io.TextIOWrapper) -> None:
    """ UART data read worker. Intended to store everything to a queue with
    higher priotiry than everything else.

    Args:
        queue (Queue): Multiprocessing storage
        serial (sr): UART serial port
        stream (io.TextIOWrapper): Text stream to process CR/LF
    """
    jsonlog = open(f'{output_loc}/jsonLog.txt', 'a')
    try:
        # while(1):
        #     q.put(jsonlog.readline())
        
        while serial.inWaiting():
            data    : str = stream.readline()
            queue.put(data)
            jsonlog.write(data)
    except KeyboardInterrupt:
        print("Terminating UART Worker")
        AWAITING = False
        jsonlog.close()

def worker_toCSV(queue : Queue) -> None:
    """ Data JSON parser and writer to separate CSV file based on CANid and 
    type of reseived information: Voltage, Temperature, Balancers.

    Args:
        queue (Queue): A torage populated by worker_UART with whatever messages
                       has arrived by now.
    """
    while(queue.qsize() > 0):
        data = queue.get()   
        try:
            iType, values = list(json.loads(data).items())[0]
            record = [
                    f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]}',
                    f'{values["RT"]}',
                    #f'{values["BalanceVoltage"]/1000}'
                ]
            if(iType == 'BalanceInfo'):
                record += [f'{values["BalanceVoltage"]/1000}']
                record += [
                        f'{v}' if v != ' ' else '0'
                            for v in format(values["BalanceState"], '10b')[::-1]
                    ]
            else:
                key = list(values.keys())[2]
                record += [f'{v}' for v in values[key]]
            record += '\n'
            with open(f'{output_loc}{iType}/CANid_{values["BMS"]}.csv', 'a') as f:
                f.write(','.join(record))
            print(f"removed: {queue.qsize()}")
            queue.task_done()
        except:
            if(data[:-2] == str(0x69FF69FE)):
                print('AMS charge check')
                # ser.write(bytes(on_sequence, "asci")) # "uint8"
                s.write(on_sequence)
                s.flush()
                # ser.write(0x69006901)
                # ser.write(np.array(0x69006901, dtype=np.uint8)[0])
                # ser.write(bytes('\r\n', "utf-8")) # "uint8"
                queue.task_done()
            else:
                print("Something failed")
                print(data)
                queue.task_done()
            pass
# %%
if __name__ == '__main__':
    print("AMS Monitor Starting\n")
    aports  : list[str] = serial_ports()
    print("====================\n")
    for i in range(0, len(aports)):
        print("({}) - {}\n".format(i+1, aports[i]))
    
    port    : int = int(input("Please select a serial port: "))-1
    ser     : sr  = sr.Serial(aports[port], DEFAULT_BAUD)
    
    # Configurate Line terminator - CR/LF (0x0D 0x0A) - EOL
    s = io.TextIOWrapper(io.BufferedRWPair(ser, ser),newline="\r\n")
    print("Connected to serial port, awaiting AMS wakeup sequence...\n")
    print("Press Stop Loop button on figure to terminate\n")

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
    # #! Define a Queue
    # q = queue.Queue()
    # AWAITING = True
    #     try:
    #         with open(f'{output_loc}/jsonLog.txt', 'r') as js:
    #             while(1):
    #                 q.put(js.readline())
    #                 print(f"added: {q.qsize()}")
    #                 time.sleep(1)
    #     except KeyboardInterrupt:
    #         print('Finishing the daemon')
            
    # # on_sequence = str(0x69006901)
    
    # # jsonlog = open(f'{output_loc}/jsonLog.txt', 'a')
    
    # daemon = Process(target=worker_UART, daemon=True, args=((pqueue),) )
    # daemon.start()
    
    #! Spawn multiple workers
    # while(AWAITING):
    #     while(q.qsize() > 0):
    #         worker_CSV()
    #         print(f'Size of the queue: {q.qsize()}')
    #     print('Awaiting more samples')
    #     time.sleep(0.5)
    # daemon.join()

    #! Try concurrent worker
    # with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    #     executor.map(worker_CSV)
    # daemon.join()
    # q.join()

    #! Wrap up streams
    # ser.close()
# %%
    jsonlog = open(f'{output_loc}/jsonLog.txt', 'a')    
    try:
        while(1):
            while ser.inWaiting():
                data = s.readline()
                jsonlog.write(data)
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
                    #! Perhaps I should try dictinory to keep them running
                    with open(f'{output_loc}{iType}/CANid_{values["BMS"]}.csv', 'a') as f:
                        f.write(','.join(record))

                except:
                    if(data[:-2] == str(0x69FF69FE)):
                        print('AMS charge check')
                        ser.write(bytes(on_sequence, "ascii")) # "uint8"
                        # s.write('69006901')
                        # s.flush()
                        # s.write('1761634561')
                        # ser.write(np.array(0x69006901, dtype=np.uint8)[0])
                        # ser.write(bytes('\r\n', "utf-8")) # "uint8"
                    else:
                        print(data)
                # data = jlfid.readline()
                # time.sleep(1)
                # print("Data out")
    except KeyboardInterrupt:
        print('Exiting ...')
        jsonlog.close()
        ser.close()
        # s.close()
    except Exception as e:
        print('Unhanled exception')
        print(e)
        jsonlog.close()
        ser.close()
# %%
