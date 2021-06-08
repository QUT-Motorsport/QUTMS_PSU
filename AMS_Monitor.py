#!/usr/bin/python
# %% [markdown]
# # # AMS Monitoring and logging script
# Ensure this script is started before the AMS is powered on, as it must be 
# ready to respond to the AMS's charging check request
# %%
import sys, glob
import io
from numpy.core.fromnumeric import repeat
import serial as sr

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
    
    # port    : int = int(input("Please select a serial port: "))-1
    # ser     : sr  = sr.Serial(aports[port], DEFAULT_BAUD)
    
    # # Configurate Line terminator - CR/LF (0x0D 0x0A) - EOL
    # s = io.TextIOWrapper(io.BufferedRWPair(ser, ser),newline="\r\n")
    print("Connected to serial port, awaiting AMS wakeup sequence...\n")
    print("Press Stop Loop button on figure to terminate\n")

# %%
    iter : int = 0
    jlfid = open('jsonLog.json', 'a')
    jlfid.write('{\n')
    tlfid = open('textLog.txt', 'a')

    try:
        while(1):
            while ser.in_waiting():
                data = readline(s)

            # time.sleep(1)
    except:
        print('Exiting ...')
        jlfid.write('}')
        jlfid.close()
        tlfid.close()
        ser.close()
        # s.close()

# %%
