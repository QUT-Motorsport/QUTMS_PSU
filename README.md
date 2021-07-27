# QUTMS_PSU
## Matlab
Only MatLab full installation above 2020a were tested.
## Python
Serial library installationn
```
# Using PIP        
pip install PySerial

# Using conda

conda install PySerial

# if missing:
conda install matplotlib pandas numpy
```

Solve permission error by chmod 666 or a+rw to port. To solve Unix permission error
permenately, add yourself to the group 'dialout'.
```
sudo usermod -a -G dialout $USER
```

Use following command to debug plots thoru jupyter notebook in VScode
```
%matplotlib qt
```

## Bring missing library for novel method
```
ln -s ../TF/Battery_SoCv4/py_modules/AutoFeedBack.py AutoFeedBack.py

```

## Commince logging and plotting
0) Create a tmp folder in the user directory if intended to use default setup

1) First execute AMS python logger and verefy that all folders gets filled with data.
Note that if some of BMSes do not send either Voltage, Temperature or balancing, plotting will fail.
```
python AMS_Monitor.py
```
2) All data has to be saved into quick write area. Cache space or RAMdisk created area will do.
Standard location for all OSes will be $USER$/tmp/{date-today}
Make sure tmp folder exists
```
python AMS_heavy_plot.py
```
