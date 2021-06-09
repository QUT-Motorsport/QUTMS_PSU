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