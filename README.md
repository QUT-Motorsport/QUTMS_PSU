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
```

Solve permission error by chmod 666 or a+rw to port. To solve Unix permission error
permenately, add yourself to the group 'dialout'.
```
sudo usermod -a -G dialout $USER
```