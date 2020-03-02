# Quantum device tuning via hypersurface sampling

The quantum devices used to implement spin qubits in semiconductors can be challenging to tune and charicterise. Often the best aproaches to tuning such devices is manual tuning or a simple huristic algorithm which is not flexible across devices. This repository contains the statistical tuning approach detailed in https://arxiv.org/abs/2001.02589 with some additional functionallity. This approach is promising as it make few assumptions about the device being tuned and hence can be applied to many systems without alteration.

## Dependancies
The required packages required to run the algorithm are:
```
cma
scikit_image
scipy
numpy
matplotlib
config
GPy
mkl
pyDOE
scikit_learn
skimage
```
## Using the algorithm
Using the aglgorithm varies depending on what measurment software you use in your lab. Specififically if your lab utilises pygor then you should call a different function to initiate the tuning.
### Without pygor
To use the algorithm without pygor you must create the following:
- jump
- measure
- check
- config_file

<ins>jump:</ins>
jump should be a function that takes an array of values and sets them to the device. It should also accept a flag that details whether the investiagation gates (typically plunger gates) should be used. Below is an example of how jump should be defined for a 5 gate device with 2 investigation (in this case plunger) gates.
```python
def jump(params,inv=False):
  if inv:
    labels = ["dac4","dac6"] #plunger gates
  else:
    labels = ["dac3","dac4","dac5","dac6","dac7"] #all gates
    
  assert len(params) == len(labels) #params needs to be the same length as labels
  for i in range(len(params)):
    set_value_to_dac(labels[i],params[i]) #function that takes dac key and value and sets dac to that value
  return params
```
<ins>measure:</ins>
measure should be a function that returns the measured current on the device.
```python
def measure():
  current = get_value_from_daq() #recieve a single current measurement from the daq
  return current
```
<ins>check:</ins>
check should be a function that returns the state of all relevant dac channels.
```python
def check():
  labels = ["dac3","dac4","dac5","dac6","dac7"] #all gates
  dac_state = [None]*len(labels)
  for i in range(len(labels)):
    dac_state[i] = get_current_dac_state(labels[i]) #function that takes dac key and returns state that channel is in
  return dac_state
```
<ins>config_file:</ins>
config_file should be a string that specifies the file path of a .json file containing a json opject that specifies the desired settings the user wants to use for tuning. An example string would be "demo_config.json". For information on what the config file should contain see the json config section.
### With pygor
To use the algorithm without pygor you must create the following:
- config_file
<ins>config_file:</ins>
config_file should be a string that specifies the file path of a .json file containing a json opject that specifies the desired settings the user wants to use for tuning. An example string would be "demo_config.json". For information on what the config file should contain see the json config section.
