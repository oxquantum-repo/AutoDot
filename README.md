# Quantum device tuning via hypersurface sampling

The quantum devices used to implement spin qubits in semiconductors can be challenging to tune and charicterise. Often the best aproaches to tuning such devices is manual tuning or a simple huristic algorithm which is not flexible across devices. This repository contains the statistical tuning approach detailed in https://arxiv.org/abs/2001.02589 with some additional functionallity. This approach is promising as it make few assumptions about the device being tuned and hence can be applied to many systems without alteration.

## Dependancies
The required packages required to run the algorithm are:
```
scikit_image
scipy
numpy
matplotlib
GPy
mkl
pyDOE
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

#### How to run
To run tuning without pygor once the above has been defined call the following:
```python
import AutoDot
AutoDot.tune.tune_from_file(jump,measure,check,config_file)
```
### With pygor
To use the algorithm without pygor you must create the following:

<ins>config_file:</ins>
config_file should be a string that specifies the file path of a .json file containing a json opject that specifies the desired settings the user wants to use for tuning. An example string would be "demo_config.json". For information on what the config file should contain see the json config section. Additional fields are required to specify pygor location and settup.
#### How to run
To run tuning with pygor once the above has been defined call the following:
```python
import AutoDot
AutoDot.tune.tune_with_pygor_from_file(config_file)
```
## Config structure
Here is an [example config file](demo_config.json) containing all the relevent fields and below is a dicussion about each fields function
```
"path_to_pygor":"/path_to/pygor/package"
```
Absolute path to pygor package. (only required if using pygor)
```
"ip":"http://123.123.123.12:8000/RPC2"
```
URL of pygor server. If not specified mock device is built (only required if using pygor)
```
"gates":["c10","c3","c4","c5","c6","c7","c9"]
```
Labels for the DAC channels that the gates are attached to. (only required if using pygor)
```
"plunger_gates":["c5","c9"]
```
Labels for the DAC channels that the investigation/plunger gates are attached to. (only required if using pygor)
```
"bias_chans":["c1"]
```
Label(s) for the DAC channel(s) that control the bias. (only required if using pygor)
```
"bias_vals":[20]
```
Value(s) to set the bias DAC channel(s) to. (only required if using pygor)
```
"chan_no":0
```
0 based index of the channel that is used to measure the device. (only required if using pygor)
```
"save_dir":"save_file_name"
```
Relitive path to save tuning data in.
