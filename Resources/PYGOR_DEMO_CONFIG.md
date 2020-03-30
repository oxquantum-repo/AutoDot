# Config 
structure Here is an [example config file](demo_config.json) containing all the relevant fields and below is a discussion about each fields function 

``` "path_to_pygor":"/path_to/pygor/package" ``` 
Absolute path to pygor package. (only required if using pygor) 

``` "ip":"http://123.123.123.12:8000/RPC2" ``` 
URL of pygor server. If not specified mock device is built (only required if using pygor) 

``` "gates":["c10","c3","c4","c5","c6","c7","c9"] ``` 
Labels for the DAC channels that the gates are attached to. (only required if using pygor) 

``` "plunger_gates":["c5","c9"] ``` 
Labels for the DAC channels that the investigation/plunger gates are attached to. (only required if using pygor) 

``` "bias_chans":["c1"] ``` 
Label(s) for the DAC channel(s) that control the bias. (only required if using pygor) 

``` "bias_vals":[20] ``` 
Value(s) to set the bias DAC channel(s) to. (only required if using pygor) 

``` "chan_no":0 ``` 
0 based index of the channel that is used to measure the device. (only required if using pygor) 

``` "save_dir":"save_file_name" ``` 
Relative path to save tuning data in.
