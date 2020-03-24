**Note running this demo with plot field set to true (like it is in the demo) imageio is required as an additional dependancy.**
## Before running
Provided already is a [demo config](../mock_device_demo_config.json) json that will run 50 iterations of the algorithm as used in the paper with mock versions of the investigation stage steps in the 3D enviroment plotted below. The enviroment is specified to be 3D using the "ndim" field and the shape of the enviroment is specified using primitive shapes that are defined in [shapes.py](shapes.py). The example used in the demo combines a standard Crosstalk_box and Leakage shape (note leakage does not refer to leaky gates but refers to alteritive undesired current pathways from source to drain). A 3D Crosstalk_box is the type of shape one should expect to observe for three barrier gates. A Leakage demonstrates the pruning method and would be expected if one gate was required to define the current path (like a large top gate or V_1 in the original paper).

If the "plot" flag in "playground" is set to True the algorithm will first attempt to treat the enviroment as a 3D space and extract an isosurface (note this could not be done on a real device due to the time required to measure such a plot):



![](demo_run_data/color_comp_dummy.gif)


## During running
During running the algorithm will print many outputs. An example output for an iteration is:
```
============### ITERATION 29 ###============
GPR: True GPC: True prune: True GPR1: True GPC1: True Optim: False
START
True
0.19766634805822517
Score thresh:  0.36156802062375365
STOP
dvec pinches:  [ True  True  True]
There are 30 training examples for model 0 and 26 are positive
There are 26 training examples for model 1 and 17 are positive
conditional_idx: 2
vols_pinchoff: [-1146.48524481  -813.5532      -647.01667969]
detected: True
r_vals: 1285.0125867654601
```
Below is a breakdown of what each line means:
```
GPR: True GPC: True prune: True GPR1: True GPC1: True Optim: False
```
Denotes whether the gpr training, gpc training, gpr selection, gpc selection and gp optimisation is running for a given iteration

```
START
```
Brownian motion sampling has started running in parralell with algorithm

```
True
```
Result of mock_peak_check (only printed if "verbose":true)

```
0.19766634805822517
```
Result of mock_score_func.

```
Score thresh:  0.36156802062375365
```
Current value of the score threshold.

```
STOP
```
Brownian motion sampling has stopped running in parralell with algorithm

```
dvec pinches:  [ True  True  True]
```
If pruning is active this shows which gates are showing a pinch off

```
There are 30 training examples for model 0 and 26 are positive
There are 26 training examples for model 1 and 17 are positive
```
Shows how many positives and training examples there are for the conditional gpc models (in the paper model 0 models if a pinch off will be observed and model 1 if a coulomb peak will be observed)

```
conditional_idx: 2
```
Denotes how many stages in the investigation stage were passed. in the paper 0: no pinch, 1: no colomb peak, 2: colomb peak but poor regime, 3: nice regime. (NOTE only printed if "conditional_idx" included in top level "verbose")

```
vols_pinchoff: [-1146.48524481  -813.5532      -647.01667969]
```
Point of pinch off in voltage space. (NOTE only printed if "vols_pinchoff" included in top level "verbose")

```
detected: True
```
If pinch off was observed. (NOTE only printed if "detected" included in top level "verbose")

```
r_vals: 1285.0125867654601
```
Distance from origin to the pinch off




## After running
All results are saved in the save directory ("save_dir") in a file called tuning.pkl. The saved fields are specified in "track". Useful fields that you might want to look at are "vols_pinchoff" (list of all pinch-off's or points where the algorithm hit the boundary), "detected" (if the point is a pinch-off or boundary point), "conditional_idx" (denotes how sucessful a given iteration was at passing charicterisation tests), "r_vals" (distance from origin to the pinch-off/boundary), and "extra_measure" (contains list of dicts that contain all information gathered by the investigation stage including stability diagrams if running on a real device). 

If the "plot" flag in "playground" is set to True the algorithm will finally attempt to treat fit gausian process prediction of the hypersurface (gpr) as a 3D space and extract an isosurface. It will also plot the gausian process classification (gpc) as the colour map of the gpr. The point used for training of these models will also be plotted:

![](demo_run_data/gpr_and_gpc.gif)

Finally the improvement of the empirical probability of observing the situation that the gpc models is plotted as well:
![](demo_run_data/improvment.png)
