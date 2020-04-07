# Algorithm overview
This file explains an animation of the algorithm so that it is easier to understand all it's componants. A '.mp4' animation containg 15 iterations of the algorithm (as detailed in our [paper](https://arxiv.org/abs/2001.02589)) in a 2D space can be found [here](movie.mp4). Below we explain in detail specific iterations.

## Initialisation
When the agorithm is first started it for the first fer iterations it performs initilisation. In the provided animation initilisation lasts 3 iterations and in the paper initilisation stage lasted 30 iterations.

For the first few iteration candidate unitvectors to investigate are selected uniformly in angle. The pinch off is then found by first moving to the origin. After moving to the origin the algorithm starts searching along the direction corrisponding to the selected unitvector. This search ends if the current falls below a specific threshold or if the bounding box is hit (in this case -2V). When the pinch off is detected a local search for colomb peaks is perfomed. After this is performed the algorithm starts pruning this is done by moving back a small amount (+100mV in this example) and searching parrallel to the gate axes. During this procedure information about whether these searches find pinch off's or if they just hit the bounding box are logged.
### Iteration 1
In the below example a candidate unitvector is selected and the algorithm performs a search along the direction from the origin. The search ends when the bounding box is hit (ie no pinch off is observed). A coulomb peak check is perfomed and no coulomb peaks are observed. The algorithm then performs pruning by first stepping back and then performing two searches parrallel to the two axes of the voltage space. The first of these traces finds a pinch off, however the second of these traces does not.
![](iteration1.gif)

### Iteration 2
At the begining of this iteration the origin is updated. Another candidate unitvector is selected and the algorithm performs a search along the direction from the updated origin. The search ends when a pinch off is identified. A coulomb peak check is perfomed and no coulomb peaks are observed. The algorithm then performs pruning by first stepping back and then performing two searches parrallel to the two axes of the voltage space. Both of these traces finds a pinch off.
![](iteration2.gif)

## Main sampling
When the initilisation is over the algorithm moves to main sampling this lasts for the ramaining amount of iterations which is 12 in this case (in the paper it was typically 470). 

A gaussian process model is used to predict the distance from the origin to the pinch off (or bounding box) given a candidate unit vector. This model effectivly aims to model the pinch off hypersurface. Given this hypersurface model a candidate unit vector is selected by simulating particles undergoing brownian motion within the predicted surface. Whenever one of these particles attempts to cross the surface the point where it crosses is added to a list of potential candidates. 

### Iteration 4
sg
![](iteration4.gif)



### Iteration 12
sdg
![](iteration12.gif)
