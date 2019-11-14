from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_util

#save_dir = Path('./save0702')
save_dir = Path('./save_YutianFlorian')
#name = 'scanc5c9_0_512_lowbias'
name = 'scanc6c10_0_128'

data = np.load(str(save_dir/(name+'.npz')))
w_from = data['w_from']
w_to = data['w_to']
img = data['img'][0]


plot_util.plot_2d_map(img, str(save_dir/(name+'_cbar')), extent=[w_from[1], w_to[1], w_from[0], w_to[0]], vmin=img.min(), vmax=img.max(), colorbar=True, dpi=300)
