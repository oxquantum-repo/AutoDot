import sys
import os
import numpy as np

from scipy import interpolate

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Last_score import final_score_cls

#save_dir = Path('./save_Dominic_okscore_randomHS_origin100_2')
save_dir = Path('./')

extra = np.load(save_dir/'extra_measure.npy', allow_pickle=True)

num_examples = len(extra)
print(num_examples)

len_all = [len(ext) for ext in extra]
print(np.unique(len_all))

#peaks = [len(ext) != 12 for ext in extra] # True, if there are peaks
peaks = [len(ext) != 11 for ext in extra] # True, if there are peaks

score_lowres = np.zeros(num_examples)
score_highres = np.zeros(num_examples)
#score_predict = np.zeros(num_examples)

score_object = final_score_cls(-1.8e-10,4.4e-10,5e-11,-1.4781e-10,150)

for i in range(num_examples):
    if peaks[i]:
        #score_lowres[i] = extra[i][16][1]
        #score_highres[i] = extra[i][17][1]
        #cu_map_low = extra[i][12]
        #cu_map_high = extra[i][18]
        cu_map_low = extra[i][11]
        cu_map_high = extra[i][17]

        #plt.imsave(str(save_dir/(str(i)+'lowres')), cu_map_low, origin='lower')
        #plt.imsave(str(save_dir/(str(i)+'highres')), cu_map_high, origin='lower')

        #print(cu_map_high.max(), cu_map_high.min())
        # compute low-res score
        diff = 1
        #score_lowres[i] = ok_score.og_2av_score(cu_map_low, diff=diff)
        score_lowres[i] = score_object.score(cu_map_low, diff=diff)


        # compute high-res score
        diff = 1/3
        #score_highres[i] = ok_score.og_2av_score(cu_map_high, diff=diff)
        score_highres[i] = score_object.score(cu_map_high, diff=diff)

        '''
        # using interpolated data
        h, w = cu_map_low.shape
        x, y  = np.arange(h), np.arange(w)
        f = interpolate.interp2d(x, y, cu_map_low, kind='cubic')

        h_, w_ = cu_map_high.shape
        x_, y_ = np.linspace(x.min(), x.max(), h_), np.linspace(y.min(), y.max(), w_)

        cu_map_predicted = f(x_, y_)
        score_predict[i] = ok_score.og_2av_score(cu_map_predicted, diff=diff)
        '''
    print(i)

idxs = np.arange(0,num_examples)
sort_idxs = np.argsort(-score_highres)

np.savetxt(str(save_dir/'result.csv'), [peaks, score_lowres, score_highres, idxs, sort_idxs], delimiter=',')

