from pathlib import Path
import numpy as np

def plot(save_dir):
    ub_history = np.load(save_dir/'ub_history.npy', allow_pickle=True)
    ub = np.load(save_dir/'upperbound.npy', allow_pickle=True)
    changeidx = np.load(save_dir/'changeindex_history.npy', allow_pickle=True)
    print(ub)
    #print(changeidx)

    v = np.array(np.load(save_dir/'vols_poff.npy', allow_pickle=True))

    num_close_ub = np.sum(v[30:,0] > ub[0]-100.)
    print(num_close_ub, len(v-30))
    
'''
save_dirs = [Path('./save_Florian_gpc2_run1'),
            Path('./save_Florian_gpc2_run2'),
            Path('./save_Florian_gpc2_run3'),
            Path('./save_Florian_gpc2_run4'),
            Path('./save_Florian_gpc2_run5')]
'''
save_dirs = [Path('./save_Dominic_gpc2_run1'),
            Path('./save_Dominic_gpc2_run2'),
            Path('./save_Dominic_gpc2_run3'),
            Path('./save_Dominic_gpc2_run4'),
            Path('./save_Dominic_gpc2_run5'),
            Path('./save_Dominic_gpc2_run6'),
            Path('./save_Dominic_gpc2_run7'), 
            Path('./save_Dominic_gpc2_run8'), 
            Path('./save_Dominic_gpc2_run9') 
            ]


for save_dir in save_dirs:
    plot(save_dir)
    print('-----------------------')

