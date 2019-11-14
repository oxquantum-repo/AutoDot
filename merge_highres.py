import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dir_list = [
'./save_Dominic_ABL_group_run1',
'./save_Dominic_ABL_fullmethod_run1',
'./save_Dominic_ABL_group2_fivegates',
'./save_Dominic_ABL_highres',
'./save_Dominic_gpc2_set2_run1',
'./save_Dominic_test_newpruning4_run2',
'./save_Dominic_test_newpruning_highres_always',
'./save_Dominic_gpc2_run1',
'./save_Dominic_gpc2_run6',
'./save_Dominic_gpc2_run7',
'./save_Dominic_gpc2_run8',
'./save_Dominic_gpc2_run9',
'./save_Dominic_gpc2_run2',
'./save_Dominic_gpc2_run3',
'./save_Dominic_gpc2_run4',
'./save_Dominic_gpc2_run5',
'./save_Florian_gpc2_run1',
'./save_Florian_gpc2_run2',
'./save_Florian_gpc2_run3',
'./save_Florian_gpc2_run4',
'./save_Florian_gpc2_run5',
'./save_Florian_gpc2_run7',
'./save_Florian_gpc2_run6',
'./save_Dominic_newpruning4_thcyclce2_run1',
'./save_Dominic_newpruning4_thcyclce2_run2',
'./save_Florian_newpruning4_run3',
'./save_Florian_newpruning4_run2',
'./save_Florian_newpruning4_run1',
'./save_Florian_newpruning',
'./save_Dominic_test_newpruning4_optparam_run2',
'./save_Dominic_test_newpruning4_optparam',
'./save_Dominic_test_newpruning4_highres_always',
'./save_Dominic_finalscore_randomHS_origin100_threshold05',
'./save_Dominic_finalscore_randomHS_origin100_threshold05_optparam',
'./save_Dominic_finalscore_randomHS_origin100_threshold05_optparam_run2'
]

dir_list = [Path(x) for x in dir_list]
highres_all = list()
lowres_all = list()
highres_good = list()
counter = 0
for d in dir_list:
    result_list = np.load(d/'result_collected.npy', allow_pickle=True)
    for result in result_list:
        if result[4] is not None:
            highres_all.append(result[4])
        if result[2] is not None:
            lowres_all.append(result[2])
    if not (d/'good').exists():
        print('No good indice set in ', str(d))
print(len(highres_all))
print(len(lowres_all))
sys.exit(1)

save_dir = Path('./highres')
save_dir.mkdir(exist_ok=True)
np.array(highres_all).dump(str(save_dir/'highres.npy'))
for i, ex in enumerate(highres_all):
    plt.imsave(str(save_dir/(str(i)+'highres.png')), ex, origin='lower', cmap='viridis')
