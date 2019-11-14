from pathlib import Path
import numpy as np

def L2_norm(arr, axis=None, keepdims=False):
    return np.sqrt(np.sum(np.square(arr), axis=axis, keepdims=keepdims))

save_dir = Path('./save_Dominic_gpc2_run1')
idx_success = [136, 200, 233, 268, 373, 411, 455]

v = np.load(save_dir/'vols_poff.npy', allow_pickle=True)
v_success = v[idx_success]
diff = v_success[:,np.newaxis,:] - v_success[np.newaxis,...]
dist_mat = L2_norm(diff, axis=2)
np.set_printoptions(precision=2)
print(dist_mat)


print(np.all(np.fabs(diff)<=200, axis=2))
print(v_success)
