import numpy as np
import score_driver
from Last_score import final_score_cls
from pygor_fixvol import PygorRewire

def do_extra_meas(pg, vols, threshold):
    #gates = ["c3","c4","c5","c6","c7","c8","c9","c10"]
    #v_gates=["c5","c9"]
    #score_object = final_score_cls(-1.8e-10,4.4e-10,5e-11,-1.4781e-10,150)

    gates = ["c3","c4","c8","c10","c11","c12","c16"]
    v_gates=["c8","c12"]
    score_object = final_score_cls(-1.5667e-12,1.172867e-10,9.5849571e-12,4.604e-12,150)

    #pygor = pg.pygor
    #pygor.setvals(gates,vols.tolist())
    pg.set_params(vols.tolist())
    if isinstance(pg, PygorRewire):
        vols = np.array(pg.convert_to_raw(vols))

    score, measurements = score_driver.decision_aqu(pg.pygor, vols, gates, v_gates, score_object, decision_function=threshold, low_res=16, high_res=48)
    print('Score: ', score)

    return [score] + measurements
