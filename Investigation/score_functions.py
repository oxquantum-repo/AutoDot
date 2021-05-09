# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:30:00 2021

@author: kamp
"""
import numpy as np

def score_nothing(invest_em_results, config):
    return 0


def mock_count_stages(invest_results, config):
    return -invest_results['conditional_idx']


def mock_stage_score(invest_resut, config):
    if invest_resut['conditional_idx'] < config.get("stage"):
        return np.inf
    else:
        return -invest_resut['extra_measure'][config.get("stage") - 1][0]
