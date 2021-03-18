# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:30:00 2021

@author: kamp
"""

def score_nothing(invest_em_results):
    return 0

def mock_count_stages(invest_results):
    return -invest_results['conditional_idx']

def mock_peak_score(invest_resut):
    if invest_resut['conditional_idx'] < 2:
        return 0
    else:
        return -invest_resut['extra_measure'][1][0]
