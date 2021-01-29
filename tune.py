# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:10:26 2019

@author: thele
"""
import sys
import json
from .Sampler_factory import Paper_sampler, Redo_sampler
from .Investigation.Investigation_factory import Investigation_stage
from .main_utils.utils import Timer, plot_conditional_idx_improvment
from .main_utils.model_surf_utils import show_gpr_gpc, show_dummy_device
from .Playground.mock_device import build_mock_device_with_json






def tune_with_pygor_from_file(config_file):
    
    with open(config_file) as f:
        configs = json.load(f)
        
    pygor_path = configs.get('path_to_pygor',None)
    if pygor_path is not None:
        sys.path.insert(0,pygor_path)
    import Pygor
    pygor = Pygor.Experiment(xmlip=configs.get('ip',None))
        
    gates = configs['gates']
    plunger_gates = configs['plunger_gates']
    
    chan_no = configs['chan_no']
    
    grouped = any(isinstance(i, list) for i in gates)
    
    if grouped:
        def jump(params,plungers=False):
            
            if plungers:
                labels = plunger_gates
            else:
                labels = gates
            
            for i,gate_group in enumerate(labels):
                pygor.setvals(gate_group,[params[i]]*len(gate_group))
                
            return params
    else:
        def jump(params,plungers=False):
            #print(params)
            if plungers:
                labels = plunger_gates
            else:
                labels = gates
            pygor.setvals(labels,params)
            return params
    def measure():
        cvl = pygor.do0d()[chan_no][0]
        return cvl
    def check():
        return pygor.getvals(plunger_gates)
    
    assert len(gates) == len(configs['general']['origin'])
        
    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump,measure,check,configs['investigation'],inv_timer)
        
    return tune(jump,measure,investigation_stage,configs)
    
    
    
    
    
def tune_with_playground_from_file(config_file):
    
    with open(config_file) as f:
        configs = json.load(f)
        
        
        
    device = build_mock_device_with_json(configs['playground'])
    
    if configs['playground'].get('plot',False): show_dummy_device(device,configs)
    
    plunger_gates = configs['plunger_gates']
    
    
    def jump(params,inv=False):
        if inv:
            return params
        else:
            return device.jump(params)
    
    
    measure = device.measure
    
    check = lambda: device.check(plunger_gates)
    
    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump,measure,check,configs['investigation'],inv_timer)
        
    results,sampler = tune(jump,measure,investigation_stage,configs)
    
    fields = ['vols_pinchoff','conditional_idx','origin']
    
    if configs['playground'].get('plot',False): 
        show_gpr_gpc(sampler.gpr, configs, *sampler.t.get(*fields), gpc=sampler.gpc.predict_comb_prob)
        plot_conditional_idx_improvment(sampler.t['conditional_idx'],configs)
    
    return results,sampler


def tune_with_playground_from_file(config_file):
    
    with open(config_file) as f:
        configs = json.load(f)
        
        
        
    device = build_mock_device_with_json(configs['playground'])
    
    if configs['playground'].get('plot',False): show_dummy_device(device,configs)
    
    plunger_gates = configs['plunger_gates']
    
    
    def jump(params,inv=False):
        if inv:
            return params
        else:
            return device.jump(params)
    
    
    measure = device.measure
    
    check = lambda: device.check(plunger_gates)
    
    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump,measure,check,configs['investigation'],inv_timer)
        
    results,sampler = tune(jump,measure,investigation_stage,configs)
    
    fields = ['vols_pinchoff','conditional_idx','origin']
    
    if configs['playground'].get('plot',False): 
        show_gpr_gpc(sampler.gpr, configs, *sampler.t.get(*fields), gpc=sampler.gpc.predict_comb_prob)
        plot_conditional_idx_improvment(sampler.t['conditional_idx'],configs)
    
    return results,sampler




def redo_with_pygor_from_file(config_file, pointcloud):
    with open(config_file) as f:
        configs = json.load(f)
        
    pygor_path = configs.get('path_to_pygor',None)
    if pygor_path is not None:
        sys.path.insert(0,pygor_path)
    import Pygor
    pygor = Pygor.Experiment(xmlip=configs.get('ip',None))
        
    gates = configs['gates']
    plunger_gates = configs['plunger_gates']
    
    chan_no = configs['chan_no']
    
    grouped = any(isinstance(i, list) for i in gates)
    
    if grouped:
        def jump(params,plungers=False):
            
            if plungers:
                labels = plunger_gates
            else:
                labels = gates
            
            for i,gate_group in enumerate(labels):
                pygor.setvals(gate_group,[params[i]]*len(gate_group))
                
            return params
    else:
        def jump(params,plungers=False):
            #print(params)
            if plungers:
                labels = plunger_gates
            else:
                labels = gates
            pygor.setvals(labels,params)
            return params
    def measure():
        cvl = pygor.do0d()[chan_no][0]
        return cvl
    def check():
        return pygor.getvals(plunger_gates)
    
    assert len(gates) == len(configs['general']['origin'])
        
    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump,measure,check,configs['investigation'],inv_timer)
        
    return redo(jump,measure,investigation_stage,configs)






def redo_with_playground_from_file(config_file,pointcloud):
    
    with open(config_file) as f:
        configs = json.load(f)
        
        
        
    device = build_mock_device_with_json(configs['playground'])
    
    if configs['playground'].get('plot',False): show_dummy_device(device,configs)
    
    plunger_gates = configs['plunger_gates']
    
    
    def jump(params,inv=False):
        if inv:
            return params
        else:
            return device.jump(params)
    
    
    measure = device.measure
    
    check = lambda: device.check(plunger_gates)
    
    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump,measure,check,configs['investigation'],inv_timer)
        
    results,sampler = redo(jump,measure,investigation_stage,configs,pointcloud)
    
    fields = ['vols_pinchoff','conditional_idx','origin']
    
    if configs['playground'].get('plot',False): 
        show_gpr_gpc(sampler.gpr, configs, *sampler.t.get(*fields), gpc=sampler.gpc.predict_comb_prob)
        plot_conditional_idx_improvment(sampler.t['conditional_idx'],configs)
    
    return results,sampler


def tune_from_dict(jump,measure,check,configs):
    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump,measure,check,configs['investigation'],inv_timer)
    results,sampler = tune(jump,measure,investigation_stage,configs)
    return results,sampler
    

def tune_from_file(jump,measure,check,config_file):
    with open(config_file) as f:
        configs = json.load(f)
        
    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump,measure,check,configs['investigation'],inv_timer)
    results,sampler = tune(jump,measure,investigation_stage,configs)
    return results,sampler
    
    
    
    
def redo_from_file(jump,measure,check,config_file,pointcloud):
    with open(config_file) as f:
        configs = json.load(f)
        
    inv_timer = Timer()
    investigation_stage = Investigation_stage(jump,measure,check,configs['investigation'],inv_timer)
    results,sampler = redo(jump,measure,investigation_stage,configs,pointcloud)
    return results,sampler
    

    

def tune(jump,measure,investigation_stage,configs):
    configs['jump'] = jump
    configs['measure'] = measure
    configs['investigation_stage_class'] = investigation_stage
    ps = Paper_sampler(configs)
    for i in range(configs['general']['num_samples']):
        print("============### ITERATION %i ###============"%i)
        results = ps.do_iter()
        for key,item in results.items():
            print("%s:"%(key),item[-1])
            
    return results, ps
            
            
def tune_origin_variable(jump,measure,par_invstage,child_invstage,par_configs,child_configs):

    par_configs['jump'],child_configs['jump'] = jump,jump
    par_configs['measure'],child_configs['measure'] = measure,measure
    par_configs['investigation_stage_class'], child_invstage['investigation_stage_class'] = par_invstage, child_invstage
    par_ps = Paper_sampler(par_configs)
    child_ps_list = []
    
    ps = par_ps
    par_flag = True
    for i in range(par_configs['general']['num_samples']):
        print("============### ITERATION %i ###============"%i)
        results = ps.do_iter()
        for key,item in results.items():
            print("%s:"%(key),item[-1])
        if par_flag:
            if new_origin_condition(ps):
                child_configs_new = config_constructor(child_configs,results)
                child_ps_list+=[Paper_sampler(child_configs_new)]
            
        ps,par_flag = task_selector(par_ps,child_ps_list,i)
        
        
        
def redo(jump, measure, investigation_stage, configs, pointcloud):
    configs['jump'] = jump
    configs['measure'] = measure
    configs['investigation_stage_class'] = investigation_stage
    ps = Redo_sampler(configs, pointcloud)
    for i in range(configs['general']['num_samples']):
        print("============### ITERATION %i ###============"%i)
        results = ps.do_iter()
        for key,item in results.items():
            print("%s:"%(key),item[-1])
            
    return results, ps
    
    
            
    
if __name__ == '__main__':
   pass
   #tune_with_pygor_from_file('tuning_config.json') 
   
        
        
