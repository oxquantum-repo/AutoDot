#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:17:50 2020

@author: oxml
"""

from . import utils
import matplotlib.pyplot as plt

def show_dummy_device(device,configs,save=True):
    conf_g = configs['general']
    print("Plotting dummy enviroment")
    dev_vol = utils.extract_volume_from_mock_device(conf_g['lb_box'],conf_g['ub_box'],100,device)
    ax = utils.plot_volume(dev_vol,conf_g['lb_box'],conf_g['ub_box'],conf_g['ub_box'],100)
    
    if save: utils.rotate_save(ax,configs['save_dir']+'dummy_surf/')
    plt.ion()
    plt.show(block = False)
    plt.pause(10)
    
    
def show_gpr_gpc(gpr,configs,points,condidx,origin,gpc=None,save=True):
    conf_g = configs['general']
    print("Plotting gpr")
    gpr_vol = utils.extract_volume_from_gpr(conf_g['lb_box'],conf_g['ub_box'],100,gpr)
    ax = utils.plot_volume(gpr_vol,conf_g['lb_box'],conf_g['ub_box'],origin,100,cmap_func=gpc)
    
    ax = utils.plot_3d_scatter(points,condidx=condidx,ax=ax)
    
    if save: utils.rotate_save(ax,configs['save_dir']+'gpr_surf/')
    plt.ion()
    plt.show(block = False)
    plt.pause(10)


