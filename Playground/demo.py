# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:24:44 2021

@author: thele
"""
from .mock_device import build_mock_device_with_json
from ..main_utils.utils import extract_volume_from_mock_device, extract_volume_from_gpr
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes_lewiner
from ..Investigation.Investigation_factory import Investigation_stage
from ..main_utils.utils import Timer

def create_mock_device_from_file(configs, pm = 60.0):

    device = build_mock_device_with_json(configs['playground'])
    
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
    
    def score(vol):
        inv = investigation_stage.do_extra_measure(vol,0.0,1.0, score_thresh=0.001)['extra_measure']
        if len(inv) >1:
            scorev = inv[1][0]
        else:
            scorev = 0.0
            
            
        uvec = vol/np.linalg.norm(vol)
        device.jump(vol + (uvec*pm))
        device_p_pinch = measure()
        device.jump(vol - (uvec*pm))
        device_m_pinch = measure()
        device.jump(vol)
        
        good = np.logical_xor(device_p_pinch, device_m_pinch)
            
            
        return scorev*good
    
    return device, jump, measure, check, score





def plot_device_demo(device, configs, cmap = 'winter'):
    conf_g = configs['general']
    dev_vol = extract_volume_from_mock_device(conf_g['lb_box'],conf_g['ub_box'],50,device)
    
    verts, faces, points_surf = get_surface(dev_vol,conf_g['lb_box'],conf_g['ub_box'],conf_g['ub_box'], 50)
    perm=[0,1,2]
    
    preds = np.array([0.5]*points_surf.shape[0])
    cmap = plt.get_cmap(cmap)
    c_preds = cmap(preds).squeeze()
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    
    surf = ax.plot_trisurf(verts[:, perm[0]], verts[:, perm[1]], faces, verts[:, perm[2]],
                           lw=0.1,edgecolor="black",alpha=0.5,vmin=0,vmax=1.)
    surf.set_facecolor(c_preds.tolist())

    ax.set_xlim([conf_g['ub_box'][0],conf_g['lb_box'][0]])
    ax.set_ylim([conf_g['ub_box'][1],conf_g['lb_box'][1]])
    ax.set_zlim([conf_g['ub_box'][2],conf_g['lb_box'][2]])
    
    ax.set_ylabel("Gate 1 / mV")
    ax.set_xlabel("Gate 2 / mV")
    ax.set_zlabel("Gate 3 / mV")
    plt.show()


def plot_gpr_demo(gpr, configs, origin = [0,0,0], obs = None, cmap = 'winter'):
    conf_g = configs['general']
    dev_vol = extract_volume_from_gpr(conf_g['lb_box'],conf_g['ub_box'],50,gpr)
    
    verts, faces, points_surf = get_surface(dev_vol,conf_g['lb_box'],conf_g['ub_box'],conf_g['ub_box'], 50)
    
    vol_origin = np.array(origin)
    vol_origin[[0,1,2]]=vol_origin[[1,0,2]]
    verts = (verts)+vol_origin
    
    
    
    perm=[0,1,2]
    
    
    if not isinstance(cmap, str):
        preds = cmap[1](points_surf)
    else:
        preds = np.array([0.5]*points_surf.shape[0])
        
    cmap = plt.get_cmap(cmap[0])
    c_preds = cmap(preds).squeeze()
    
        
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    alpha = 0.5
    if obs is not None:
        ax = plot_scatter(ax, *obs)
        alpha = 0.2
    
    surf = ax.plot_trisurf(verts[:, perm[0]], verts[:, perm[1]], faces, verts[:, perm[2]],
                           lw=0.1,edgecolor="black",alpha=alpha,vmin=0,vmax=1.)
    surf.set_facecolor(c_preds.tolist())

    ax.set_xlim([conf_g['ub_box'][0],conf_g['lb_box'][0]])
    ax.set_ylim([conf_g['ub_box'][1],conf_g['lb_box'][1]])
    ax.set_zlim([conf_g['ub_box'][2],conf_g['lb_box'][2]])
    
    ax.set_ylabel("Gate 1 / mV")
    ax.set_xlabel("Gate 2 / mV")
    ax.set_zlabel("Gate 3 / mV")
    plt.show()
    
    
def plot_scatter(ax, points, scores = None, cmap = 'inferno'):
    points=np.array(points)
    points[:,[0,1,2]] = points[:,[1,0,2]]
        
    if scores is not None:
        preds = scores
    else:
        preds = np.array([0.5]*points.shape[0])
        
    cmap = plt.get_cmap(cmap)
    c_preds = cmap(preds).squeeze()
        
    ax.scatter(*np.array(points).T, c = c_preds, s = 12)
    return ax
    
    
    
def manual_control_3d(jump, measure, configs, res = 30):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy.ma as ma
    from matplotlib.widgets import Slider
    
    data = ma.array(np.empty([res,res,res]), mask = np.ones([res,res,res]))
    data[0,0,0] = 1.0
    conf_g = configs['general']

    lb, ub = conf_g['lb_box'], conf_g['ub_box']
    grid = np.array(np.meshgrid(*[np.linspace(ub[i], lb[i], res) for i in range(3)]))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    fig.subplots_adjust(left=0.25, bottom=0.25)

    verts, c = voxel_polly(data)
    pc = Poly3DCollection(verts, facecolors = c)
    ax.add_collection3d(pc)


    ax.set_xlim([0.0,res])
    ax.set_xticks([0, res])
    ax.set_xticklabels([0, ub[0] - lb[0]])
    ax.set_xlabel('$\Delta$Gate 2 (mV)')
    
    ax.set_ylim([0.0,res])
    ax.set_yticks([0, res])
    ax.set_yticklabels([0, ub[1] - lb[1]])
    ax.set_ylabel('$\Delta$Gate 1 (mV)')
    
    ax.set_zlim([0.0,res])
    ax.set_zticks([0, res])
    ax.set_zticklabels([0, ub[2] - lb[2]])
    ax.set_zlabel('$\Delta$Gate 3 (mV)')
    
    
    gate1_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    gate1_slider = Slider(gate1_slider_ax, '$\Delta$Gate 1 (mV)', 0.0, ub[1] - lb[1], valinit=0.0)

    gate2_slider_ax  = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    gate2_slider = Slider(gate2_slider_ax, '$\Delta$Gate 2 (mV)', 0.0, ub[0] - lb[0], valinit=0.0)

    gate3_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    gate3_slider = Slider(gate3_slider_ax, '$\Delta$Gate 3 (mV)', 0.0, ub[2] - lb[2], valinit=0.0)
    
    def fill_array(val):
        '''
        Update function for plot
        '''
        idx = [(gate1_slider.val/(ub[1] - lb[1]))*res, 
               (gate2_slider.val/(ub[0] - lb[0]))*res, 
               (gate3_slider.val/(ub[2] - lb[2]))*res]
        idx_int = np.round(idx).astype(int)
        jump(grid[:,idx_int[1], idx_int[0], idx_int[2]])
        data[idx_int[1], idx_int[0], idx_int[2]] = np.squeeze(measure())
        verts, c = voxel_polly(data)
        pc.set_verts(verts)
        pc.set_facecolor(c)
        fig.canvas.draw_idle()
        
    gate1_slider.on_changed(fill_array)
    gate2_slider.on_changed(fill_array)
    gate3_slider.on_changed(fill_array)

    plt.show()
        




#==HELPER FUNCTIONS==

def get_surface(vols, lb_box, ub_box, origin, res):
    verts, faces, normals, values = marching_cubes_lewiner(vols,0)

    vol_origin = np.array(origin)
    vol_origin[[0,1,2]]=vol_origin[[1,0,2]]
    verts = (((verts)/res)*-2000)+vol_origin

    points_surf = verts[faces[:,0],:]
    points_surf[:,[0,1,2]] = points_surf[:,[1,0,2]]
    return verts, faces, points_surf



def cuboid_data(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def voxel_polly(cube):
    locs = np.array(np.where(~cube.mask)).T
    col = cube[~cube.mask]
    col_a = np.zeros([locs.shape[0], 4])
    col_a[col == 0.0] = [0.0, 0.1, 1.0, 0.1]
    col_a[col == 1.0] = [1.0, 0.1, 0.1, 1.0]
    
    
    g = []
    for loc in locs:
        g.append( cuboid_data(loc))
    return np.concatenate(g), np.repeat(col_a,6, axis=0)
