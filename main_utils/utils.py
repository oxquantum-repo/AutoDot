# 20200304
# Brandon Severin

# Brief Info 
# Logs and stores the times of steps during the tuning algorithm


# Import modules
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from pathlib import Path



def plot_conditional_idx_improvment(cond_idx, config,cond_count=2,save=True):
    plt.figure()
    found = (np.array(cond_idx)>=cond_count).astype(np.float)
    
    count = [None]*len(found)
    for i in range(len(found)):
        count[i] = np.average(found[:i])
        
    plt.plot(np.arange(len(count)),count)
    plt.xlabel("Iteration")
    plt.ylabel("Empirical probability of conditional idx reaching %i"%cond_count)
    
    if save: plt.savefig(config['save_dir']+'improvment.png')

    plt.ion()
    plt.show()



def extract_volume_from_gpr(lb,ub,res,gpr):
    
    
    axes_linspace = [np.linspace(ub[i],lb[i],res) for i in range(len(lb))]
    
    X = np.array(np.meshgrid(*axes_linspace)).reshape([3,-1])
    X = np.swapaxes(X,0,1)
    X_r = np.linalg.norm(X,axis=1)[...,np.newaxis]
    U = X/X_r

    r_est , sig = gpr.predict(U)
    Y = X_r<=r_est

    Y = Y.reshape([res]*len(axes_linspace))

    return Y


def extract_volume_from_mock_device(lb,ub,res,device):
    
    
    axes_linspace = [np.linspace(ub[i],lb[i],res) for i in range(len(lb))]
    
    X = np.array(np.meshgrid(*axes_linspace)).reshape([len(lb),-1])
    X = np.swapaxes(X,0,1)
    Y = device.arr_measure(X)
    Y = Y.reshape([res]*len(axes_linspace))

    return Y


def plot_volume(vols,plot_lb,plot_ub,vol_origin,res,cmap_func=None,cmap='winter',perm=[0,1,2],ax=None):
    verts, faces, normals, values = measure.marching_cubes_lewiner(vols,0)

    vol_origin = np.array(vol_origin)
    vol_origin[[0,1,2]]=vol_origin[[1,0,2]]
    verts = (((verts)/res)*-2000)+vol_origin

    points_surf = verts[faces[:,0],:]
    points_surf[:,[0,1,2]] = points_surf[:,[1,0,2]]

    
    #preds = sampler.gpc.predict_comb_prob(points_surf)
    preds = cmap_func(points_surf) if cmap_func is not None else np.array([0.5]*points_surf.shape[0])
    cmap = plt.get_cmap(cmap)
    c_preds = cmap(preds).squeeze()
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.view_init(azim=0, elev=15)

    
    #if points is not None: ax.scatter(*points,c=condidx,zorder=1)

    surf = ax.plot_trisurf(verts[:, perm[0]], verts[:, perm[1]], faces, verts[:, perm[2]],
                           lw=0.1,edgecolor="black",alpha=0.5,vmin=0,vmax=1.)
    surf.set_facecolor(c_preds.tolist())

    ax.set_xlim([plot_ub[0],plot_lb[0]])
    ax.set_ylim([plot_ub[1],plot_lb[1]])
    ax.set_zlim([plot_ub[2],plot_lb[2]])
    
    ax.set_ylabel("Gate 1 / mV")
    ax.set_xlabel("Gate 2 / mV")
    ax.set_zlabel("Gate 3 / mV")
    return ax

def plot_3d_scatter(points,condidx=None,condidx_max=3,cmap='plasma',ax=None):
    points=np.array(points)
    points[:,[0,1,2]] = points[:,[1,0,2]]

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.view_init(azim=0, elev=15)
        
    if condidx is not None:
        ax.scatter(*points.T,c=condidx)
    else:
        ax.scatter(*points.T)
        
    return ax

def rotate_save(ax,path,step=9):
    import imageio
    
    Path(path).mkdir(parents=True,exist_ok=True)
    
    fnames = []
    for i in range(0,360,step):

        ax.view_init(azim=i, elev=15)
        fnames += [path+'/%i.png'%i]
        plt.savefig(fnames[-1])
        
    images = []   
    for fname in fnames:
        images.append(imageio.imread(fname))
    imageio.mimsave(path+'/movie.gif', images)




# Define Timer Error class
class TimerError(Exception): 
    """ A custom exception used to report errors in use of Timer class"""
        
# Define Timer class
class Timer():
    
    def __init__(self, verbose=False , elapsed_time_text="Elapsed time: {:0.4f} seconds", runtime_text="Current runtime: {:0.4f} seconds"):
        # Initialise <._start_time> attrtibute
        self._start_time = None
        self.verbose = verbose
        # create format: default text to report current runtime for logging and elapsed time upon stopping timer
        self.runtime_text = runtime_text
        self.elapsed_time_text = elapsed_time_text
        # initiate times list
        self.times_list = []

    # start the timer
    def start(self):
        # Check if the timer is already running
        if self._start_time is not None:
            raise TimerError(f"Timer is running, Use .stop() to stop it")
        # start timer
        self._start_time = time.perf_counter()
        # Append an empty list (for storing logtimes) to times list
        self.times_list.append([])
    
    # Log time elapsed, keep timer running
    def logtime(self):
        # First, check if the timer is running
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it.")

        # calculate current runtime
        runtime = time.perf_counter() - self._start_time
        # Append runtime to list within times_list
        self.times_list[-1].append(runtime)

        # print current runtime to console by first formatting text template
        if self.verbose: print(self.runtime_text.format(runtime))
    
    # stop the timer
    def stop(self):
        # stop timer, and report the elapsed time
        # First, check if the timer is running
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it.")

        # calculate total elapsed time 
        elapsed_time = time.perf_counter() - self._start_time
        
        # Append elapsed time to list within times_list
        self.times_list[-1].append(elapsed_time)
        
        # reset start time back to 'None' so that the timer can be restarted
        self._start_time = None
       
        # print elapsed time to console by first formatting text template
        if self.verbose: print(self.elapsed_time_text.format(elapsed_time))

