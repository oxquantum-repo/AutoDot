import matplotlib.pyplot as plt

def plot_2d_map(img, fpath, extent, vmin=None, vmax=None, cmap='RdBu_r', colorbar=False, dpi=150):
    vmin = vmin or img.min()
    vmax = vmax or img.max()
    fig = plt.figure(dpi=dpi)
    ax_img = plt.imshow(img, cmap=cmap, extent=extent, origin='lower',vmin=vmin,vmax=vmax, aspect='equal')
    if colorbar:
        fig.colorbar(ax_img)
    plt.savefig(fpath)
    plt.close()
