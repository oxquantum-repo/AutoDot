from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot(save_dir):
    origin = np.load(save_dir/'origin.npy')
    vals = np.load(save_dir/'vals.npy')

    #vals = vals[-1:]
    pinchoff_counter = 0
    plt.figure()
    for i in range(len(vals)):
        line = vals[i]
        plt.plot(np.arange(len(line)), line)
        if np.any(np.array(line) < 3e-10):
            pinchoff_counter += 1
    plt.savefig(str(save_dir/'lines'))
    plt.close()

    print('The number of pinched off lines: {} out of {}'.format(pinchoff_counter,len(vals)))

    r_all = np.load(save_dir/'dist_origin.npy')
    #print(r_all)
    plt.figure()
    plt.hist(r_all, bins=20)
    plt.savefig(str(save_dir/'hist_r'))
    plt.close()

    d_all = np.load(save_dir/'dist_surface.npy')
    d_all = np.array(d_all)
    d_tot = np.load(save_dir/'obj_val.npy')
    #print(d_tot)
    plt.figure()
    plt.hist(d_tot, bins=20)
    plt.savefig(str(save_dir/'hist_dtot'))
    plt.close()

    plt.figure()
    plt.plot(d_tot)
    plt.savefig(str(save_dir/'dtot'))
    plt.close()

    plt.figure(figsize=(4.8,10))
    for i in range(d_all.shape[1]):
        plt.subplot(d_all.shape[1],1,i+1)
        plt.plot(d_all[:,i])
    plt.savefig(str(save_dir/'dall'))
    plt.close()

    u_all = np.load(save_dir/'unit_vector.npy')
    plt.figure(figsize=(4.8,10))
    for i in range(u_all.shape[1]):
        plt.subplot(u_all.shape[1],1,i+1)
        plt.plot(u_all[:,i])
    plt.savefig(str(save_dir/'uall'))
    plt.close()


    v_all = u_all * r_all[:,np.newaxis] + origin[np.newaxis,:]
    plt.figure(figsize=(4.8,10))
    for i in range(v_all.shape[1]):
        plt.subplot(v_all.shape[1],1,i+1)
        plt.plot(v_all[:,i])
    plt.savefig(str(save_dir/'vall'))
    plt.close()

    #temp
    temp = d_all - v_all
    plt.figure(figsize=(4.8,10))
    for i in range(temp.shape[1]):
        plt.subplot(temp.shape[1],1,i+1)
        plt.plot(temp[:,i])
    plt.savefig(str(save_dir/'d_from0'))
    plt.close()

    invalid_counter = 0
    for i in range(len(d_all)):
        if np.any(np.array(d_all[i]) < 0.0):
            invalid_counter += 1
    print('The number of points where d cannot be measured: ', invalid_counter)

def main():
    #save_dir = Path('./save_Dominic_random')
    save_dir = Path('./save_Dominic_newscore3')
    #save_dir = Path('./save_YutianFlorian_edge')

    gate_names= ['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
    #gate_names = ['c3', 'c4', 'c8', 'c10', 'c11', 'c12', 'c16']
    num_gates = len(gate_names)

    #save_dir = Path('./save0702')

    #plot(save_dir)

    extra_all = np.load(save_dir/'extra_measure.npy')
    score_all = list()
    for i, extra in enumerate(extra_all):
        plt.figure()
        for g in range(num_gates):
            plt.plot(extra[g])
        ax = plt.gca()
        ax.legend(gate_names)
        plt.savefig(str(save_dir/'{}_line_to_hf'.format(i)))
        plt.close()

        score = extra[num_gates]
        print(score)
        score_all.append(score)

        '''
        last_line = len(extra)
        if score > 0. : last_line -= 1

        plt.figure()
        for j in range(num_gates+1, last_line):
            plt.plot(extra[j])
        plt.savefig(str(save_dir/'{}_morelines'.format(i)))
        plt.close()

        if score > 0.:
            fig=plt.figure()
            ax_img = plt.imshow(extra[-1][0], cmap='RdBu_r', origin='lower',vmin=None,vmax=None)
            fig.colorbar(ax_img)
            plt.savefig(str(save_dir/'{}_img'.format(i)))
            plt.close()
        '''

    #np.save(str(save_dir / 'extra_measure_'), extra_all)


    vols_all = np.load(save_dir / 'vols.npy')
    d_all = np.load(save_dir / 'dist_surface.npy')
    d_tot = np.array(np.load(save_dir/'obj_val.npy'))
    pinchoff_idx = np.load(save_dir / 'pinchoff_idx.npy')
    u_all = np.load(save_dir/'unit_vector.npy')
    vols_pinchoff_all = list()
    for i in range(len(vols_all)):
        vols_pinchoff_all.append(vols_all[i][pinchoff_idx[i]])

    np.savetxt(save_dir/'vols_pinchoff.csv', np.array(vols_pinchoff_all), delimiter=',', fmt='%4.2f')
    np.savetxt(save_dir/'u.csv', np.array(u_all), delimiter=',', fmt='%4.2f')
    np.savetxt(save_dir/'d.csv', d_all, delimiter=',', fmt='%4.2f')
    np.savetxt(save_dir/'dtot.csv', d_tot[:,np.newaxis], delimiter=',', fmt='%4.2f')

    np.savetxt(save_dir/'score.csv', np.array(score_all)[:,np.newaxis], delimiter=',', fmt='%4.2f')




    '''
    r_all = np.load(save_dir/'dist_origin.npy')
    u_all = np.load(save_dir/'unit_vector.npy')
    vols_pinchoff = u_all * r_all[:,np.newaxis]
    np.set_printoptions(precision=0)
    np.set_printoptions(suppress=True)
    '''

    '''
    d_all = np.load(save_dir/'dist_surface.npy')
    d_all = np.array(d_all)
    #print(d_all)

    summed = d_all - vols_pinchoff
    dull = np.sum(summed > 1980, axis=0)
    print(dull)
    '''

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main()


"""
# other info
other_info = np.load(save_dir / 'other_info.npy')
num_init = 40

def plot_all(lines, save_dir, name):
    (save_dir/name).mkdir(exist_ok=True)
    for i in range(len(lines)):
        line = lines[i]
        plt.figure()
        plt.plot(line)
        plt.savefig(str(save_dir/name/(name+'_'+str(i))))
        plt.close()

c5c9scan = other_info[0]
plot_all(c5c9scan, save_dir, 'c5c9')
c6c8scan = other_info[1]
plot_all(c6c8scan, save_dir, 'c6c8')
"""
