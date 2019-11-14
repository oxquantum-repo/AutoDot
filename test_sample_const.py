import numpy as np

def main():
    # Create a sample function
    target_f = lambda x : (x-0.2)*(x-0.5)*(x-0.9)
    lb, ub = np.array([0.]), np.array([10.])
    assert len(lb) == len(ub)
    ndim = len(lb)

    # Run the algorithm
    num_init_samples = 3
    num_samples = 30

    x_init = (ub-lb)[np.newaxis,:]*np.random.rand(num_init_samples,ndim) + lb[np.newaxis,:]
    print(x_init)


if __name__ == '__main__':
   main()
