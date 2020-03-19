# 20200304
# Brandon Severin

# Brief Info 
# Logs and stores the times of steps during the tuning algorithm


# Import modules
import time
import numpy as np
import matplotlib.pyplot as plt



def plot_conditional_idx_improvment(cond_idx,cond_count=2):
    
    found = (np.array(cond_idx)>=cond_count).astype(np.float)
    
    count = [None]*len(found)
    for i in range(len(found)):
        count[i] = np.average(found[:i])
        
    plt.plot(count)
    plt.show()






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

