"""
Simulating a stochastic delayed differential equation
=====================================================

- computationally complete draft in numpy here
- goal to have a pycuda version running faster

elements
--------

noise 
``````

http://documen.tician.de/pycuda/array.html#module-pycuda.curandom

PyCuda has built-ins for generating noise, so this could be done 
either for single steps or multiple steps at once

delays
``````

requires same indexing approach but maybe we'll be able to improve
over basic approach by being clever. the distribution of delays 
could be relevant for some batch-partition scheme.

sparse coupling matrix
``````````````````````
 
http://pycuda.2962900.n2.nabble.com/PyCUDA-pycuda-sparse-td5540777.html

Less well supported by PyCUDA, but still there, are sparse matrix 
dense vector multiply (as well as sparse systems), so this will 
take care of the challenge of surface simulations over region level. 

"""

from pylab import *

def sdde(N, tf=50, dt=0.2, k=5):
    A = randn(N, N)
    X = randn(N)
    L = (abs(randn(N, N))/dt).astype(int32)
    horizon = L.max()
    hist = zeros((horizon+1, N))
    nids = np.tile(np.arange(N), (N, 1))
    ys = zeros((int(tf/dt), N))
    for i in xrange(1, int(tf/dt)):
        delstate = hist[(i-1-L)%horizon, nids]
        x = hist[(i-1)%horizon,:]
        dx = (x - 5*x**3)/5 + k*np.sum(A*delstate, axis=0)/N
        hist[i%horizon,:] = x + dt*(dx+randn(N)/5)
        ys[i,:] = hist[i%horizon,:]
    return ys

# setup data in memory
# transfer to gpu
# execute loop
# copy back 

