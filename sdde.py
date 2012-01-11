"""
Simulating a stochastic delayed differential equation

- computationally complete draft in numpy here
- goal to have a pycuda version running faster
"""

from pylab import *

def sdde(N, tf=50, dt=0.2):
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
        dx = - x + np.sum(A*delstate, axis=0)/N
        hist[i%horizon,:] = x + dt*(dx+randn(N)/5)
        ys[i,:] = hist[i%horizon,:]
    return ys

