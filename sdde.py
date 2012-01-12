"""
Simulating a stochastic delayed differential equation
=====================================================

- computationally complete draft in numpy here
- goal to have a pycuda version running faster

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

"""
int i = threadIdx.i;
for (j=0;j<n;j++) sum += k*a[n*i + j]*h[n*d[n*i + j] + j];
xp1[i] = x[i] + dt*(x[i] - 5*pow(x[i],3))/5 + sum/n + q[i];
"""

# setup data in memory
# transfer to gpu
# execute loop
# copy back 

"""
based on programmin guide p171
and moderngpu.com

- reduce block size to hide memory latencies
- combine heavy memory and heavy arithmetics in same
  kernel because they can hide each other (?)

per mproc:

8 blocks of 32 * 2 threads -> 512 threads / mproc
-> 64 registers, 96 B sh 1 KB local mem / thread

this is conservative, play with num blocks and block
size based on memory performance

on quadro 600 - 2 mproc, 1024 threads - large region level
on m2050 - 14 mproc, 7168 threads - surface

remember: you can't loop inside kernel because semantics
of shared state not consistent! (but py fn calls are ~107 ns)

normally we'd partition neural space by delays, but if we 
overlap the domains (and use identical seeds for duplicate
nodes) we can achieve locality on both sides. 
"""

