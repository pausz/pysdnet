"""
Simulating a stochastic delayed differential equation
=====================================================

- computationally complete draft in numpy here
- goal to have a pycuda version running faster

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

import pycuda.autoinit
import pycuda.gpuarray as gary
from pycuda.compiler import SourceModule as SrcMod
from string import Template as STmp

# assume grid=(8,1), block=(2,32)
kernel_src = STmp("""
__global__ void step(//int pos, 
                     float * __restrict__ a, 
                     float * __restrict__ h, 
                     int * __restrict__ d) {

    int pos = 0;
    int j, i = blockIdx.x*blockDim.x + 2*threadIdx.x + threadIdx.y;
    float sum = 0.0, 
        x = h[ $n*(pos%$hrzn) + i],
        q = 0.0;

    for (pos=0;pos<10;pos++) {        

        for (j=0;j<$n;j++)
            sum += $k*a[$n*i + j]*h[$n*((pos - 1 - d[$n*i + j])%$hrzn) + j];
        
        x += $dt*(0.2*x - __powf(x, 3)) + sum*$rn + q;
        __syncthreads();
        h[$n*((pos+1)%$hrzn) + i] = x;

    }
}
""")

def build_mod(n, hrzn, k=1.0, dt=0.1):
    code = kernel_src.substitute(n=n, rn=1.0/n, hrzn=hrzn, k=k, dt=dt)
    return SrcMod(code)

class gpsdde:

    def __init__(self, n, k=5.0, dt=0.1):

        a = randn(n, n).astype(float32)
        d = abs(randn(n,n)/dt).astype(int32)
        hrzn = d.max()
        h = vstack((randn(n), zeros((hrzn,n))))
        mod = build_mod(n, hrzn, k=k, dt=dt)
        func = mod.get_function('step')
        ag, dg, hg = map(gary.to_gpu, [a, d, h])

        for v in ['a', 'd', 'h', 'hrzn', 'mod', 'func', 'ag', 'dg', 'hg']:
            exec("self.%s = %s" % (v, v))

    def step(self, grid=(8, 1)):
        self.func(self.ag, self.hg, self.dg, block=(32,2,1), grid=grid)

