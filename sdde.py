"""
Simulating a stochastic delayed differential equation
=====================================================

- computationally complete draft in numpy here
- goal to have a pycuda version running faster


TODOs
-----

o  write pycuda wrapper
o  test against cpu versions
o  look at delay distributions
o  work out redundant partition scheme

block/grid setup
----------------


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

import ctypes
import os.path
from string import Template

try:
    import pyublas
    import pycuda.autoinit
    import pycuda.gpuarray as gary
    from pycuda.compiler import SourceModule as SrcMod
except:
    print "couldn't load PyCUDA libraries"


from pylab import *
import numpy.random
from numpy.ctypeslib import ndpointer

reset_rng = lambda : numpy.random.seed(42)
prep_array = lambda a: ascontiguousarray(a).ctypes.data

ndpcontig = lambda:ndpointer(flags='CONTIGUOUS,ALIGNED')

class c_step(object):
    """
    call into C to compute step

    static void sdde_step(int i, int horizon, int *nids, int *idelays, double dt,
        double k, int N, double *x, double *G, double *hist, double *randn)

    """

    cfunc = ctypes.CDLL('./csdde.so').sdde_step
    cfunc.restype = ctypes.c_voidp
    cfunc.argtypes = [ctypes.c_int, ctypes.c_int, ndpcontig(), ndpcontig(),
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ndpcontig(), ndpcontig(),
        ndpcontig(), ndpcontig()]

    def __call__(self, i, horizon, nids, idelays, dt, k, N, x, G, hist):
        self.cfunc(i, horizon, nids, idelays, dt, k, N, x, G, hist, randn(N))


class gpustep(object):

    kernel_pars = ['threadid', 'N', 'horizon', 'k', 'dt']

    with open('./gsdde.cu', 'r') as fd:
        kernel_src = Template(fd.read())

    # TODO rewrite for new kernel
    def __init__(self):

        # need to setup i, idelays, G, hist, randn

        # build module
        kernel_pars = {k:None for k in self.kernel_pars}
        self.mod = self.SrcMod(self.kernel_src.substitute(**kernel_pars))

        a = randn(n, n).astype(float32)
        d = abs(randn(n,n)/dt).astype(int32)
        hrzn = d.max()
        h = vstack((randn(n), zeros((hrzn,n))))
        mod = build_mod(n, hrzn, k=k, dt=dt)
        func = mod.get_function('step')
        ag, dg, hg = map(gary.to_gpu, [a, d, h])

        """
        for v in ['a', 'd', 'h', 'hrzn', 'mod', 'func', 'ag', 'dg', 'hg']:
            exec("self.%s = %s" % (v, v))
        """

    # TODO rewrite for new kernel
    def step(self, grid=(8, 1)):
        self.func(self.ag, self.hg, self.dg, block=(32,2,1), grid=grid)

    def get_state(self):
        self.get_state(self.i, self.gpu_hist, self.gpu_xout)
        self.xout = self.gpu_xout.get()
        return self.xout


class sdde2(object):
    """
    Here, we reformat the above simulation in a way that separates
    initialization from the integration step.

    - what can be compiled in kernel?
    - what needs to be passed in at each call?

    """

    def step(self, i, horizon, nids, idelays, dt, k, N, x, G, hist):
        """
        Perform single integration step

        i - current step number/count
        horizon - constant, size of maximumdelay
        nids - array of node ids used to index numpy
        idelays - array of delays, unit of integration step
        k - global coupling scale, constant
        N - num nodes, constant
        x - current state
        G - coupling matrix
        hist - history matrix, shape[0] = horizon + 1

        """

        # compute delayed state information, memory bandwidth hog
        delstate = hist[(i - 1 - idelays) % horizon, nids]

        # easy aligned memory access & copy, maybe use pointer
        x = hist[(i - 1) % horizon, :]

        # all math functions occur here + loop for G*delstate sum
        # k is constant for simulation
        dx = (x - 5*x**3)/5 + k*np.sum(G*delstate, axis=1)/N

        # aligned memory access again
        # random number generator used
        hist[i%horizon,:] = x + dt*(dx+ randn(N)/5)

    def __call__(self, N, tf=50, dt=0.2, k=0.01, delayscale=1, step_fn=None,
                    debug_out=False):

        # initialize
        G = randn(N, N)
        x = randn(N)
        idelays = (delayscale*abs(randn(N, N))/dt).astype(int32)
        horizon = idelays.max()
        hist = zeros((horizon + 1, N))
        nids = tile(arange(N), (N, 1))
        xout = zeros((int(tf/dt), N))

        if not step_fn:
            step_fn = self.step

        # step
        for i in xrange(1, int(tf/dt)):
            step_fn(i, horizon, nids, idelays, dt, k, N, x, G, hist)
            xout[i, :] = hist[i % horizon, :]

        if debug_out:
            return G, x, idelays, horizon, hist, nids, xout
        else:
            return xout

if __name__ == '__main__':

    from pylab import *
    import time

    # for idx, k in enumerate(e**r_[-3:2:100j]):

    dt = 0.02
    k = 4.2
    tf = 5000
    N = 50
    NFFT = 4096
    ts = r_[0:tf:dt]
    smoothn = 10

    smooth = lambda sig: convolve(sig, ones(smoothn), 'same')/smoothn

    run = sdde2()

    figure(figsize=(14, 14))

    # without significant delay
    tic = time.time()
    reset_rng()
    ys = run(N, tf=tf, dt=dt, delayscale=0.1, k=k)
    print 'numpy integration without delays took ', time.time() - tic
    subplot(531)
    [plot(ts, y, 'k', alpha=0.1) for y in ys.T]
    grid(1)
    subplot(534)
    ys0 = ys - ys.mean(axis=0)
    cv = cov(ys0.T)
    pcolor(cv)
    colorbar()
    subplot(537)
    es, ev = eig(cv)
    plot(es)
    grid(1)
    subplot(5,3,10)
    pcas = dot(ev[:3, :], ys.T)
    [plot(ts, pc, alpha=0.5) for pc in pcas]
    grid(1)
    subplot(5,3,13)
    freq = fftfreq(pcas.shape[1], d=dt/1000)
    #[loglog(freq, freq*smooth(abs(fft(pc))), alpha=0.5) for pc in pcas]
    specgram(pcas[0], NFFT, 1000.0/dt)
    ylim([0, 100])
    #grid(1)

    # with delay
    tic = time.time()
    reset_rng()
    ys = run(N, tf=tf, dt=dt, delayscale=50, k=k)
    print 'numpy integration with delays took ', time.time() - tic
    subplot(532)
    [plot(ts, y, 'k', alpha=0.1) for y in ys.T]
    grid(1)
    subplot(535)
    ys0 = ys - ys.mean(axis=0)
    cv = cov(ys0.T)
    pcolor(cv)
    colorbar()
    subplot(538)
    es, ev = eig(cv)
    plot(es)
    grid(1)
    subplot(5,3,11)
    pcas = dot(ev[:3, :], ys.T)
    [plot(ts, pc, alpha=0.5) for pc in pcas]
    grid(1)
    subplot(5,3,14)
    freq = fftfreq(pcas.shape[1], d=dt/1000)
    #[loglog(freq, freq*smooth(abs(fft(pc))), alpha=0.5) for pc in pcas]
    specgram(pcas[0], NFFT, 1000.0/dt)
    ylim([0, 100])
    #xlim([0, freq.max()])
    #grid(1)

    # with delay, C integrator
    tic = time.time()
    reset_rng()
    ys = run(N, tf=tf, dt=dt, delayscale=50, k=k, step_fn=c_step())
    print 'C integration with delays took ', time.time() - tic
    subplot(533)
    [plot(ts, y, 'k', alpha=0.1) for y in ys.T]
    grid(1)
    subplot(536)
    ys0 = ys - ys.mean(axis=0)
    cv = cov(ys0.T)
    pcolor(cv)
    colorbar()
    subplot(539)
    es, ev = eig(cv)
    plot(es)
    grid(1)
    subplot(5,3,12)
    pcas = dot(ev[:3, :], ys.T)
    [plot(ts, pc, alpha=0.5) for pc in pcas]
    grid(1)
    subplot(5,3,15)
    freq = fftfreq(pcas.shape[1], d=dt/1000)
    #[loglog(freq, freq*smooth(abs(fft(pc))), alpha=0.5) for pc in pcas]
    specgram(pcas[0], NFFT, 1000.0/dt)
    ylim([0, 100])
    #xlim([0, freq.max()])
    #grid(1)

    suptitle('k=%s' % (k,))
    #savefig('compare%02d.png'%(idx,))
    show()
    #close()
