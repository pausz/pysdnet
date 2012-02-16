"""
Simulating a stochastic delayed differential equation
=====================================================

- computationally complete draft in numpy here
- goal to have a pycuda version running faster

First

- write a python version that is mathematically identical
    that doesn't use NumPy abstractions

"""

from pylab import *

import numpy.random

reset_rng = lambda : numpy.random.seed(42)

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
        dx = (x - 5*x**3)/5 + k*np.sum(G*delstate, axis=0)/N

        # aligned memory access again
        # random number generator used
        hist[i%horizon,:] = x + dt*(dx+randn(N)/5)

    def __call__(self, N, tf=50, dt=0.2, k=5, delayscale=1):

        # initialize
        G = randn(N, N)
        x = randn(N)
        idelays = (delayscale*abs(randn(N, N))/dt).astype(int32)
        horizon = idelays.max()
        hist = zeros((horizon + 1, N))
        nids = tile(arange(N), (N, 1))
        xout = zeros((int(tf/dt), N))

        # step
        for i in xrange(1, int(tf/dt)):
            self.step(i, horizon, nids, idelays, dt, k, N, x, G, hist)
            xout[i, :] = hist[i % horizon, :]

        return xout

if __name__ == '__main__':

    from pylab import *
    import time

    for idx, k in enumerate(e**r_[-3:2:100j]):

        tic = time.time()
        dt = 0.2
        # k = 1.0
        tf = 1000
        ts = r_[0:tf:dt]
        smoothn = 10

        smooth = lambda sig: convolve(sig, ones(smoothn), 'same')/smoothn

        run = sdde2()

        figure(figsize=(10, 14))

        # without significant delay
        reset_rng()
        ys = run(30, tf=tf, dt=dt, delayscale=0.1, k=k)
        subplot(521)
        [plot(ts, y, 'k', alpha=0.1) for y in ys.T]
        subplot(523)
        ys0 = ys - ys.mean(axis=0)
        cv = cov(ys0.T)
        pcolor(cv)
        colorbar()
        subplot(525)
        es, ev = eig(cv)
        plot(es)
        subplot(527)
        pcas = dot(ev[:3, :], ys.T)
        [plot(ts, pc, alpha=0.5) for pc in pcas]
        subplot(529)
        freq = fftfreq(pcas.shape[1], d=dt/1000)
        [loglog(freq, smooth(abs(fft(pc))), alpha=0.5) for pc in pcas]
        xlim([0, freq.max()])

        # with delay
        reset_rng()
        ys = run(30, tf=tf, dt=dt, delayscale=50, k=k)
        subplot(522)
        [plot(ts, y, 'k', alpha=0.1) for y in ys.T]
        subplot(524)
        ys0 = ys - ys.mean(axis=0)
        cv = cov(ys0.T)
        pcolor(cv)
        colorbar()
        subplot(526)
        es, ev = eig(cv)
        plot(es)
        subplot(528)
        pcas = dot(ev[:3, :], ys.T)
        [plot(ts, pc, alpha=0.5) for pc in pcas]
        subplot(520)
        freq = fftfreq(pcas.shape[1], d=dt/1000)
        [loglog(freq, smooth(abs(fft(pc))), alpha=0.5) for pc in pcas]
        xlim([0, freq.max()])

        suptitle('k=%s' % (k,))
        savefig('compare%02d.png'%(idx,))
        close()
        print 'took ', time.time() - tic
