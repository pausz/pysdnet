
from time import time
import string
import numpy as np
from cuda import *

"""
prototype 32x32 par space exploration
=====================================

- optimize axes' order for memory access
- optimize block/grid layouts

"""

### setup data and parameters

def main(save_data=False):

    n = 96

    tf = 100
    dt = 0.05
    ds = 10
    ts = np.r_[0:tf:dt]

    vel =  2.0
    gsc = ( 0, 30, 32j)
    exc = (-10, 10, 32j)

    idel = (np.random.uniform(low=3, high=160/vel, size=(n, n))/dt).astype(np.int32)
    hist = np.zeros((idel.max()+1, n, 1, 1024), dtype=np.float32)
    conn = np.random.normal(scale=0.1, size=(n, n)).astype(np.float32)
    X    = np.random.uniform(low=-0.1, high=0.1, size=(n, 1, 1024)).astype(np.float32)
    Xs   = np.empty((len(ts)/ds, n, 1, 1024), dtype=np.float32)

    print 'using %0.1f MB on GPU' % (sum(map(lambda a: a.nbytes, [idel, hist, conn, X]))/2**20, )

    # setup cuda kernel
    mod = srcmod('parsweep.cu', ['kernel', 'update'],
                 horizon=idel.max()+1, dt=dt, ds=ds, n=n,
                 gsc0=gsc[0], dgsc=(gsc[1]-gsc[0])/gsc[2].imag,
                 exc0=exc[0], dexc=(exc[1]-exc[0])/exc[2].imag)

    with arrays_on_gpu(idel=idel, hist=hist, conn=conn, X=X) as g:

        # step through simulation
        toc = 0.0
        for step, t in enumerate(ts):
            
            tic = time()

            mod.kernel(np.int32(step), g.idel, g.hist, g.conn, g.X, block=(32, 1, 1), grid=(32, 1))
            mod.update(np.int32(step), g.hist, g.X, block=(1024, 1, 1), grid=(1, 1))

            if step%ds == 0:
                Xs[step/ds, ...] = g.X.get()

            toc += time() - tic

    # normalize timing
    toc /= len(ts)

    # save data with parameter grids
    gsc, exc = np.mgrid[gsc[0]:gsc[1]:gsc[2], exc[0]:exc[1]:exc[2]]
    if save_data:
        np.savez('sim-data', ts, Xs, np.array([vel]), gsc, exc)

    print '%f ms / iteration' % (1000*toc, )

if __name__ == '__main__':
    main(save_data=True)
