
from time import time
import string
import numpy as np
from cuda import *
import data.dsi

"""
prototype 32x32 par space exploration
=====================================

optimization
------------

- optimize axes' order for memory access
- optimize block/grid layouts

three-d full sweep
------------------

- vel = 2**r_[1:6:32j]

eventually
----------

systematic sweep over

- conduction velocity (1 m/s to 100 m/s)
- coupling scaling (0 to 1 (normalized)
- bifurcation parameter (per model type, generalized idea of excitability)
- bifurcation type (pitchfork, hopf, etc)
- connectivity skeleton

to create an catalog of whole-brain dynamics

"""

### setup data and parameters

def main(save_data=False, dataset_id='ay', vel=2.0, file_id=0, gsc=( 0, 3, 32j), exc=(-5, 5, 32j), meminfo=True,
         tf=200, dt=0.1, ds=10):

    dataset = data.dsi.load_dataset(dataset_id)

    n = dataset.weights.shape[0]
    ts = np.r_[0:tf:dt]

    idel = (dataset.distances/vel/dt).astype(np.int32)
    hist = np.zeros((idel.max()+1, n, 1, 1024), dtype=np.float32)
    conn = dataset.weights.astype(np.float32)
    X    = np.random.uniform(low=-1, high=1, size=(n, 1, 1024)).astype(np.float32)
    Xs   = np.empty((1+len(ts)/ds, n, 1, 1024), dtype=np.float32)

    # make sure first step is in otherwise zero'd history
    hist[-1, ...] = X

    if meminfo:
        print 'using %0.1f MB on GPU' % (sum(map(lambda a: a.nbytes, [idel, hist, conn, X]))/2**20, )

    # setup cuda kernel
    mod = srcmod('parsweep.cu', ['kernel', 'update'],
                 horizon=idel.max()+1, dt=dt, ds=ds, n=n, cvar=0,
                 gsc0=gsc[0], dgsc=(gsc[1]-gsc[0])/gsc[2].imag,
                 exc0=exc[0], dexc=(exc[1]-exc[0])/exc[2].imag)

    with arrays_on_gpu(idel=idel, hist=hist, conn=conn, X=X) as g:

        Xs[0, ...] = g.X.get()

        toc = 0.0
        for step, t in enumerate(ts):
            
            tic = time()

            mod.kernel(np.int32(step), g.idel, g.hist, g.conn, g.X, block=(32, 1, 1), grid=(32, 1))
            mod.update(np.int32(step), g.hist, g.X, block=(1024, 1, 1), grid=(1, 1))

            if step%ds == 0:
                Xs[1+step/ds, ...] = g.X.get()

            toc += time() - tic

    # normalize timing
    toc /= len(ts)

    # save data with parameter grids
    gsc, exc = np.mgrid[gsc[0]:gsc[1]:gsc[2], exc[0]:exc[1]:exc[2]]
    if save_data:
        if not type(save_data) in (str, unicode):
            save_data = 'sim-data'
        np.savez('%s-%s-%0.2f-%d' % (save_data, dataset_id, vel, file_id), 
                 ts=ts, Xs=Xs, vel=np.array([vel]), gsc=gsc, exc=exc)

    print '%f ms / iteration' % (1000*toc, )
    return Xs

if __name__ == '__main__':

    for i, v in enumerate(2**np.r_[1:6:32j]):
        for j in range(100):
            print i, j
            main(save_data='bistable/sim', vel=v, file_id=j, meminfo=False)
