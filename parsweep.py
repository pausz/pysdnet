
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

"""

### setup data and parameters

def main(save_data=False, dataset_id='ay', vel=2.0, file_id=0, gsc=( 0, 30, 32j), exc=(-10, 10, 32j)):

    dataset = data.dsi.load_dataset(dataset_id)

    n = dataset.weights.shape[0]

    tf = 100
    dt = 0.05
    ds = 10
    ts = np.r_[0:tf:dt]

    idel = (dataset.distances/vel/dt).astype(np.int32)
    hist = np.zeros((idel.max()+1, n, 1, 1024), dtype=np.float32)
    conn = dataset.weights.astype(np.float32)
    X    = np.random.uniform(low=-1, high=1, size=(n, 1, 1024)).astype(np.float32)
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
        if not type(save_data) in (str, unicode):
            save_data = 'sim-data'
        np.savez('%s-%s-%0.2f-%d' % (save_data, dataset_id, vel, file_id), 
                 ts=ts, Xs=Xs, vel=np.array([vel]), gsc=gsc, exc=exc)

    print '%f ms / iteration' % (1000*toc, )

if __name__ == '__main__':

    for i, v in enumerate(2**np.r_[1:6:32j]):
        for j in range(100):
            print i, j
            main(save_data='bistable/sim', vel=v, file_id=j)
