import string, time, sys, data.dsi, multiprocessing, gc, util, itertools, os
from cuda import estnthr, srcmod, arrays_on_gpu
from pylab import *
from numpy import *

model_nsvs = dict(fhn_euler=2, bistable_euler=1)

def gpu(gsc, exc, vel, dt, dataset, tf=1500, ds=80, model="fhn_euler", cvar=0,
        kblock=128, ublock=1024, cat=concatenate):

    ts      = r_[0:tf:dt]
    n       = dataset.weights.shape[0]
    nsv     = model_nsvs[model]
    nthr    = len(gsc)
    npad    = nthr % ublock if nthr%kblock or nthr%ublock and nthr > ublock else 0
    nthr   += npad
    idel    = (dataset.distances/vel/dt).astype(int32)
    hist    = random.uniform(low=-1., high=1., size=(idel.max()+1, n, nthr))
    conn    = dataset.weights
    X       = random.uniform(low=-2, high=2, size=(n, nsv, nthr))
    Xs      = empty((1+len(ts)/ds, n, nsv, nthr), dtype=float32)

    hist[-1, ...] = X[:, cvar, :]

    mod = srcmod('parsweep.cu', ['kernel', 'update'], 
                 horizon=idel.max()+1, dt=dt, ds=ds, n=n, cvar=cvar, model=model, nsv=nsv)


    with arrays_on_gpu(_timed=False, _memdebug=True, idel=idel.astype(int32), 
                       hist=hist.astype(float32), conn=conn.astype(float32), 
                       X=X.astype(float32), exc=cat((exc, zeros((npad,)))).astype(float32), 
                       gsc=cat((gsc, zeros((npad,)))).astype(float32)) as g:

        Xs[0, ...] = g.X.get()

        for step, t in enumerate(ts):
            mod.kernel(int32(step), g.idel, g.hist, g.conn, g.X, g.gsc, g.exc,
                       block=(kblock, 1, 1), grid=(nthr/kblock, 1))
            mod.update(int32(step), g.hist, g.X, 
                       block=(ublock if nthr>=ublock else nthr, 1, 1), 
                       grid=(nthr/ublock if nthr/ublock > 0 else 1, 1))

            if step%ds == 0 and not (1+step/ds)>=len(Xs):
                Xs[1+step/ds, ...] = g.X.get()

    Xs = rollaxis(Xs, 3)
    return Xs[:-npad] if npad else Xs


def launches(datasets, models, vel, gsc, exc, nic, dt, blockDim_x=256, gridDim_x_full=256):

    for i, cfg in enumerate(itertools.product(datasets, models, vel)):

        dataset, model, v = cfg

        nthr_     = estnthr(dataset.distances, v, dt, model_nsvs[model])
        gridDim_x = nthr_/blockDim_x if nthr_/blockDim_x < gridDim_x_full else gridDim_x_full
        nthr      = blockDim_x*gridDim_x
        G_, E_    = map(lambda a: repeat(a.flat, nic), meshgrid(gsc, exc))
        nlaunch = G_.size/nthr + (0 if G_.size%nthr + nthr > nthr_ else -1)

        if G_.size <= nthr:
            yield i, dataset, model, v, G_, E_
        else:
            for l in range(nlaunch):
                yield i, dataset, model, v, G_[  l   *nthr :(l+1)*nthr ], E_[  l   *nthr :(l+1)*nthr ]
            if G_.size - G_.size/nthr*nthr > 0:
                yield i, dataset, model, v, G_[ (l+1)*nthr :           ], E_[ (l+1)*nthr :           ]


if __name__ == '__main__':

    ngrid         = 64
    vel, gsc, exc = logspace(-0.3, 2, ngrid), logspace(-6., -0.5, ngrid), r_[0.8:1.2:ngrid*1j]
    datasets      = map(data.dsi.load_dataset, [0])
    models        = ['fhn_euler']
    nic           = 32

    dt            = 0.5
    tf            = 1000
    ds            = 5

    for j, d, m, v, g, e in launches(datasets, models, vel, gsc, exc, nic, dt):

        savename = 'run3/launch-%06d.npy' % (j,)
        if not file_exists(savename):
            result = gpu(g, e, v, dt, d, tf=tf, model=m, ds=ds)
            save(savename, result)
