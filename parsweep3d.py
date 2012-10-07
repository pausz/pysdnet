import string, time, sys, data.dsi, multiprocessing, gc, util, itertools
from pylab import *
from numpy import *

model_nsvs = dict(fhn_euler=2, bistable_euler=1)

def gpu(gsc, exc, vel, dt, dataset, tf=1500, ds=80, model="fhn_euler", cvar=0,
        kblock=128, ublock=1024, cat=concatenate):
    from cuda import srcmod, arrays_on_gpu # do not place in top of module!

    ts      = r_[0:tf:dt]
    n       = dataset.weights.shape[0]
    nsv     = model_nsvs[model]
    nthr    = len(gsc)
    npad    = nthr % ublock if nthr%kblock or nthr%ublock else 0
    nthr   += npad
    idel    = (dataset.distances/vel/dt).astype(int32)
    hist    = random.uniform(low=-1., high=1., size=(idel.max()+1, n, nthr))
    conn    = dataset.weights
    X       = random.uniform(low=-2, high=2, size=(n, nsv, nthr))
    Xs      = empty((1+len(ts)/ds, n, nsv, nthr), dtype=float32)

    mem = sum(map(lambda a: a.size*4, [idel, hist, conn, X, gsc, exc]))/2.0**20
    print '%d iterations, %0.1f MB GPU, %0.1f addnl CPU' % (len(ts), mem, Xs.nbytes/2.**20)

    mod = srcmod('parsweep.cu', ['kernel', 'update'], 
                 horizon=idel.max()+1, dt=dt, ds=ds, n=n, cvar=cvar, model=model, nsv=nsv)
    hist[-1, ...] = X[:, cvar, :]
    with arrays_on_gpu(_timed="integration", idel=idel.astype(int32), 
                       hist=hist.astype(float32), conn=conn.astype(float32), 
                       X=X.astype(float32), exc=cat((exc, zeros((npad,)))).astype(float32), 
                       gsc=cat((gsc, zeros((npad,)))).astype(float32)) as g:
        Xs[0, ...] = g.X.get()
        for step, t in enumerate(ts):
            mod.kernel(int32(step), g.idel, g.hist, g.conn, g.X, g.gsc, g.exc, block=(kblock, 1, 1), grid=(nthr/kblock, 1))
            mod.update(int32(step), g.hist, g.X, block=(ublock, 1, 1), grid=(nthr/ublock, 1))
            if step%ds == 0 and not (1+step/ds)>=len(Xs):
                Xs[1+step/ds, ...] = g.X.get().copy()
    return rollaxis(Xs, 3)[:-npad]

def launches(datasets, models, vel, G, E, nic, dt, blockDim_x=256, gridDim_x=256):
    for dataset, model, v in itertools.product(datasets, models, vel):
        nsv       = model_nsvs[model]
        nthr      = util.estnthr(dataset.distances, v, dt, nsv)
        gridDim_x = nthr/blockDim_x if nthr/blockDim_x < gridDim_x else gridDim_x
        nthr      = blockDim_x*gridDim_x
        G_, E_    = map(lambda a: repeat(a.flat, nic), meshgrid(gsc, exc))

        if G_.size <= nthr:
            yield_ = G_, E_
        else:
            for l in range(G_.size/nthr-1):
                yield_ = G_[  l   *nthr :(l+1)*nthr ], E_[  l   *nthr :(l+1)*nthr ]
            if G_.size - G_.size/nthr*nthr > 0:
                yield_ = G_[ (l+1)*nthr :           ], E_[ (l+1)*nthr :           ]
        yield (dataset, model, v) + yield_

def reducer(Xs, npar, nic):
    for X in Xs[:, -Xs.shape[1]/2:].reshape((npar, nic*Xs.shape[1]/2, -1)).copy():
        try:
            yield (lambda s: cumsum(s**2/sum(s**2))[3])(svd(X, full_matrices)[1])
        except LinAlgError:
            yield 1.0

if __name__ == '__main__':
    random.seed(42)
    ngrid         = 16
    vel, gsc, exc = logspace(0, 2, 128), logspace(-4, -1.5, ngrid), r_[0.75:1.25:ngrid*1j]
    datasets      = [data.dsi.load_dataset('ay')]
    models        = ['fhn_euler']
    nic           = 32
    dt            = 0.5
    tf            = 500
    results       = empty(map(len, [datasets, models, vel, gsc, exc]))
    buffer        = []
    i             = 0
    worker        = multiprocessing.Pool(1)

    for d, m, v, g, e in launches(datasets, models, vel, gsc, exc, nic, dt):
        buffer.append(worker.apply(gpu, (g, e, v, dt, d), dict(tf=tf, model=m)))
        if sum([len(x) for x in buffer]) == len(gsc)*len(exc)*nic:
            for r in reducer(temp, len(gsc)*len(exc), nic):
                results.flat[i] = r
                i += 1

    save('results.npy', results)
