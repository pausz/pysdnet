import string, time, sys, data.dsi, multiprocessing, gc, util, itertools, os
from cuda import srcmod, arrays_on_gpu # do not place in top of module!
import pycuda.driver
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
    mod = srcmod('parsweep.cu', ['kernel', 'update'], 
                 horizon=idel.max()+1, dt=dt, ds=ds, n=n, cvar=cvar, model=model, nsv=nsv)
    hist[-1, ...] = X[:, cvar, :]
    with arrays_on_gpu(_timed=False, _memdebug=True, idel=idel.astype(int32), 
                       hist=hist.astype(float32), conn=conn.astype(float32), 
                       X=X.astype(float32), exc=cat((exc, zeros((npad,)))).astype(float32), 
                       gsc=cat((gsc, zeros((npad,)))).astype(float32)) as g:
        Xs[0, ...] = g.X.get()
        for step, t in enumerate(ts):
            try:
                mod.kernel(int32(step), g.idel, g.hist, g.conn, g.X, g.gsc, g.exc, block=(kblock, 1, 1), grid=(nthr/kblock, 1))
                pycuda.driver.Context.synchronize()
            except Exception as E:
                raise Exception('mod.kernel failed', step, t, nthr, npad, kblock, nthr/kblock)

            try:
                mod.update(int32(step), g.hist, g.X, block=(ublock if nthr>=ublock else nthr, 1, 1), 
                           grid=(nthr/ublock if nthr/ublock > 0 else 1, 1))
                pycuda.driver.Context.synchronize()
            except Exception as E:
                raise Exception('mod.update failed', step, t, nthr, npad, ublock if nthr>=ublock else nthr, nthr/ublock if nthr/ublock>0 else 1)
            if step%ds == 0 and not (1+step/ds)>=len(Xs):
                Xs[1+step/ds, ...] = g.X.get()
    return (lambda X: X[:-npad] if npad else X)(rollaxis(Xs, 3))

def launches(datasets, models, vel, gsc, exc, nic, dt, blockDim_x=256, gridDim_x_full=256, _just_counting=False):
    for i, cfg in enumerate(itertools.product(datasets, models, vel)):
        dataset, model, v = cfg                                             # increase this since x0.7 anyway
        nthr_     = util.estnthr(dataset.distances, v, dt, model_nsvs[model], dispo=5300*2**20)
        gridDim_x = nthr_/blockDim_x if nthr_/blockDim_x < gridDim_x_full else gridDim_x_full
        nthr      = blockDim_x*gridDim_x
        G_, E_    = map(lambda a: repeat(a.flat, nic), meshgrid(gsc, exc))
        nlaunch = G_.size/nthr + (0 if G_.size%nthr + nthr > nthr_ else -1)

        if _just_counting:
            yield # BUG! needed to put this down below inside for loops
        else:
            print 'vel %0.2f, %d threads to run, blocks of %d possible, %d used in %d launches' % (v, G_.size, nthr_, nthr, nlaunch)
            if G_.size <= nthr:
                yield dataset, model, v, G_, E_
            else:
                for l in range(nlaunch):
                    yield dataset, model, v, G_[  l   *nthr :(l+1)*nthr ], E_[  l   *nthr :(l+1)*nthr ]
                if G_.size - G_.size/nthr*nthr > 0:
                    yield dataset, model, v, G_[ (l+1)*nthr :           ], E_[ (l+1)*nthr :           ]

# I need to start pulling the pieces apart here from the launch generator
# to the gpu workers to the reduction / output. we want this to be restartable
# as well as having access to the launch set later during analyses...

def reducer(Xs, npar, nic):
    for X in Xs[:, -(Xs.shape[1]/2):].reshape((npar, Xs.shape[1]/2*nic, -1)).copy():
        try:
            yield (lambda s: cumsum(s**2/sum(s**2))[3])(svd(X, full_matrices=0)[1])
        except LinAlgError:
            yield 1.0

if __name__ == '__main__':

    random.seed(42)
    ngrid         = 32
    vel, gsc, exc = logspace(-1, 2, ngrid), logspace(-5., -1.5, ngrid), r_[0.95:1.1:ngrid*1j]
    datasets      = map(data.dsi.load_dataset, range(9))
    models        = ['fhn_euler']
    nic           = 32
    dt            = 0.5
    tf            = 500
    ds            = 10
    j             = 0
    start_time    = time.time()
    launch_count  = 0

    # this will be completely unreliable if we're restarting
    print 'counting launches needed...'
    ls = launches(datasets, models, vel, gsc, exc, nic, dt, _just_counting=True)
    for l in ls:
        launch_count += 1
    print 'launch count at ', launch_count
    sys.stdout.flush()

    for d, m, v, g, e in launches(datasets, models, vel, gsc, exc, nic, dt):
        with open('./status.txt', 'w') as fd:
            elapsed = time.time() - start_time
            fd.write('%0.2f since start, %dth launch with vel %0.2f, nthr %r, %0.2f s remaining\n'
                        % (elapsed, j, v, g.shape, elapsed/(j+1.) * (launch_count - j)))
            fd.flush()

        savename = 'launch-%06d.npy' % (j,)
        try:
            os.stat(savename)
            print 'found this launch, skipping...'
        except OSError:
            try:
                result = gpu(g, e, v, dt, d, tf=tf, model=m, ds=ds)
                save(savename, result)
                print 'result.shape = ', result.shape
            except Exception as E:
                print 'launch', j, v, 'borked with ', E
        j += 1
        sys.stdout.flush()
