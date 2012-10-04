import string, time, sys, data.dsi, multiprocessing
from pylab import *
from numpy import *
from cuda import *

random.seed(42)

def estnthr(dist, vel, dt, nsv, n, dispo=pycuda.autoinit.device.total_memory()):
    idelmax = long(dist.max()/vel/dt)
    # dispo >= 8*n*n + 4*nthr*(2 + n*nsv + n*idelmax)
    return (dispo - 8*n*n)/(2 + n*nsv + n*idelmax)/4

# now goal is to obtain just cumsum(s**2/sum(s**2))[3]
vel, gsc, exc = r_[1:11:10j], logspace(-4, 0, 256), r_[0.75:1.25:256j]
G, E = meshgrid(gsc, exc)
dataset = data.dsi.load_dataset('ay')
n       = dataset.weights.shape[0]
nic     = 5 
nsv     = 2
dt      = 0.1

def gpu(*args):
    print map(lambda x: x.shape, args)

for i, v in enumerate(vel):

    # determine launch config based on device characteristics
    blockDim_x     = 256
    gridDim_x      = 256
    nthr           = estnthr(dataset.distances, v, dt, nsv, n)
    if nthr/blockDim_x < grid:
        gridDim_x = nthr/blockDim_x 
    nthr = blockDim_x*gridDim_x

    # implicit loop for n initial conditions per parameter set
    G_, E_ = repeat(G.flat, nic)[:], repeat(E.flat, nic)[:]

    nthrn = nic*G_.size 
    nlaunch = 1 + nthrn/nthr
    print '%dth velocity %0.1f, %d launches, nthr %d' % (i, v, nlaunch, nthr)

    Xss = []

    if nthrn <= nthr:
        Xss = [gpu(G_, E_)]
    else:
        for l in range(nlaunch-1):
            # this is bug! assuming nic strided in gpu, but here indexing G, E without replicating
            # them for nic correctly!! soln to pull out and loop nic here? no that makes ugliness
            # we can just create an implicit loop by replicating G, E in place as soon as they are
            # generated ...
            Xss.append(gpu(G_[  l   *nthr :(l+1)*nthr ], E_[  l   *nthr :(l+1)*nthr ]))

        if nthrn - (nic*G.size)/nthr*nthr > 0:
            Xss.append(gpu(G_[ (l+1)*nthr :           ], E_[ (l+1)*nthr :           ]))
            
    
    # reduce Xss correctly
        
        


if 0:

    tf      = 1500
    ds      = 80
    model   = "fhn_euler"
    cvar    = 0


    ts      = r_[0:tf:dt]
    idel    = (dataset.distances/vel/dt).astype(int32)
    hist    = random.uniform(low=-1., high=1., size=(idel.max()+1, n, nthr))
    conn    = dataset.weights
    X       = random.uniform(low=-2, high=2, size=(n, nsv, nthr))
    Xs      = empty((1+len(ts)/ds, n, nsv, nthr), dtype=float32)
    gsc     = -(logspace(-3.523, -0.745, npar)[:, newaxis])*ones((npar, nic)).astype(float32)
    exc     = 1.01*ones((npar, nic)).astype(float32)

    hist[-1, ...] = X[:, cvar, :]

    mem = sum(map(lambda a: a.nbytes, [idel, hist, conn, X, gsc, exc]))/2.0**20
    print '%d iterations, %0.1f MB GPU, %0.1f addnl GPU' % (len(ts), mem, Xs.nbytes/2.**20)

    mod = srcmod('parsweep.cu', ['kernel', 'update'],
                 horizon=idel.max()+1, dt=dt, ds=ds, n=n, cvar=cvar, model=model, nsv=nsv)

    with arrays_on_gpu(_timed="integration", idel=idel.astype(int32), 
                       hist=hist.astype(float32), conn=conn.astype(float32), 
                       X=X.astype(float32), exc=exc.astype(float32), gsc=gsc.astype(float32)) as g:

        Xs[0, ...] = g.X.get()
        for step, t in enumerate(ts):
            mod.kernel(int32(step), g.idel, g.hist, g.conn, g.X, g.gsc, g.exc, 
                       block=(64, 1, 1), grid=(64, 1))
            mod.update(int32(step), g.hist, g.X,
                       block=(1024, 1, 1), grid=(4, 1))
            if step%ds == 0 and not (1+step/ds)>=len(Xs):
                Xs[1+step/ds, ...] = g.X.get()

    tic = time.time()
    Xs -= Xs.mean(axis=0)
    Xs = rollaxis(Xs, 3)

