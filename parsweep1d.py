import string, time, sys, data.dsi, multiprocessing
from pylab import *
from numpy import *
from cuda import *

random.seed(42)

dataset = data.dsi.load_dataset('ay')
n       = dataset.weights.shape[0]
nic     = 64
npar    = 64
nthr    = npar*nic
tf      = 1500
dt      = 0.1
ds      = 80
vel     = 4.0
model   = "fhn_euler"
nsv     = 2
cvar    = 0

assert int(dataset.distances.max()/vel/dt)*nthr*nsv*n*4 * 1.1 < pycuda.autoinit.device.total_memory()

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

def one_fig(i, Xsi, s, vt):
    figure(figsize=(8, 8)), subplot(221)
    plot(cumsum(s**2/sum(s**2)), linewidth=8), xlim([0, 10]), ylim([0, 1])
    title("cumulative variance / component")
    for j, X in enumerate(Xsi):
        xs = (s[:7]*dot(vt[:7], X.reshape((-1, nsv*96)).T).T/s[0]).T
        subplot(222)
        plot(xs[0], xs[1], 'k', alpha=0.1)
        title("2D projection of trial time series")
        subplot(212)
        for x, c in zip(xs, 'kbgrcmy'):
            nts = len(ts[::ds])
            plot(ts[::ds], x[:nts], c, alpha=0.1)
        title("component time series all trials")
    suptitle('vel=%0.1f, gsc=%0.4f, exc=%0.2f' % (vel, gsc[i,0], exc[i, 0]))
    savefig('iter%03d.png' % i, dpi=200)

pool = multiprocessing.Pool(10)
variations = enumerate(Xs.reshape((npar, nic, Xs.shape[1], n, nsv)))

res = [pool.apply_async(svd, (Xsi[:, -(len(ts)/ds/2):].reshape((-1, nsv*96)),), {'full_matrices':0}) for i, Xsi in variations]
svds = [r.get() for r in res]
pool.close()
pool.join()

pool = multiprocessing.Pool(10)
res = [pool.apply_async(one_fig, (i, Xsi, svds[i][1], svds[i][2])) for i, Xsi in variations]
[r.get() for r in res]
pool.close()
pool.join()

print 'postproc & plotting %0.3f s' % (time.time() - tic,)
