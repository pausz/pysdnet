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

mem_est = int(dataset.distances.max()/vel/dt)*nthr*nsv*n*4/2.**30
print "GPU mem use O(%0.3f GB)" % mem_est

ts      = r_[0:tf:dt]
idel    = (dataset.distances/vel/dt).astype(int32)
hist    = random.uniform(low=-1., high=1., size=(idel.max()+1, n, nthr))
conn    = dataset.weights
X       = random.uniform(low=-2, high=2, size=(n, nsv, nthr))
Xs      = empty((1+len(ts)/ds, n, nsv, nthr), dtype=float32)
gsc     = -(logspace(-3.523, -0.745, npar)[:, newaxis])*ones((npar, nic)).astype(float32)
exc     = 1.01*ones((npar, nic)).astype(float32)

hist[-1, ...] = X[:, cvar, :]

print 'will do %d iterations' % (len(ts),)
ars = [idel, hist, conn, X, gsc, exc]
print 'using %0.1f MB on GPU' % (sum(map(lambda a: a.nbytes, ars))/2.0**20, )
print 'plus additional %0.1f MB on CPU' % (Xs.nbytes/2.0**20,)

mod = srcmod('parsweep.cu', ['kernel', 'update'],
             horizon=idel.max()+1, dt=dt, ds=ds, n=n, cvar=cvar, model=model, nsv=nsv)

with arrays_on_gpu(_timed="integration",
                   idel=idel.astype(int32), 
                   hist=hist.astype(float32), conn=conn.astype(float32), 
                   X=X.astype(float32), exc=exc.astype(float32), gsc=gsc.astype(float32)) as g:

    Xs[0, ...] = g.X.get()

    for step, t in enumerate(ts):
        
        mod.kernel(int32(step), g.idel, g.hist, g.conn, g.X, g.gsc, g.exc, 
                   block=(64, 1, 1), grid=(64, 1))
        mod.update(int32(step), g.hist, g.X,
                   block=(1024, 1, 1), grid=(4, 1))

        if step%ds == 0:
            Xs[1+step/ds, ...] = g.X.get()

tic = time.time()
Xs -= Xs.mean(axis=0)
Xs = rollaxis(Xs, 3)

def one_fig(arg):
    i, Xsi = arg
    u, s, vt = svd( Xsi.reshape((-1, nsv*96)), full_matrices=0 )
    figure(figsize=(16, 8))
    for j, X in enumerate(Xsi):
        """
        subplot(121)
        xs = (s[:7]*dot(vt[:7], X.reshape((-1, nsv*96)).T).T/s[0]).T
        plot(xs[0], xs[1], 'k', alpha=0.1)
        subplot(122)
        """
        for x, c in zip(xs, 'kbgrcmy'):
            plot(ts[::ds], x[1:], c, alpha=0.1)
    title('vel=%0.1f, gsc=%0.4f, exc=%0.2f' % (vel, gsc[i,0], exc[i, 0]))
    savefig('iter%03d.png' % i)

pool = multiprocessing.Pool(10)
pool.map(one_fig, enumerate(Xs.reshape((npar, nic, Xs.shape[1], n, nsv))))
    
print 'postproc & plotting %0.3f s' % (time.time() - tic,)
