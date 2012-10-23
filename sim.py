from numpy import *
from data.dsi import load_dataset
from gen import module, model, fhn, step, wrap
from cee import srcmod
import ctypes, time

C = ascontiguousarray

tf = 500
dt = 0.5
ts = r_[0:tf:dt]
vel = 1.0
n = 96
exc = 0.95
gsc = 0.001
ds = 10
nic = 1000

dataset = load_dataset(0)

idel = C((dataset.distances/vel/dt).astype(int64))
horizon = idel.max() + 1
conn = C(dataset.weights)
gsc = C(array([gsc]))
exc = C(array([exc]))
Xs = empty((nic, len(ts)/ds,) + X.shape, dtype=float16)
print Xs.nbytes/2**30.


mod = srcmod(module(model(dt=dt, **fhn), 
                    step(n, len(fhn['eqns']), model=fhn['name'], nunroll=1),
                    wrap(horizon)),
             ['step'], debug=False)


tic = time.time()

for j in range(nic):

    hist = C(zeros((horizon, n)))
    X = C(random.normal(size=(n, 2)))/10.
    p_idel, p_conn, p_hist, p_X, p_gsc, p_exc = \
        map(lambda A: A.ctypes.data_as(ctypes.c_void_p), [idel, conn, hist, X, gsc, exc])

    for i, t in enumerate(ts):
        mod.step(i, p_idel, p_hist, p_conn, p_X, p_gsc, p_exc)
        if i%ds == 0 and i/ds < Xs.shape[1]:
            Xs[j, i/ds] = X.copy()

toc = time.time() - tic
print "%.2f s total, %.3f ms / iter" % (toc, 1000*toc/len(ts))

from pylab import *
for trial in Xs:
    [plot(x + 4*i, 'k', alpha=0.1) for i, x in enumerate(trial[:, ::10, 0].T)];
