
from time import time
import string
import numpy as np
from cuda import *

"""
prototype 32x32 par space exploration
"""

### setup data and parameters

n = 96

tf = 100
dt = 0.05
ds = 10
ts = np.r_[0:tf:dt]

vel =  2.0
gsc = ( -100, -100, 32j)
exc = (-10, 10, 32j)

idel = (np.random.uniform(low=3, high=160/vel, size=(n, n))/dt).astype(np.int32)
hist = np.zeros((idel.max()+1, n, 1, 1024), dtype=np.float32)
conn = np.random.normal(scale=0.1, size=(n, n)).astype(np.float32)
X    = np.random.uniform(low=-0.1, high=0.1, size=(n, 1, 1024)).astype(np.float32)
Xs   = np.empty((len(ts)/ds, n, 1, 1024), dtype=np.float32)

# setup cuda kernel
with open('./parsweep.cu') as fd:
    source = string.Template(fd.read())

pars = dict(horizon=idel.max()+1, dt=dt, ds=ds, n=n,
            gsc0=gsc[0], dgsc=(gsc[1]-gsc[0])/gsc[2].imag,
            exc0=exc[0], dexc=(exc[1]-exc[0])/exc[2].imag)

module = SourceModule(source.substitute(**pars))
kernel = module.get_function("kernel")
update = module.get_function("update")

g_idel = gary.to_gpu(idel)
g_hist = gary.to_gpu(hist)
g_conn = gary.to_gpu(conn)
g_X    = gary.to_gpu(X)

# block=(32, 1, 1), grid=(32, 1)

# step through simulation
tk, tu, tx = 0., 0., 0.
for step, t in enumerate(ts):
    
    tic = time()
    kernel(np.int32(step), g_idel, g_hist, g_conn, g_X, block=(32, 1, 1), grid=(32, 1))
    tac = time()
    update(np.int32(step), g_hist, g_X, block=(1024, 1, 1), grid=(1, 1))
    toc = time()

    if step%ds == 0:
        Xs[step/ds, ...] = g_X.get()

    txc = time()

    tk += tac - tic
    tu += toc - tac
    tx += txc - toc

# cleanup memory
del g_idel
del g_hist
del g_conn
del g_X

# normalize timing
tk /= len(ts)
tu /= len(ts)
tx /= len(ts)

# save data
np.savez('sim-data', ts, Xs)

print '%f ms / iteration' % (1000*(tk+tu+tx), )
