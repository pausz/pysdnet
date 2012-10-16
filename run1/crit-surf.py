# coding: utf-8
from enthought.mayavi import mlab
get_ipython().system(u'ls -F ')
metrix = []
get_ipython().system(u'ls -F data')
import cPickle as cp
for i in range(705):
    with open('launch-%06d.metrics' % (i,), 'r') as fd:
        metrix.append(cp.load(fd))
    print i
    
metrix[0]
VAR = empty((9, 32, 32, 32))
from pylab import *
from numpy import *
VAR = empty((9, 32, 32, 32))
concatenate([m['var'] for m in metrix]).shape
VAR.flat.size
VAR.size
VAR.flat[:] = concatenate([m['var'] for m in metrix])
ml
from enthought.mayavi import mlab as ml
ml.figure()
get_ipython().system(u'ls -F ')
mlab
mlab.close()
mlab.show()
mlab.figure()
mlab.contour3d(*VAR[0])
mlab.contour3d(VAR[0])
mlab.show()
EXC = empty((9, 32, 32, 32))
GSC = empty((9, 32, 32, 32))
EXC.flat[:] = concatenate([m['exc'] for m in metrix])
GSC.flat[:] = concatenate([m['gsc'] for m in metrix])
GSC.flat[:160:16]
GSC
GSC.flat[:10]
GSC.flat[:320:32]
GSC.flat[1:320:32]
EXC.flat[1:320:32]
EXC.flat[0:320:32]
mlab.contour3d(VAR[0])
mlab.axes()
mlab.show()
#mlab.contour3d(VAR[0])
VEL = empty((9, 32, 32, 32))
metrix[0]['vel']
metrix[1]['vel']
metrix[1]['vel']*ones(metrix[1]['gsc'].shape)
VEL = concatenate([m['vel']*ones(m['gsc'].shape) for m in metrix])
VEL = empty((9, 32, 32, 32))
VEL.flat[:] = concatenate([m['vel']*ones(m['gsc'].shape) for m in metrix])
mlab.contour3d(VEL[0], EXC[0], GSC[0], VAR[0])
mlab.axes()
mlab.show()
get_ipython().magic(u'pinfo mlab.contour3d')
GSC[0]
GSC[0].ptp()
EXC[0].ptp()
VEL[0].ptp()
mlab.contour3d(VEL[0]/VEL[0].ptp(), EXC[0]/EXC[0].ptp(), GSC[0]/GSC[0].ptp(), VAR[0])
mlab.axes(); mlab.show()
EXC[0].ptp()
scl = lambda X: (X-X.min())/X.ptp()
mlab.contour3d(scl(VEL[0]), scl(EXC[0]), scl(GSC[0]), VAR[0])
mlab.show()
get_ipython().magic(u'pinfo meshgrid')
mgrid[0:1:3j, 4:5:4j]
VEL_, EXC_, GSC_ = mgrid[-1:2:32j, -5.:-1.5:32j, .95:1.1:32j]
VEL_ = log(VEL_)
EXC_ = log(EXC_)
mlab.contour3d(VEL_, EXC_, GSC_, VAR[0])
mlab.axes()
mlab.show()