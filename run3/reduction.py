#!/usr/bin/python

import sys
import os
import time
import numpy as np

def msgr():
    tic = time.time()
    def msg(txt):
        print '%06.2f\t%s' % (time.time() - tic, txt)
    return msg

def reducer(i, _fail=[np.array([])]*3):

    print 'reducing condition ', i

    npy = np.lib.format.open_memmap(savename)
    npy_ = npy.reshape((-1, 32*npy.shape[1], 192))
    data = npy_[i]

    if np.isfinite(data).all():
        u, s, vt = np.linalg.svd(data, full_matrices=0)
        nc = where(np.cumsum(s**2/np.sum(s**2))>0.95)[0][0] + 1
        u1, s1, vt1 = u[:, :nc].astype(np.float32), s[:nc].astype(np.float32), vt[:nc].astype(np.float32)

        if (( np.dot(u1, np.dot(np.diag(s1), vt1)) - data )**2).mean()/data.ptp() >= 0.02:
            return _fail
        else:
            return u1, s1, vt1

    else:
        return _fail

    sys.stdout.flush()

if __name__ == '__main__':

    idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    ncores = int(sys.argv[2]) if len(sys.argv)>2 else 1

    savename = 'launch-%06d.npy' % (idx,)
    outname = 'reduced-%06d.pickle' % (idx,)

    msg = msgr()

    try:
        os.stat(outname)
        msg('found result file!')
    except:
        msg('no result found, proceeding to do reduction')
        msg('loading dataset %s' % savename)

        import cPickle as cp
        from numpy import *
        from numpy.linalg import svd
        from numpy.lib.format import open_memmap
        from multiprocessing import Pool

        npy = open_memmap(savename)
        npy_ = npy.reshape((-1, 32*npy.shape[1], 192))
        pool = Pool(ncores)

        svds = pool.map(reducer, range(npy_.shape[0]))

        msg('writing data')
        with open(outname, 'w') as fd:
            cp.dump(svds, fd)

