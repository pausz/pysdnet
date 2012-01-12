import time
from numpy import *
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gary
from pycuda.sparse.packeted import PacketedSpMV as pSpMV
import scipy.io

class mySpMV(pSpMV):
    def __call__(self, x, y):

        self.get_kernel().prepared_call(
                (self.block_count, 1),
                self.packet_base_rows.gpudata,
                self.thread_starts.gpudata,
                self.thread_ends.gpudata,
                self.index_array.gpudata,
                self.data_array.gpudata,
                x.gpudata,
                y.gpudata)

        self.remaining_coo_gpu(x, y)

def setup(fname):

    spmc = scipy.io.mmread(fname).tocsr().astype(float32)
    spmg = mySpMV(spmc, False, spmc.dtype)

    xc = random.uniform(size=spmc.shape[0]).astype(float32)
    xg = gary.to_gpu(xc)

    yc = zeros(xc.shape[0]).astype(float32)
    yg = gary.to_gpu(yc)

    return spmc, spmg, xc, xg, yc, yg

def bench(fname, n=10000):

    print 'using %s' % (fname, )
    spmc, spmg, xc, xg, yc, yg = setup(fname)
    
    tic = time.time()
    for i in xrange(n):
        yc = spmc.dot(xc)
    toc = time.time()
    total = toc - tic
    per = 1000*total/n
    msg = 'cpu - size %s, %d loops took %f s, avg %f ms per' 
    print msg % (spmc.shape, n, total, per)

    tic = time.time()
    for i in xrange(n):
        spmg(xg, yg)
    toc = time.time()
    total = toc - tic
    per = 1000*total/n
    msg = 'gpu - size %s, %d loops took %f s, avg %f ms per' 
    print msg % (spmc.shape, n, total, per)

    spmc, spmg, xc, xg, yc, yg = setup(fname)
    yc = spmc.dot(xc)
    spmg(xg, yg)
    print 'ssq error = %f' %(sum((yc-yg.get())**2),)


if __name__ == '__main__':
    
    mtx_files = ['fmricov.mtx', 'fmricovbig.mtx']
    map(bench, mtx_files)
