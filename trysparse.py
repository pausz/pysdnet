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





