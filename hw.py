import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

class hw:

    src = """
    __global__ void doublify(float *a) {
        int i = threadIdx.x + threadIdx.y*%d;
        a[i] *= 2;
    }
    """

    def __init__(self, N=16):
    
        self.N = N

        # setup data
        self.a = np.random.randn(N, N).astype(np.float32)
        self.a_gpu = cuda.mem_alloc(self.a.nbytes)
        cuda.memcpy_htod(self.a_gpu, self.a)

        # setup code
        self.mod = SourceModule(self.src % (N,))
        self.func = self.mod.get_function("doublify")

    def __call__(self, cpu=False):
        if cpu:
            self.a = 2*self.a
        else:
            self.func(self.a_gpu, block=(self.N, self.N, 1))

    def geta(self):
        cuda.memcpy_dtoh(self.a, self.a_gpu)
        return self.a





