# common helpers to boot up pycuda

import time

try:
    import pyublas
except ImportError as exc:
    global __pyublas__available__
    __pyublas__available__ = False

try:
    import pycuda.autoinit
    import pycuda.gpuarray as gary
    from pycuda.compiler import SourceModule
    from pycuda.tools import DeviceData, OccupancyRecord

except Exception as exc:
    print "importing pycuda modules failed with exception", exc
    print "please check PATH and LD_LIBRARY_PATH variables"


def orinfo(n):
    orec = OccupancyRecord(DeviceData(), n)
    return """occupancy record information
        thread blocks per multiprocessor - %d
        warps per multiprocessor - %d
        limited by - %s
        occupancy - %f
    """ % (orec.tb_per_mp, orec.warps_per_mp, orec.limited_by, orec.occupancy)

from kernels import srcmod

class arrays_on_gpu(object):

    def __init__(self, _timed="gpu timer", **arrays):

        self.__array_names = arrays.keys()
    
        for key, val in arrays.iteritems():
            setattr(self, key, gary.to_gpu(val))

        self._timed_msg = _timed

    def __enter__(self, *args):
        self.tic = time.time()
        return self

    def __exit__(self, *args):

        if self._timed_msg:
            print "%s %0.3f s" % (self._timed_msg, time.time() - self.tic)

        for key in self.__array_names:
            delattr(self, key)

   
