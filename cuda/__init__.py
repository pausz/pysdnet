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

    def __init__(self, _timed="gpu timer", _memdebug=False, **arrays):

        self.__array_names = arrays.keys()

        if _memdebug:
            memuse = sum([v.size*v.itemsize for k, v in arrays.iteritems()])/2.**20
            memavl = pycuda.autoinit.device.total_memory()/2.**20
            print 'GPU mem use %0.2f MB of %0.2f avail.' % (memuse, memavl)
            for k, v in arrays.iteritems():
                print 'gpu array %s.shape = %r' % (k, v.shape)
            assert memuse <= memavl
    
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

   
