# common helpers to boot up pycuda

try:
    import pyublas
except ImportError as exc:
    print "pyublas module not found; some C++ <-> Python data converters"
    print "will be unavaible, possibly causing data type exceptions"

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

    def __init__(self, **arrays):

        self.__array_names = arrays.keys()
    
        for key, val in arrays.iteritems():
            setattr(self, key, gary.to_gpu(val))

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):

        for key in self.__array_names:
            delattr(self, key)
