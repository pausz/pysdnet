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


