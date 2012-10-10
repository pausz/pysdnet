import multiprocessing

def pmap(func):
    def inner(args):
        pool = multiprocessing.Pool()
        pool.map(func, args)
    return inner

#                                       dispo=pycuda.autoinit.device.total_memory()
def estnthr(dist, vel, dt, nsv, pf=0.7, dispo=1535*2**20                           ):
    n = dist.shape[0]
    idelmax = long(dist.max()/vel/dt)
    return long( (dispo*pf - 4*n*n + 4*n*n)/(4*idelmax*n + 4*nsv*n + 4 + 4) )


