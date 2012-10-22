import cPickle as cp
import bz2 as bz
import time

def io(idx, compress=True, name='reduced-%06d.pickle'):

    tic = time.time()
    pm, bm = 'rw' if compress else 'wr'
    name %= idx
    with open(name, pm) as fdp:
        with bz.BZ2File(name+'.bz', bm) as fdb:
            if compress:
                cp.dump(cp.load(fdp), fdb)
            else:
                cp.dump(cp.load(fdb), fdp)
    print '%d required %.2f s' % (idx, time.time() - tic)


if __name__=='__main__':
    import multiprocessing, sys
    tic = time.time()
    pool = multiprocessing.Pool(8)
    pool.map(io, range(int(sys.argv[1])))
    print 'all done in %.3 h' % ((time.time() - tic)/3600.,)
    
