import string, time, sys, data.dsi, multiprocessing, gc, util, itertools, os, cPickle
from pylab import *
from numpy import *

model_nsvs = dict(fhn_euler=2, bistable_euler=1)

def launches(datasets, models, vel, gsc, exc, nic, dt, blockDim_x=256, gridDim_x_full=256, _just_counting=False):
    for i, cfg in enumerate(itertools.product(datasets, models, vel)):
        dataset, model, v = cfg                                             # increase this since x0.7 anyway
        nthr_     = util.estnthr(dataset.distances, v, dt, model_nsvs[model], dispo=5300*2**20)
        gridDim_x = nthr_/blockDim_x if nthr_/blockDim_x < gridDim_x_full else gridDim_x_full
        nthr      = blockDim_x*gridDim_x
        G_, E_    = map(lambda a: repeat(a.flat, nic), meshgrid(gsc, exc))
        nlaunch = G_.size/nthr + (0 if G_.size%nthr + nthr > nthr_ else -1)

        if _just_counting:
            yield # BUG! needed to put this down below inside for loops
        else:
            print 'vel %0.2f, %d threads to run, blocks of %d possible, %d used in %d launches' % (v, G_.size, nthr_, nthr, nlaunch)
            if G_.size <= nthr:
                yield dataset, model, v, G_, E_
            else:
                for l in range(nlaunch):
                    yield dataset, model, v, G_[  l   *nthr :(l+1)*nthr ], E_[  l   *nthr :(l+1)*nthr ]
                if G_.size - G_.size/nthr*nthr > 0:
                    yield dataset, model, v, G_[ (l+1)*nthr :           ], E_[ (l+1)*nthr :           ]


if __name__ == '__main__':

    random.seed(42)
    ngrid         = 64
    vel, gsc, exc = logspace(-0.3, 2, ngrid), logspace(-6., -0.5, ngrid), r_[0.8:1.2:ngrid*1j]
    datasets      = map(data.dsi.load_dataset, [0])
    models        = ['fhn_euler']
    nic           = 32
    dt            = 0.5
    tf            = 1000
    ds            = 5
    j             = 0
    start_time    = time.time()
    launch_count  = 0


    for d, m, v, g, e in launches(datasets, models, vel, gsc, exc, nic, dt):
        savename = 'launch-%06d.npy' % (j,)

        # do this on the whole cluster
        """
        npy = load(savename)
        npy_ = npy.reshape((-1, 32*npy.shape[1], 192))

        svds = []
        for idx, data in enumerate(npy_):
            u, s, vt = svd(data, full_matrices=0)
            nc = where(cumsum(s**2/sum(s**2))>0.95)[0][0]
            u1, s1, vt1 = u[:, :nc], s[:nc], vt[:nc]
            assert (( dot(u1, dot(diag(s1), vt1)) - data )**2).mean()/data.ptp() < 0.02
            svds.append((u1, s1, vt1))
            print j, idx
        """

        cfg = {}

        cfg['centers'] = d.centers
        cfg['distances'] = d.distances
        cfg['weights'] = d.weights
        cfg['ngrid'] = ngrid
        cfg['vel'] = v
        cfg['gsc'] = g
        cfg['exc'] = e
        cfg['model'] = m
        cfg['nic'] = nic
        cfg['dt'] = dt
        cfg['tf'] = tf
        cfg['ds'] = ds
        #cfg['svds'] = svds

        with open('config-%06d.pickle' % (j, ), 'w') as fd:
            cPickle.dump(cfg, fd)

        j+= 1
        print j
        sys.stdout.flush()
