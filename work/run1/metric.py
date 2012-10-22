import cPickle, se, time, sys
from numpy import load, array

start_time = time.time()
def log(msg):
    print '%06.1f\t%s' % (time.time() - start_time, msg)

launch_id = (int(sys.argv[1])-1) if len(sys.argv)>1 else 0
log('this is launch_id=%d'% launch_id)

npy = load('launch-%06d.npy'%launch_id)
with open('launch-%06d.pickle'%launch_id) as fd:
    cfg=cPickle.load(fd)
log('data loaded')


npt = 32*101*96*2
log('launch contains %d conditions'% (npy.size/npt,))

exc = cfg['exc'].reshape((-1, 32))[:,0]
gsc = cfg['gsc'].reshape((-1, 32))[:,0]
vel = cfg['vel']
log('launch config obtained')

var = npy.reshape((-1, npt)).var(axis=1)
log('global variance done')
ses = []
for i, cond in enumerate(npy.reshape(-1, 32, 101, 96, 2)):
    temp = 0.0
    for j, trial in enumerate(cond[:, :, :, 0].reshape((32, -1))):
        temp += se.se(trial, m=3, taus=5) 
        log('sample entropy, %dth signal done, %dth trial' % (i, j))
    ses.append(temp/(j+1))
ses = array(ses)
log('global sample entropy done')


with open('launch-%06d.metrics'%launch_id, 'w') as fd:
    metrics = dict(exc=exc, gsc=gsc, vel=vel, var=var, ses=ses)
    cPickle.dump(metrics, fd)

log('data written to launch-%06d.metric'%launch_id)

