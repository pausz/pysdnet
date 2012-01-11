#

"""
quick DE integration benchmark with noise and delays
using CUDA

parts to work on:

- node update kernel
- coupling kernel
- noise kernel

questions:

? Do we really have to update the coupling with same time step
    as the state? 
? We can take advantage of the delays using a more complex scheme
    which can do short and long delays separately

.. moduleauthor Marmaduke Woodman <duke@eml.cc>
"""

import numpy as np
import time
import logging as lg

lg.basicConfig(level=lg.INFO)
log = lg.getLogger(__name__)

dt = 2**-8 # ~ 0.00195 ms
maxdelay = 2**6 # 64 ms 

randn = np.random.normal
rand = np.random.uniform

def fhn(state, input, tau=3, a=1.001, b=0):
    len = state.shape[0]/2
    x, y = state[:len], state[len:]
    dx = (x-x**3/3+y)*tau
    dy = -(x-a+b*y-input)/tau
    return np.concatenate((dx, dy))

def integrate(nsteps, ssize, 
               dt=dt, vf=fhn, maxdelay=maxdelay, pars={}):
    tic = time.time()
    horizon = int(maxdelay/dt) + 1
    ys = np.zeros((horizon, ssize*2))
    
    RX, RY = rand(size=(ssize, ssize)), rand(size=(ssize, ssize))
    idelays = np.asarray((horizon-1)*(RX-RY)**2, dtype='int64')
    
    gij = randn(size=(ssize,ssize))
    k = pars.pop('k') if pars.has_key('k') else 1.0
    sigma = pars.pop('sigma') if pars.has_key('sigma') else 2.0
    node_ids = np.tile(np.arange(ssize), (ssize, 1))
    
    toc = time.time()
    log.info('done allocing memory and setting up, took %f seconds' % (toc-tic))
    for key, val in locals().iteritems():
        log.debug('integrate(): %s = %s'%(key,val))
    log.info('starting integration')
    tic = time.time()
    for i in xrange(1, nsteps):

        delstate = ys[(i-1-idelays)%horizon, node_ids]
        input = k*np.sum(delstate*gij, axis=0)/ssize

        ys[i%horizon,:] = ys[(i-1)%horizon,:] +\
                             dt*(fhn(ys[(i-1)%horizon,:], input, **pars) +
                                 randn(size=ssize*2, scale=sigma))
    toc = time.time()
    log.info('integration took %s seconds' % (toc-tic,))
    return ys

update_src = """
__global__ void update(int step, float *ts, float *ys, float *gij) {

    unsigned int i = threadIdx.x + $stride * threadIdx.y, j;
    float delstate, input = 0.;

    for (j=0;j<$N;j++) {
        delstate = ys[ nodeid[j]*$stride + (1-idelay[j])%horizon];
        input += k*gij[i + $N*j]*delstate;
    }

    float x = ys[step*$stride + 2*j], y = ys[step*$stride + 2*j + 1];
    ys[((step+1)*$stride)%horizon + 2*j+0] = x + $dt*(x - pow(x, 3.0) + y)*$tau;
    ys[((step+1)*$stride)%horizon + 2*j+1] = y - $dt*(x - a + b*y - input)/$tau;

}
"""

# template with [dt, horizon, 

sim_sizes = {'small'  : 2**2,  # 4 nodes
             'medium' : 2**6,  # 64 nodes
             'large'  : 2**13} # 8192 nodes

num_steps = {'small'  : int(2**7/dt),  # 128 ms
             'medium' : int(2**15/dt), # 32 seconds
             'large'  : int(2**21/dt)} # 35 minutes

params = {'k' : {'default' : 1.0,
                 'lo' : 0.0,
                 'hi' : 10.0},
          
          'sigma' : {'default' : 2.0,
                     'lo' : 0.0,
                     'hi' : 10.0},
          
          'a' : {'default' : 2.0,
                 'lo' : 0.8,
                 'hi' : 1.2}}

def random_params():
    return dict((k, rand(low=v['lo'], high=v['hi']))
                for k, v in params.iteritems())


# example usage
if __name__ == '__main__':
        
    ys = integrate(num_steps['small'],
                   sim_sizes['small'],
                   pars=random_params())

