/*

implement the following python numpy-based method, assuming
we'll be called by ctypes

def step(self, i, horizon, nids, idelays, dt, k, N, x, G, hist):
    """
    Perform single integration step

    i - current step number/count
    horizon - constant, size of maximumdelay
    nids - array of node ids used to index numpy
    idelays - array of delays, unit of integration step
    k - global coupling scale, constant
    N - num nodes, constant
    x - current state
    G - coupling matrix
    hist - history matrix, shape[0] = horizon + 1

    """

    # compute delayed state information, memory bandwidth hog
    delstate = hist[(i - 1 - idelays) % horizon, nids]

    # easy aligned memory access & copy, maybe use pointer
    x = hist[(i - 1) % horizon, :]

    # all math functions occur here + loop for G*delstate sum
    # k is constant for simulation
    dx = (x - 5*x**3)/5 + k*np.sum(G*delstate, axis=0)/N

    # aligned memory access again
    # random number generator used
    hist[i%horizon,:] = x + dt*(dx+randn(N)/5)

TODO: declare points not aliased once the module is working correctly
TODO: ctypes wrapper

*/

#include <math.h>
#include <stdio.h>


int wrap(int i, int h) {

    if (i >= 0)
        return i % h;
    else
        if (i == -h)
            return 0;
        else
            return h + (i % h);

}


static void node_step(int j, int i, int horizon, int *nids, int *idelays, float dt,
    float k, int N, float *x, float *G, float *hist, float *randn)
    /*
    step individual node; this is most like what we will
    use as a kernel for cuda

    j - is this node's id
    i - current step number/count
    horizon - constant, size of maximumdelay
    *nids - array of node ids used to index numpy, shape (N, N)
    *idelays - array of delays, unit of integration step, shape (N, N)
    k - global coupling scale, constant
    N - num nodes, constant
    *x - current state, has shape (N, )
    *G - coupling matrix, has shape (N, N)
    *hist - history matrix, has shape (horizon + 1, N)
    *randn - random numbers for this step, shape (N,)

    */
{

    float xj = hist[N*wrap(i - 1, horizon) + j];

    float input = 0.0, Iij;

    for (int idx=0; idx<N; idx++) {
        Iij = G[j*N + idx]*hist[N*(wrap((i - 1 - idelays[j*N + idx]), horizon)) + idx];
        input += Iij;
    }

    float dxj = (xj - 5.0*pow(xj, 3.0))/5.0 + k*input/N;

    hist[N*wrap(i, horizon) + j] = xj + dt*(dxj + randn[j]/5.0);
}



void sdde_step(int i, int horizon, int *nids, int *idelays, float dt,
    float k, int N, float *x, float *G, float *hist, float *randn)
    /*
    step all the nodes
    */
{
    int j;

    
    for (j=0; j<N; j++) {
        node_step(j, i, horizon, nids, idelays, dt, k, N, x, G, hist, randn);
    }

}


