/* step.c

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

static void node_step(int j, int i, int horizon, int *nids, int *idelays, double dt,
    double k, int N, double *x, double *G, double *hist, double *randn)
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

    // >>> x = hist[(i - 1) % horizon, :]
    double xj = hist[N*((i - 1) % horizon) + j];

    /*
    compute delayed state information, memory bandwidth hog

    >>> delstate = hist[(i - 1 - idelays) % horizon, nids]

    all math functions occur here + loop for G*delstate sum
    k is constant for simulation

    >>> dx = (x - 5*x**3)/5 + k*np.sum(G*delstate, axis=0)/N

    Here though, it behooves us to combine the delayed state access
    and input computation. Then, we can compute dxj.

    */

    double input = 0.0;

    for (int idx=0; idx<N; idx++)
        input += G[j*N + idx]*hist[idelays[j*N + idx] + idx];

    double dxj = (xj - 5.0*pow(xj, 3.0))/5.0 + k*input/N;

    // # aligned memory access again
    // # random number generator used
    // hist[i%horizon,:] = x + dt*(dx+randn(N)/5)

    hist[i % horizon + j] = xj + dt*(dxj + randn[j]/5.0);

}



static void sdde_step(int i, int horizon, int *nids, int *idelays, double dt,
    double k, int N, double *x, double *G, double *hist, double *randn)
    /*
    step all the nodes
    */
{
    int j;

    for (j=0; j<N; j++)
        node_step(j, i, horizon, nids, idelays, dt, k, N, x, G, hist, randn);
}


