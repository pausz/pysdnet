/* CUDA kernel template for parameter sweeping 

    - striding requires attention!

    - hist contains only one state variable, cvar, while X contains all of them
        between the node and parsweep dims

    - the mapping between launch configuration and parameter space grid should be
        done more systematically 

*/

inline __device__ int wrap(int i) {

    if (i >= 0)
        return i % $horizon;
    else
        if (i == - $horizon)
            return 0;
        else
            return $horizon + (i % $horizon);

}

__global__ void kernel(int step, int *idel, float *hist, float *conn, float *X)
{

    /* hist[step, node, var, paridx], X[node, var, paridx] */

    int par_i  = blockIdx.x
      , par_j  = threadIdx.x
      , par_ij = blockDim.x*par_i + par_j
      , n_thr  = blockDim.x*gridDim.x
      , hist_idx
      ;

    float x, dx, input
      , gsc = $gsc0 + par_i*$dgsc
      , exc = $exc0 + par_j*$dexc
      ;

    for (int i=0; i<$n; i++)
    {
        input = 0.0;

        for (int j=0; j<$n; j++) {
            hist_idx = $n*n_thr*wrap(step - 1 - idel[j*$n + i])  // step
                     +    n_thr*i                                // node index 
                     +        1*par_ij;                          // parsweep index
            input += conn[j*$n + i]*hist[hist_idx];
        }

        input *= gsc/$n;

        x = X[n_thr*nsv*i + n_thr*0 + par_ij]

        // <model specific code>
        dx = (x - x*x*x/3.0)/20.0 + input + exc;
        // </model specific code>

        X[n_thr*i + 0 + par_ij] = x + $dt*dx;
    }
}

// update history
__global__ void update(int step, float *hist, float *X)
{
    int par_ij = threadIdx.x
      , n_thr  = blockDim.x
      ;

    for (int i=0; i<$n; i++)
        hist[n_thr*$n*wrap(step) + n_thr*i + par_ij] = X[nsv*n_thr*i + n_thr*$cvar*0 + par_ij];
}

