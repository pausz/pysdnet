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

// begin defining models
#define defmodel(name, X, pars, n_thr, par_ij, input) inline __device__ void name\
    (float * __restrict__ X, void * __restrict__ pars, int n_thr, int par_ij, float input)

defmodel(bistable_euler, Y, p, nt, pi, i)
{
    float x   = Y[nt*0 + pi]
       ,  exc = *((float*) p)
       ,  dx  = (x - x*x*x/3.0)/20.0 + i + exc;

    Y[nt*0 + pi] = x + $dt*dx;
}

defmodel(fhn_euler, X, pars, nt, pi, in)
{
    float x = X[nt*0 + pi]
        , y = X[nt*1 + pi]
        , a = *((float*) pars)

        , dx = (x - x*x*x/3.0 + y)*3.0
        , dy = (a - x + in)/3.0;

    X[nt*0 + pi] = x + $dt*dx;
    X[nt*1 + pi] = y + $dt*dy;
}

#undef defmodel
// end model definitions

__global__ void kernel(int step, int * __restrict__ idel, 
                       float * __restrict__ hist, 
                       float * __restrict__ conn, 
                       float * __restrict__ X
                       )
{

    int par_i  = blockIdx.x
      , par_j  = threadIdx.x
      , par_ij = blockDim.x*par_i + par_j
      , n_thr  = blockDim.x*gridDim.x
      , hist_idx
      ;

    float input
      , gsc = $gsc0 + par_i*$dgsc
      , exc = $exc0 + par_j*$dexc
      ;

    for (int i=0; i<$n; i++)
    {
        input = 0.0;

        for (int j=0; j<$n; j++) {

                    //   stride*index
            hist_idx = $n*n_thr*wrap(step - 1 - idel[j*$n + i])  // step
                     +    n_thr*i                                // node index 
                     +        1*par_ij;                          // parsweep index

            input += conn[j*$n + i]*hist[hist_idx];
        }

        input *= gsc/$n;

        $model(X + n_thr*$nsv*i, &exc, n_thr, par_ij, input);
    }
}

// update history
__global__ void update(int step, float *hist, float *X)
{
    int par_ij = threadIdx.x
      , n_thr  = blockDim.x
      ;

    for (int i=0; i<$n; i++)
        hist[n_thr*$n*wrap(step) + n_thr*i + par_ij] = X[$nsv*n_thr*i + n_thr*$cvar*0 + par_ij];
}

