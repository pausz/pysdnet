/* CUDA kernel template for parameter sweeping 

    - striding requires attention! may be good to template that as well, so that
        Python level code just specs axes by semantic labels rather than manually
        speccing strides to generate linear indices.

    - adding a model with more than one state variable will require rewriting the
        index expressions to insert n_state_vars 

    - use templating to avoid function pointers for efficient use of various 
        models

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

/* models we use */
__device__ void pitchfork();
__device__ void hopf();
__device__ void kuramoto();
// etc

__global__ void kernel(int step, int *idel, float *hist, float *conn, float *X)
{

    /* hist[step, node, var, paridx], X[node, var, paridx] */

    int par_i  = blockIdx.x
      , par_j  = threadIdx.x
      , par_ij = blockDim.x*par_i + par_j
      , n_thr  = blockDim.x*gridDim.x
      , idel_ij, hist_idx
      ;

    float x, dx, input, conn_ij
      , gsc = $gsc0 + par_i*$dgsc
      , exc = $exc0 + par_j*$dexc
      ;

    for (int i=0; i<$n; i++)
    {
        input = 0.0;
        for (int j=0; j<$n; j++) {
            conn_ij = conn[j*$n + i];
            idel_ij = idel[j*$n + i];
            hist_idx = n_thr*$n*wrap(step - 1 - idel_ij) + n_thr*i + 0 + par_ij;
            input += conn_ij*hist[hist_idx];
        }

        input *= gsc/$n;

         x = hist[n_thr*$n*wrap(step - 1) + n_thr*i + 0 + par_ij];

        // <model specific code>
        dx = (x - x*x*x/3.0)/20.0 + input + exc;
        // </model specific code>

        X[n_thr*i + 0 + par_ij] = x + $dt*dx;
    }
}

__global__ void update(int step, float *hist, float *X)
{
    int par_ij = threadIdx.x
      , n_thr  = blockDim.x
      ;

    for (int i=0; i<$n; i++)
        hist[n_thr*$n*wrap(step) + n_thr*i + 0 + par_ij] = X[n_thr*i + 0 + par_ij];
}

