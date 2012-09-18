/* CUDA kernel template for parameter sweeping */

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
            hist_idx = n_thr*$n*wrap(step - 1 - idel_ij) + i + 0 + par_ij;
            input += conn_ij*hist[hist_idx];
        }

         x = hist[n_thr*$n*wrap(step - 1) + i + 0 + par_ij];
        dx = (x - x*x*x/3.0)/20.0 + gsc*input + exc;

        X[n_thr*i + 0 + par_ij] = x + $dt*dx;
    }
}

__global__ void update(int step, float *hist, float *X)
{
    int par_ij = threadIdx.x
      , n_thr  = blockDim.x
      ;

    for (int i=0; i<$n; i++)
        hist[n_thr*$n*wrap(step) + i + 0 + par_ij] = X[n_thr*i + 0 + par_ij];
}

