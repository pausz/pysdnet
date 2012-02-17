/*

CUDA kernel template for integrating an SDDE.

o  write pycuda wrapper
o  test against cpu versions
o  look at delay distributions
o  work out redundant partition scheme

template params: threadid, N, horizon, k, dt

functions

    - wrap() takes the step number and returns the corresponding step number 
        assuming periodic boundaries.

    - step() 

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


__global__ void step(int i, // current step number/count
                     int * __restrict__ idelays, // delays in steps (N, N)
                     float * __restrict__ G, // coupling matrix (N, N)
                     float * __restrict__ hist, // history (horizon + 1, N)
                     float * __restrict__ randn) // randnums for this step (N,)

{

    int j = $threadid;

    float xj, dxj, input;

    input = 0.0;
    for (int idx=0; idx<$N; idx++)
        input += G[j*$N + idx]*hist[$N*wrap(i - 1 - idelays[j*$N + idx]) + idx];

     xj = hist[$N*wrap(i - 1) + j];
    dxj = $dt*(  (xj - 5.0*pow(xj, 3.0))/5.0 + $k*input/$N + randn[j]/5.0 );

    __synchthreads();
    hist[$N*wrap(i) + j] = xj + dxj;

}


__global__ void get_state(int i, // current step no.
                          float * __restrict__ hist, // history
                          float * __restrict__ xout) // output

{
    int j = $threadid;
    xout[j] = hist[$N*wrap(i) + j];
}

