/*

CUDA kernel template for integrating an SDDE.

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


__global__ void step(int * _i, // current step number/count
                     int * __restrict__ idelays, // delays in steps (N, N)
                     float * __restrict__ G, // coupling matrix (N, N)
                     float * __restrict__ hist, // history (horizon + 1, N)
                     float * __restrict__ xout, // current state
                     float * __restrict__ randn) // randnums for this step (N,)

{

    int i = _i[0], j = $threadid;

    float xj, dxj, input;

    input = 0.0;
    for (int idx=0; idx<$N; idx++)
        input += G[j*$N + idx]*hist[$N*wrap(i - 1 - idelays[j*$N + idx]) + idx];

     xj = hist[$N*wrap(i - 1) + j];
    dxj = $dt*(  (xj - 5.0*pow((float)xj, 3.0f))/5.0 + $k*input/$N + randn[j]/5.0 );

    xout[j] = xj + dxj;

}

__global__ void update_hist(int * _i, // current step
                            float * __restrict__ hist, // history array
                            float * __restrict__ x)    // current state

{

    int i = _i[0], j = $threadid;
    hist[$N*wrap(i) + j] = x[j];

}

