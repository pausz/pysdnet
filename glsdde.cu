/*
    file: glsdde.c
    auth: marmaduke <duke@eml.cc>

    CUDA kernel template for integrating an SDDE, with looping.

    Template params: threadid, N, horizon, k, dt, m, sync1, sync2, sync3

    Functions:

        - wrap() takes the step number and returns the corresponding step number
            assuming periodic boundaries.

        - step() takes the system state and updates it $m times
*/


inline __device__ int wrap(int i)

{
    if (i >= 0)
        return i % $horizon;
    else
        if (i == - $horizon)
            return 0;
        else
            return $horizon + (i % $horizon);
}


__global__ void steps(int i, // current step number/count
                      int * __restrict__ idelays, // delays in steps (N, N)
                      float * __restrict__ G, // coupling matrix (N, N)
                      float * __restrict__ hist, // history (horizon + 1, N)
                      float * __restrict__ randn // randnums for this step (N,)
                      float * __restrict__ out) // output

{
    int j = $threadid, i_end = i + $m;

    float xj, dxj, input;

    for (; i < i_end; i++)
    {
        input = 0.0;
        for (int idx=0; idx<$N; idx++)
            input += G[j*$N + idx]*hist[$N*wrap(i - 1 - idelays[j*$N + idx]) + idx];

         xj = hist[$N*wrap(i - 1) + j];
        dxj = $dt*(  (xj - 5.0*pow(xj, 3.0))/5.0 + $k*input/$N + randn[j]/5.0 );

        $sync1;
        hist[$N*wrap(i) + j] = xj + dxj;
        $sync2;
    }

    $sync3;
    xout[j] = hist[$N*wrap(i) + j];
}
