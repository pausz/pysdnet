pysdnet
=======

what
----

This is a library to integrate stochastic delayed differential equations
that resemble more or less networks. A pure NumPy integrator is available,
and a more efficient C version runs quite fine as well. A CUDA template is
present, and I'm currently waiting to test and optimize it, but previous tests
indicate the (Py)CUDA version will shine for systems whose simulation speed, on the
CPU, is limited by memory bandwidth.

Mathematically, I use fixed-step size Euler and additive noise. If the effort
is justified, a few things could be added without much difficulty:

- multiplicative noise
- Heun or higher order methods

If the system is small and deterministic, it's nice to have sophisticated
algorithms checking for delay-induced discontinuity propagation, so use dde23
or pydelay. Here, I ignore these problems.

Hopefully once the densely coupled version is running nicely on PyCUDA, I can
tackle a sparse version; some tests of sparse multiplication is in sparse branch,
appears to do well on gpu.

how
---

- Python 2.6+, NumPy
- C compiler for C version

Cuda version requires

- PyCUDA
- Nvidia graphics card


