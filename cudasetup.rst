Notes on setting up CUDA and PyCUDA on Debian testing
=====================================================

abstract
--------

- install drivers, cuda toolkit, gpu sdk
- install boost-python, python-dev

All of the following was done on a fresh installation of Debian
testing (which right now means wheezy), from a root user terminal.

installing CUDA
---------------

This is easy, but a few things to note (examples assume default cuda path)

Kernel headers need to be installed to build the driver, but I'm not sure
how it works for the external GPUs.

You really do need

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$PATH

so put it in your .bashrc or whatever.

Debian testing comes with gcc-4.6, and it's not so easy to make the gpu sdk
use 4.5, even if you install it, so go to line 80 in

    /usr/local/cuda/include/host_config.h

and change the 5 at the end to a 7. This will allow us to use gcc-4.6 and
complete the gpu sdk build. NVIDIA doesn't want us to use gcc-4.6, but
unless and until we have serious bugs due to gcc-4.6, it's not worth using
4.5

Run some of the built programs to check they work and that the library paths
are all ok.


installing pycuda
-----------------

First,

    apt-get install libboost-all-dev python-dev

Make sure git is installed (apt-get install git), because we want the latest
development sources.  In bash, I grab all fresh packages from source with

    for pkg in pymetis pycuda pyublas
    do
        git clone http://git.tiker.net/trees/$pkg.git
    end

- pymetis is used for sparse matrix partitioning
- pyublas adds extra c++ / Python type converters

Next, we install the individual packages (as root user)

In the pyublas folder

    ./configure.py
    python setup.py build
    python setup.py install

Then change to pymetis, and do the same.

For pycuda,

    cd pycuda
    git submodule init
    git submodule update
    ./configure.py
    python setup.py build
    python setup.py install

quick test
----------

In a directory outside of the pycuda source folder, open python and do
the following quick test (from the pycuda tutorial)

    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy

    a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
    a_doubled = (2*a_gpu).get()
    print a_doubled
    print a_gpu

and normally everything should work just fine.

