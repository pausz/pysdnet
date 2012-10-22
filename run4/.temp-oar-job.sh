#!/bin/sh

export PYTHONPATH=/home/duke/pysdnet/:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/atlas-base
export PATH=/home/duke/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=/home/duke/pycuda/build/lib.linux-x86_64-2.7/:

time python parsweep3d.py
