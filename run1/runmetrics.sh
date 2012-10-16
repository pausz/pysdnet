#!/bin/sh
export LD_LIBRARY_PATH=/usr/lib/atlas-base:/usr/local/bin:/usr/bin:/bin:/usr/lib64:/usr/lib
export PATH=/usr/local/bin:/usr/bin:/bin
python metric.py $1
