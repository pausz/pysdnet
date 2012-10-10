#!/bin/sh

cat > ./.temp-oar-job.sh <<EOF
#!/bin/sh

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/atlas-base
export PATH=/home/duke/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=/home/duke/pycuda/build/lib.linux-x86_64-2.7/:$PYTHONPATH

time python parsweep3d.py
EOF

chmod +x ./.temp-oar-job.sh

rm out err
oarsub -l nodes=1,walltime=144:00:00 -p "GPU='YES'" -O out -E err ./.temp-oar-job.sh




