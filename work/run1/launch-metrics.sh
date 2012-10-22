#!/bin/sh

rm metrics_array_params
touch metrics_array_params
n=`ls launch-*.npy | nl | tail -n1 | cut -f1`
for i in `seq $n`;
do
    echo $i >> metrics_array_params
done

