# coding: utf-8
from numpy import *

def extract_end(data, nback=10):
	return data['Xs'][-nback:, :, 0, :].mean(axis=0).T.reshape((32, 32, 96))

final = empty((32, 32, 32, 96, 100))

for i, v in enumerate(2**r_[1:6:32j]):
    for j in range(100):
        npz = load('bistable/sim-ay-%0.2f-%d.npz' % (v, j))
        end = extract_end(npz)
        final[i, :, :, :, j] = end
        npz.close()
        print i, j, 1.*isnan(end).astype(int).sum()/end.size

save('bistable/final.npz', final)

