# coding: utf-8
if 2:
    from sklearn.decomposition import FastICA
    import cPickle as cp
    from enthought.mayavi import mlab as ml

launch, cond = 618, 27

with open("launch-%06d.metrics" % (launch,)) as fd: metrix = cp.load(fd)
with open("launch-%06d.pickle" % (launch,)) as fd: config = cp.load(fd)
cx, cy, cz = config['dataset'].centers.T
npy = load('launch-%06d.npy' % (launch,))

c27 = npy.reshape((-1, 32, 101, 96, 2))[cond]
ica27 = FastICA(n_components=6)
c27ics = ica27.fit(c27.reshape((-1, 96*2))).transform(c27.reshape((-1, 96*2)))

figure()
for i, sig in enumerate(c27ics.reshape((32, -1, 6))):
    for j, comp in enumerate(sig.T):
        plot(comp*10 + j, 'k', alpha=0.2)
        
for v in ica27.unmixing_matrix_:
    ml.figure()
    ml.points3d(cx, cy, cz, v.reshape((96, 2)).sum(axis=1))


