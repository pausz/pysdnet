# coding: utf-8
get_ipython().magic(u'edit scratch.py')
get_ipython().magic(u'edit scratch.py')
ml.close()
ml.close()
ml.close('all')
[ml.close(i) for i in range(10)]
close()
close()
npy.shape
npy_ = npy.reshape((-1, 32, 96, 2))
npy_.shape
npy_ = npy[:, :, :, 0].reshape((-1, 32, 101, 96))
npy_
npy.size
npy.shape
npy_.shape
icas = [FastICA(n_components=5) for i in range(10)]
for ica, data in zip(icas, npy_):
    ica.fit(data.reshape((-1, 96)))
    
dot(icas[0].unmixing_matrix_, icas[1].unmixing_matrix_.T)
figure(); imshow(icas[0].unmixing_matrix_)
figure(); imshow(icas[1].unmixing_matrix_)
icas = [FastICA(n_components=6) for i in range(10)]
npy_ = npy.reshape((-1, 32, 101, 96*2))
for ica, data in zip(icas, npy_):
    ica.fit(data.reshape((-1, 96*2)))
    
dot(icas[0].unmixing_matrix_, icas[1].unmixing_matrix_.T)
seed
seed(32); rand()
seed(32); rand()
for ica, data in zip(icas, npy_):
    seed(32); ica.fit(data.reshape((-1, 96)))
    
icas = [FastICA(n_components=6) for i in range(10)]
for ica, data in zip(icas, npy_):
    seed(32); ica.fit(data.reshape((-1, 96)))
    
dot(icas[0].unmixing_matrix_, icas[1].unmixing_matrix_.T)
figure(); hist(dot(icas[0].unmixing_matrix_, icas[1].unmixing_matrix_.T).flat, 100)
figure(); hist(concatenate([dot(icas[i].unmixing_matrix_, icas[i+1].unmixing_matrix_.T).flat for i in range(len(icas)-1)]), 100)
clf(); hist(concatenate([dot(icas[i].unmixing_matrix_, icas[i+1].unmixing_matrix_.T).flat for i in range(len(icas)-1)]), 15)
icas = [FastICA(n_components=6) for i in range(len(npy_))]
for ica, data in zip(icas, npy_):
    seed(32); ica.fit(data.reshape((-1, 96)))
    
clf(); hist(concatenate([dot(icas[i].unmixing_matrix_, icas[i+1].unmixing_matrix_.T).flat for i in range(len(icas)-1)]), 15)
clf(); hist(concatenate([dot(icas[i].unmixing_matrix_, icas[i+1].unmixing_matrix_.T).flat for i in range(len(icas)-1)]), 50)
get_ipython().magic(u'hist')
svd
pcas = []
for data in npy_:
    pcas.append(svd(data.reshape((-1, 96)))[2][:5].copy())
    print data.flat[0]
    
for data in npy_:
    pcas.append(svd(data.reshape((-1, 96)))[2][:5].copy(), full_matrices=0)
    print data.flat[0]
    
for data in npy_:
    pcas.append(svd(data.reshape((-1, 96)))[2][:5].copy(), full_matrices=0))
    print data.flat[0]
    
for data in npy_:
    pcas.append(svd(data.reshape((-1, 96)), full_matrices=0)[2][:5].copy()))
    print data.flat[0]
    
for data in npy_:
    pcas.append(svd(data.reshape((-1, 96)), full_matrices=0)[2][:5].copy())
    print data.flat[0]
    
pcas[0]
pcas[0].shape
dot(pcas[0], pcas[1].T)
dot(pcas[0], pcas[1].T).shape
figure(); hist(concatenate([dot(pcas[i], icas[i+1].T).flat for i in range(len(pcas)-1)]), 50)
figure(); hist(concatenate([dot(pcas[i], pcas[i+1].T).flat for i in range(len(pcas)-1)]), 50)
get_ipython().magic(u'save pca-vs-ica-pattern-dot-products.py 1-53')
close('all')
ml.figure()
len(pcas)
len(icas)
concatenate(map(lambda ica: ica.unmixing_matrix_, icas)).shape
concatenate(map(lambda ica: ica.unmixing_matrix_, icas)))
ica_unmix_svd = svd(concatenate(map(lambda ica: ica.unmixing_matrix_, icas)))
ica_unmix_svd = svd(concatenate(map(lambda ica: ica.unmixing_matrix_, icas)), full_matrices=0)
ica_unmix_svd[1][:10]
pca_unmix_svd = svd(concatenate(map(lambda pca: pca, pcas)), full_matrices=0)
pca_unmix_svd[1][:10]
concatenate(pcas).shape
plot(pca_unmix_svd[1][:100])
plot(ica_unmix_svd[1][:100])
pss = pca_unmix_svd[1]
iss = ica_unmix_svd[1]
clf()
plot(cumsum(pss**2/sum(pss**2))[:100])
plot(cumsum(iss**2/sum(iss**2))[:100])
grid(1)
close()
pcas[0].shape
cx
ml.points3d(cx, cy, cz, pcas[0][0]
)
ml.points3d(cx, cy, cz, pcas[0][1])
pca_unmix_svd
pca_unmix_svd[2].shape
pca_unmix_svd[2][:2].shape
dot(pca_unmix_svd[2][:2], pcas[0])
dot(pca_unmix_svd[2][:2], pcas[0].T)
dot(pca_unmix_svd[2][:2], pcas[0][0])
concatenate(pcas).shape
dot(pca_unmix_svd[2][:2], concatenate(pcas))
dot(pca_unmix_svd[2][:2], concatenate(pcas).t)
dot(pca_unmix_svd[2][:2], concatenate(pcas).T).shape
dot(pca_unmix_svd[2][:2], concatenate(pcas).T))
plot(*dot(pca_unmix_svd[2][:2], concatenate(pcas).T))
plot(*dot(pca_unmix_svd[2][:2], concatenate(pcas).T), '.')
px, py = dot(pca_unmix_svd[2][:2], concatenate(pcas).T)
plot(px, py, '.')
clf()
for x, y in zip(px, py):
    plot(x, y, '.')
    
clf()
for x, y in zip(px, py):
    plot(x, y, 'k.', alpha=0.2)
    
where(px[:10] < 0.5)
where(px[:10] < 0.)
wh = where(sqrt(px**2 + py**2) < 0.5)
len(wh)
wh
len(wh[0])
len(px)
pcas_ = concatenate(pcas)
pcas_.shape
pcas_ = concatenate(pcas)[wh]
pcas_.shape
ml.clf()
ml.points3d(cx, cy, cz, pcas_[0])
pcas_ = concatenate(pcas)
for px, py, net in zip(px[wh], py[wh], pcas_[wh])[:10]:
    print px, py
    
wh = where(sqrt(px**2 + py**2) > 0.5)
for px, py, net in zip(px[wh], py[wh], pcas_[wh])[:10]:
    print px, py
    
wh
len(px)
px
pxs, pys = dot(pca_unmix_svd[2][:2], concatenate(pcas).T)
for px, py, net in zip(pxs[wh], pys[wh], pcas_[wh])[:1000:100]:
    print px, py
    
wh
wh = where(sqrt(px**2 + py**2) > 0.5)
wh = where(sqrt(pxs**2 + pys**2) > 0.5)
for px, py, net in zip(pxs[wh], pys[wh], pcas_[wh])[:1000:100]:
    print px, py
    
for px, py, net in zip(pxs[wh], pys[wh], pcas_[wh])[:1000:100]:
    print px, py
    ml.figure()
    ml.points3d(cx, cy, cz, net)
    
config['dataset'].weights.shape
u, s, vt = svd(config['dataset'].weights)
figure(); plot(cumsum(s**2/sum(s**2))))
figure(); plot(cumsum(s**2/sum(s**2)))
for i in range(4):
    ml.figure()
    ml.points3d(cx, cy, cz, vt[i])
    
eig(config['dataset'].weights)
es, ev = eig(config['dataset'].weights)
figure(); plot(es)
for i in range(4):
    ml.figure()
    ml.points3d(cx, cy, cz, ev[i])
    
get_ipython().magic(u'save 96-networks.py 1-135')