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