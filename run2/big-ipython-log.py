from numpy.lib.format import open_memmap

n0 = open_memmap('launch-000000.npy')
n1 = open_memmap('launch-000001.npy')
n2 = open_memmap('launch-000002.npy')

n0_cond_ss = n0.reshape((-1, 32, 401, 192))[:, :, 200:].reshape((-1, 32*201, 192))
n1_cond_ss = n1.reshape((-1, 32, 401, 192))[:, :, 200:].reshape((-1, 32*201, 192))


# triple f here
figure(figsize=(15, 12))
ws = l9['dataset'].weights
ds = l9['dataset'].distances
idx = 32*10 + 21
cond = n9.reshape((-1, 32, 401, 96, 2))[idx, :, :]
ts = r_[0 : cond.shape[1]*2.5 : 1j*cond.shape[1]]
cond -= cond.reshape((-1, 192)).mean(axis=0).reshape((1, 1, 96, 2))
trial_svds = [svd(trial[:, :, 0], full_matrices=0) for trial in cond]
cond_svd = svd(cond[:, :, :, 0].reshape((-1, 96)), full_matrices=0)
for i, svdi, trial in zip(range(32), trial_svds, cond):
    subplot(335)
    x, y, z = svdi[1][:3][:, newaxis]*dot(svdi[2][:3], trial[:, :, 0].T)
    plot(x+z/3, y+z/3, 'k-', alpha=0.2)
    subplot(336)
    x, y, z = svdi[1][:3][:, newaxis]*dot(cond_svd[2][:3], trial[:, :, 0].T)
    plot(x+z/3, y+z/3, 'k-', alpha=0.3)
subplot(6,3,13)
hist(concatenate([abs(dot(svd1[2][:3], svd2[2][:3].T)).flat for i, svd1 in enumerate(trial_svds) for j, svd2 in enumerate(trial_svds) if not j==i]), 50)
xlim([0, 1.0])
subplot(3, 3, 4)
H, X, Y = histogram2d(ds.flat[:]/l9['vel'], ws.flat[:], range=[[1e-9, ts.max()], [1e-9, ws.max()]], bins=30)
pcolor(X, Y, H.T)
xlim([0, ts.ptp()])
for i, trial in enumerate(cond):
    subplot(331)
    for j, sig in enumerate(trial[:, where(ws.mean(axis=1) > (ws.mean() + ws.std()/1.75))[0], 0].T):
        plot(ts, 5*(sig/4 + j), 'k', alpha=0.1)
    subplot(332)
    xis = trial_svds[i][1][:3][:, newaxis]*dot(trial_svds[i][2][:3], trial[:, :, 0].T)
    [plot(ts, xi[:]/xis.ptp() + i, 'k', alpha=0.1) for xi, i in zip(xis, r_[:6])]
    yticks([0, 1, 2])
    subplot(333)
    xis = cond_svd[1][:3][:, newaxis]*dot(cond_svd[2][:3], trial[:, :, 0].T)
    [plot(ts, xi[:]/xis.ptp() + i, 'k', alpha=0.1) for xi, i in zip(xis, r_[:6])]
    yticks([0, 1, 2])
xs, ys = rollaxis(cond, 3)
ps = rollaxis(arctan2(ys - ys.mean(), xs - xs.mean()), 1).reshape((xs.shape[1], -1))
ps[1:] -= cumsum(dstack((ps[:-1]<0, ps[1:]>0)).all(axis=2), axis=0)*2*pi
ps = rollaxis(ps.reshape((-1, 32, 96)), 0, 2)
rps = ps - ps[:, :, xs.var(axis=1).mean(axis=0).argmax()][..., newaxis]
rps -= rps.mean(axis=2)[..., newaxis]
subplot(338)
vts = []
for rp in rps:
    u, s, vt = svd(rp, full_matrices=0)
    [plot(s_i*dot(vt_i, rp.T).T, c, alpha=0.1) for s_i, vt_i, c in zip(s, vt, 'krbg')]
    vts.append(vt[0].copy())
subplot(6,3,16)
hist([abs(dot(vts[i], vts[j])) for i in range(len(vts)) for j in range(len(vts)) if not i==j], 20)
xlim([0, 1])
subplot(339)
imshow(array(vts), aspect='auto')
colorbar()
suptitle('vel = %f, exc = %f, gsc = %f' % (l9['vel'], l9['exc'].flat[::32][456*0 + idx], l9['gsc'].flat[::32][456*0 + idx]))
