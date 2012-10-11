from sklearn.decomposition import fastica
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import Isomap
import se
import time
tic = time.time()

if 1: # load data from disk, imports
       launches = load('launch-000030.npy').reshape((-1, 32, 101, 96, 2))

"""
if 1: # run analyses
    launch = launches[888]
"""

ioff()
for launch_index, launch in enumerate(launches):

    print 'starting plot', launch_index

    print 'ensemble variance is ', launch.var()
    sampens = se.se(launch.reshape((-1, 96*2)).mean(axis=1), taus=r_[1:10], qse=False)
    print 'ensemble sampl entropy is', sampens[0]
    launch -= launch.mean(axis=0).mean(axis=0)
    n_icomp = 6
    ica = fastica(launch.reshape((-1, 96*2)), n_icomp)
    pca = svd(launch.reshape((-1, 96*2)))
    nnn = 10
    """
    lle = LLE(nnn, n_components=2, method='standard').fit_transform(launch[:,25:].reshape((-1, 96*2)))
    print 'lle finished'
    lle1 = LLE(nnn, n_components=2, method='modified', eigen_solver='dense').fit_transform(launch[:,25:].reshape((-1, 96*2)))
    print 'lle1 finished'
    lle2 = LLE(nnn, n_components=2, method='hessian', eigen_solver='dense').fit_transform(launch[:,25:].reshape((-1, 96*2)))
    print 'lle2 finished'
    """
    lle3 = LLE(nnn, n_components=2, method='ltsa', eigen_solver='dense').fit_transform(launch[:,25:].reshape((-1, 96*2)))
    print 'lle3 finished'
    iso  = Isomap(n_components=2).fit_transform(launch[:,25:].reshape((-1, 96*2)))
    print 'isomap finished'

#if 1: # do plotting

    from matplotlib import rcParams
    rcParams['font.size'] = 9
    info = lambda a,b,c: (title(a), xlabel(b), ylabel(c), grid(1))

    figure(figsize=(10, 10))
    clf()
    ts = r_[0:500:launches.shape[2]*1j]

    subplot(331)
    [plot(ts, 2*l+i, 'k', alpha=0.3) for i, l in enumerate(launch[0, :,:, 0].T)];
    info('x_i(t) for 1 trial', 'time (ms)', 'index of node')

    subplot(332)
    [plot(ts, trial[:, :, 0].mean(axis=1), 'k', alpha=0.1) for trial in launch];
    info('mean field, all trials', 'time (ms)', 'amplitude')

    subplot(334)
    plot(cumsum(pca[1]**2/sum(pca[1]**2))[:50])
    info('cumul var acctd, n princ compnts', 'no. principle components', 'cumulative variance')

    subplot(335)
    for trial in launch.reshape((32, -1, 96*2)):
        accum=0
        for i, comp in enumerate(dot(pca[2][:n_icomp], trial.T)):
            plot(ts, comp+accum, 'k', alpha=0.1)
            accum += comp.ptp()*1.2
    grid(1)
    info('proj trials onto %d princ compnts' % n_icomp, 'time (ms)', '')

    subplot(337)
    yscl = ica[2].ptp()
    for i, comp in enumerate(rollaxis(ica[2].reshape((32, -1, ica[1].shape[0])), 2)):
        [plot(ts, 3*trial/yscl + i, 'k', alpha=0.1) for trial in comp];
    grid(1)
    info('time series of %d ind. compnts' % n_icomp, 'time (ms)', '')

    subplot(338)
    plot(r_[1:10], sampens, 'k*-')
    info('sample entropy', 'scale factor, tau', 'se')

    alpha = 0.2

    subplot(333)
    fs=fft.fftfreq(len(ts), d=(ts[1]-ts[0])/1000.)[:len(ts)/2]

    for capteur in rollaxis(launch[:, :, :, 0], 2): # -> 96, (32, 101)
        spec = fft.fft(capteur.mean(axis=0))[:len(ts)/2]
        loglog(fs, abs(spec)/fs, 'k', alpha=0.1)

    """
    for trial in launch: # (32, 101, 96, 2)
        spec = fft.fft(trial[:, :, 0].mean(axis=1))[:len(ts)/2]
        loglog(fs, abs(spec)/fs, 'k', alpha=0.2)
    """
    info('per node spectrum density', 'freq (Hz)', 'power density')

    subplot(336)
    for trial in lle3.reshape((32, -1, 2)):
        plot(trial[:, 0], trial[:, 1], alpha=alpha)
    avg, std = lle3.mean(axis=0), lle3.std(axis=0)
    axis([avg[0]-std[0], avg[0]+std[0], avg[1]-std[1], avg[1]+std[1]])
    title('tangent space alignment LLE')

    subplot(339)
    for trial in iso.reshape((32, -1, 2)):
        plot(trial[:, 0], trial[:, 1], 'x-', alpha=alpha)
    title('isomap')

    tight_layout()

    savefig('launch-000030-%04d.png' % launch_index, dpi=100)

print 'total script time', time.time() - tic, 's'
