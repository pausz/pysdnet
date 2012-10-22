# coding: utf-8
if 0:
    get_ipython().system(u'ls -F --color ')
    npy = load('launch-000000.npy')
    npy.shape
    svds = []
    npy.reshape((-1, 32*401, 192)).shape
    for i, d in npy.reshape((-1, 32*401, 192)):
        svds.append(svd(d, full_matrices=0))
        print i
        
    for i, d in enumerate(npy.reshape((-1, 32*401, 192))):
        svds.append(svd(d, full_matrices=0))
        print i
        
    #for i, d in enumerate(npy.reshape((-1, 32*401, 192))):
        svds.append(svd(d, full_matrices=0))
        print i
    hist([where(cumsum(s**2/cum(s**2))>0.99)[0][0] for u, s, vt in svds], 100)
    hist([where(cumsum(s**2/sum(s**2))>0.99)[0][0] for u, s, vt in svds], 100)
    savefig('hist-nc.png')
    get_ipython().magic(u'pwd ')
    clf(); hist([where(cumsum(s**2/sum(s**2))>0.95)[0][0] for u, s, vt in svds], 100)
    get_ipython().magic(u'pwd ')
    savefig('hist-nc.png')
    len(svds)
    clf(); hist([where(cumsum(s**2/sum(s**2))>0.95)[0][0] for u, s, vt in svds], 100)

npy_ = npy.reshape((-1, 32*401, 192))

def estmeanerr(i):
    u, s, vt = svds[i]
    nc = where(cumsum(s**2/sum(s**2))>0.95)[0][0]
    recon = dot(u[:, :nc], dot(diag(s[:nc]), vt[:nc]))
    orig = npy_[i]
    mag = orig.ptp()
    return ((recon - orig)**2).mean()/mag
