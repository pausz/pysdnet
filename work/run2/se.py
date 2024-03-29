import numpy

def se(y, m=2, r=None, qse=False, taus=1, info=False,
       tile=numpy.tile, na=numpy.newaxis, abs=numpy.abs, 
       log=numpy.log, r_=numpy.r_):
    """
    Computes (quadratic) sample entropy of a given input signal y, with
    embedding dimension n, and a match tolerance of r (ref 2). If an array
    of scale factors, taus, are given, the signal will be coarsened by each
    factor and a corresponding entropy will be computed (ref 1). If no value
    for r is given, it will be set to 0.15*y.std().

    Currently, the implementation is lazy and expects or coerces scale factors
    to integer values.

    With qse=True (default) the probability p is normalized for the value
    of r, giving the quadratic sample entropy, such that results from different
    values of r can be meaningfully compared (ref 2).

    ref 1: Costa, M., Goldberger, A. L., and Peng C.-K. (2002) Multiscale Entropy
            Analysis of Complex Physiologic Time Series. Phys Rev Lett 89 (6).

    ref 2: Lake, D. E. and Moorman, J. R. (2010) Accurate estimation of entropy
            in very short physiological time series. Am J Physiol Heart Circ Physiol


    To check that algorithm is working, look at ref 1, fig 1, and run 
    
    >>> se(randn(3*10000), r=.15, taus=r_[1:20], qse=False, m=2)

    marmaduke duke@eml.cc oct 9 2012

    """

    # if multiple scales given, run on each
    if type(taus) in (list, numpy.ndarray):
        return numpy.array([se(y, m=m, r=r, qse=qse, taus=int(tau)) for tau in taus])

    # helper function to reformat arrays for matching
    subseq = lambda y, n: y[tile(r_[0:n], (y.size-n+1, 1)) + r_[0:y.size-n+1][:, na]]

    # default value of r
    if r is None:
        r = 0.15*y.std()

    # if we have a scale factor, coarsen time series 
    if taus > 1:
        y = y[:y.shape[0]/taus*taus].reshape((-1, taus)).mean(axis=1)

    # compute embedding of signal dims m, m+1, initialize match counts to 0
    Y1 = subseq(y, m  )
    Y2 = subseq(y, m+1)
    c1 = 0
    c2 = 0

    # check for matches
    for i in range(Y1.shape[0] - 1):
        c1 += (abs(tile(Y1[i], (Y1.shape[0]-i-1, 1)) - Y1[i+1:]) < r).all(axis=1).sum()

    for i in range(Y2.shape[0] - 1):
        c2 += (abs(tile(Y2[i], (Y2.shape[0]-i-1, 1)) - Y2[i+1:]) < r).all(axis=1).sum()

    # ref 2, last paragraph of methods, warn inaccurate estimate
    if c2 < 5: 
        print "m+1 template match count is low, %d < 5" % c2

    p = c2*1.0/c1
    e = -log( p/(2*r) if qse else p )

    if info:
        return e, p, c2, c1
    else:
        return e

