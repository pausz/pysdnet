# coding: utf-8

def subseq(A, n=3):
    return A[tile(r_[0:n], (A.size-n+1, 1)) + r_[0:A.size-n+1][:, newaxis]]

def se(A, n=3, r=0.1):
    a1, a2, a1_count, a2_count = subseq(A, n), subseq(A, n+1), 0, 0
    for i in range(a1.shape[0] - 1):
        a1_count += (abs(tile(a1[i], (a1.shape[0]-i-1, 1)) - a1[i+1:]) < r).all(axis=1).sum()
    for i in range(a2.shape[0] - 1):
        a2_count += (abs(tile(a2[i], (a2.shape[0]-i-1, 1)) - a2[i+1:]) < r).all(axis=1).sum()
    return -log(a2_count*1.0/a1_count)

