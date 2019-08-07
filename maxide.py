# My numpy implementation of the Maxide MATLAB implementation. Gives identical outputs.

from numpy import *
from numpy.random import rand
from numpy.linalg import inv, multi_dot, norm, LinAlgError
from scipy.linalg import svd
from time import time

def xumm(A, B, Omega):
    return (dot(A, B))[Omega]

def mm(data, ind, shape):
    Z = zeros(shape)
    Z[ind] = data
    return Z


def sidesvd2Threshold(svdt1, svdt2, svdt3, L, delta):
    A = svdt1 - svdt2/L + svdt3/L
    try:
        (U, s, VT) = svd(A,
                full_matrices=False,
                compute_uv=True)
    except LinAlgError as e:
        print 'SVD failed to converge, switching to gesvd...'
        (U, s, VT) = svd(A,
                full_matrices=False,
                compute_uv=True,
                    lapack_driver='gesvd')
    s_thresh = maximum(s - delta/L, 0)
    S_thresh = diag(s_thresh)

    return multi_dot([U, S_thresh, VT])


def sideobjectCalc(B, delta, DiffL2):
    s = norm(B, 'nuc')
    return delta*s + DiffL2, s

def Qlpl(B, Z, L, qlpl1, qlpl2):
    return qlpl1 + L/2.*(norm((B-Z),'fro'))**2 + trace(dot((B-Z).T, qlpl2))                   

def Maxide(M_Omega, Omega_linear, A, B, max_iter = 10000, min_iter=1, delta = 1e-10, Z0=None, report_rate=500):
    
    M_Omega_linear = M_Omega.flatten()[Omega_linear]
    Omega_mask = False*ones(M_Omega.flatten().shape)
    Omega_mask[Omega_linear] = True
    Omega_mask = reshape(Omega_mask, M_Omega.shape).astype(bool)
    
    #Omega_mask = M_Omega != 0
    [n,m] = M_Omega.shape
    r_a = A.shape[1]
    r_b = B.shape[1]

    L = 1.
    gamma = 2.
    if Z0 is None:
        Z0 = zeros([r_a,r_b])
    else:
        print 'Using supplied Z0'
    Z = Z0.copy()
    alpha0 = 1.
    alpha = 1.
    
    i = 0
    convergence = dict()
    convergence[0] = 0
    
    svdt3 = multi_dot([A.T, M_Omega, B])
    AZ0BOmega = xumm(dot(A, Z0), B.T, Omega_mask)
    AZBOmega = AZ0BOmega.copy()
    
    t = time()
    
    try:
        for i in range(1, max_iter + 1):
            Y = Z + alpha*(1/alpha0 - 1)*(Z - Z0)
            Z0 = Z.copy()

            AYBOmega= (1 + alpha*(1/alpha0-1))*AZBOmega - (alpha*(1/alpha0 - 1))*AZ0BOmega
            svdt2 = multi_dot([A.T, mm(AYBOmega, Omega_mask, [n,m]), B])

            Z = sidesvd2Threshold(Y, svdt2, svdt3, L, delta)
            AZ0BOmega = AZBOmega.copy()
            AZBOmega = xumm(dot(A,Z), B.T, Omega_mask)

            qlpl1 = norm(AYBOmega - M_Omega_linear, 2)**2/2
            qlpl2 = svdt2 - svdt3
            DiffL2 = norm(AZBOmega - M_Omega_linear, 2)**2/2

            while DiffL2 > Qlpl(Z,Y,L,qlpl1,qlpl2):
                L = L*gamma
                Z = sidesvd2Threshold(Y, svdt2, svdt3, L, delta)
                AZBOmega = xumm(dot(A,Z), B.T, Omega_mask)
                DiffL2 = norm(AZBOmega - M_Omega_linear, 2)**2/2

            alpha0 = alpha
            alpha = (sqrt(alpha**4 + 4*alpha**2) - alpha**2)/2

            convergence[i], s1 = sideobjectCalc(Z, delta, DiffL2)

            if i > 1 and mod(i, report_rate) == 0:
                print '{0}: measured: {1:.3e} tol: {2:.3e} |Z|_*: {3:.3e} |AZBt - M|: {4:.3e}'.format(i, abs(convergence[i] - convergence[i-1]),                                                                                                   (1e-5)*convergence[i], delta*s1, DiffL2)

            if i > min_iter:
                if abs(convergence[i] - convergence[i-1]) < (1e-5)*convergence[i]:
                    print 'Ended at the {0}th iteration.'.format(i)
                    break
    except KeyboardInterrupt:
        pass
        
    print time() - t
    M_recover = multi_dot([A, Z, B.T])
    
    return M_recover, Z