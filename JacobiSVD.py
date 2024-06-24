import numpy as np
from scipy import linalg
import math
import time

def JacobiSVD(A, tol=1.0e-16, compute='USV', mode='real'):
    '''
    function to compute the SVD of a matrix A using the Jacobi algorithm
    input: matrix A, tol tolerance for iteration
    output: sg_values array of singular values of A, U and V_t unitary such that
    USV_t = A, where S = diag(sg_values)
    '''
    m, n = np.shape(A)
    norm_A = np.linalg.norm(A)
    Q, R, P = linalg.qr(A,pivoting=True)
    rank = int(sum(abs(np.diag(R)) > tol*norm_A))
    if rank == n:
        pass
    # R_1, P_1 = linalg.qr(R[:rank].T,mode='r',pivoting=True)
    X = R[:rank].T
    x_norms = np.zeros(rank)
    for i in range(rank):
        x_norms[i] = np.linalg.norm(X[:,i])
    S, sg_values = JacobiSweep(X, x_norms)
    return 1,sg_values, 1

def JacobiSweep(X, x_norms, skips=True, eps=1.0e-16):
    
    m, n = np.shape(X)
    tol = eps*m
    first = True
    s = 0
    M = np.linalg.norm(X)
    l = 0
    b = 10
    N_bl = math.ceil(n/b)

    while first or s > tol:

        s = 0
        l += 1
        first = False

        for r in range(N_bl):
            i = r*b
            for d in range(3):
                j = i + d*b
                for p in range(j, min(j+b-1,n-1)):
                    skipped = 0
                    for q in range(p+1,min(j+b,n)):
                        if x_norms[p] < x_norms[q]:
                            temp_xp = np.copy(X[:,p])
                            temp_xq = np.copy(X[:,q])
                            norm_p = x_norms[p]
                            norm_q = x_norms[q]
                            X[:,p] = temp_xq
                            X[:,q] = temp_xp
                            x_norms[p] = norm_q
                            x_norms[q] = norm_p
                
                        angle, illcond = stable_angle(x_norms[p], x_norms[q], X[:,p], X[:,q])
                        if abs(angle) > tol:
                            cos, tan = determine_jacobi(x_norms[p], x_norms[q], angle)
                            temp_xp = np.copy(X[:,p])
                            temp_xq = np.copy(X[:,q])
                            norm_p = x_norms[p]
                            norm_q = x_norms[q]
                            X[:,p] = (temp_xp-tan*temp_xq)*cos
                            X[:,q] = (temp_xq+tan*temp_xp)*cos
                            x_norms[p] = norm_p*np.sqrt(1-tan*angle*(norm_q/norm_p))
                            x_norms[q] = norm_q*np.sqrt(max(0,1+tan*angle*norm_p/norm_q))
                        else:
                            skipped += 1

                        if skips and skipped == 2:
                            skipped = 0
                            break

                        if abs(angle) > s:
                            s = abs(angle)

            for c in range(r+1,N_bl):
                j = c*b
                for p in range(i, min(i+b,n)):
                    skipped = 0
                    for q in range(j,min(j+b,n)):
                        if x_norms[p] < x_norms[q]:
                            temp_xp = np.copy(X[:,p])
                            temp_xq = np.copy(X[:,q])
                            norm_p = x_norms[p]
                            norm_q = x_norms[q]
                            X[:,p] = temp_xq
                            X[:,q] = temp_xp
                            x_norms[p] = norm_q
                            x_norms[q] = norm_p
                
                        angle, illcond = stable_angle(x_norms[p], x_norms[q], X[:,p], X[:,q])
                        if abs(angle) > tol:
                            cos, tan = determine_jacobi(x_norms[p], x_norms[q], angle)
                            temp_xp = np.copy(X[:,p])
                            temp_xq = np.copy(X[:,q])
                            norm_p = x_norms[p]
                            norm_q = x_norms[q]
                            X[:,p] = (temp_xp-tan*temp_xq)*cos
                            X[:,q] = (temp_xq+tan*temp_xp)*cos
                            x_norms[p] = norm_p*np.sqrt(1-tan*angle*(norm_q/norm_p))
                            x_norms[q] = norm_q*np.sqrt(max(0,1+tan*angle*norm_p/norm_q))
                        else:
                            skipped += 1

                        if skips and skipped == 2:
                            skipped = 0
                            break

                        if abs(angle) > s:
                            s = abs(angle)

        if l == 3:
            skips = False

    return X, x_norms

def FirstSweep(X, x_norms, eps=1.0e-16):
    
    m, n = np.shape(X)
    tol = m*eps

    if n <= 10:
        for i in range(n-1):
            for j in range(i+1,n):

                if x_norms[i] < x_norms[j]:
                    temp_xp = np.copy(X[:,i])
                    temp_xq = np.copy(X[:,j])
                    norm_p = x_norms[i]
                    norm_q = x_norms[j]
                    X[:,i] = temp_xq
                    X[:,j] = temp_xp
                    x_norms[i] = norm_q
                    x_norms[j] = norm_p
                
                angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
                if abs(angle) > tol:
                    cos, tan = determine_jacobi(x_norms[i], x_norms[j], angle)
                    temp_xp = np.copy(X[:,i])
                    temp_xq = np.copy(X[:,j])
                    norm_p = x_norms[i]
                    norm_q = x_norms[j]
                    X[:,i] = (temp_xp-tan*temp_xq)*cos
                    X[:,j] = (temp_xq+tan*temp_xp)*cos
                    x_norms[i] = norm_p*np.sqrt(1-tan*angle*(norm_q/norm_p))
                    x_norms[j] = norm_q*np.sqrt(max(0,1+tan*angle*norm_p/norm_q))
                
        return X, x_norms
    
    k = n//2
    X_1, x_1_norms = FirstSweep(X[:,:k], x_norms[:k])
    X_2, x_2_norms = FirstSweep(X[k:,k:], x_norms[k:])
    X[:,:k] = X_1
    X[k:,k:] = X_2
    x_norms[:k] = x_1_norms
    x_norms[k:] = x_2_norms

    for i in range(k):
        for j in range(k,n):
            if x_norms[i] < x_norms[j]:
                temp_xp = np.copy(X[:,i])
                temp_xq = np.copy(X[:,j])
                norm_p = x_norms[i]
                norm_q = x_norms[j]
                X[:,i] = temp_xq
                X[:,j] = temp_xp
                x_norms[i] = norm_q
                x_norms[j] = norm_p
                
            angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
            if abs(angle) > tol:
                cos, tan = determine_jacobi(x_norms[i], x_norms[j], angle)
                temp_xp = np.copy(X[:,i])
                temp_xq = np.copy(X[:,j])
                norm_p = x_norms[i]
                norm_q = x_norms[j]
                X[:,i] = (temp_xp-tan*temp_xq)*cos
                X[:,j] = (temp_xq+tan*temp_xp)*cos
                x_norms[i] = norm_p*np.sqrt(1-tan*angle*(norm_q/norm_p))
                x_norms[j] = norm_q*np.sqrt(max(0,1+tan*angle*norm_p/norm_q))

    for i in range(k-1):
        for j in range(i+1,k):   
            if x_norms[i] < x_norms[j]:
                temp_xp = np.copy(X[:,i])
                temp_xq = np.copy(X[:,j])
                norm_p = x_norms[i]
                norm_q = x_norms[j]
                X[:,i] = temp_xq
                X[:,j] = temp_xp
                x_norms[i] = norm_q
                x_norms[j] = norm_p
                
            angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
            if abs(angle) > tol:
                cos, tan = determine_jacobi(x_norms[i], x_norms[j], angle)
                temp_xp = np.copy(X[:,i])
                temp_xq = np.copy(X[:,j])
                norm_p = x_norms[i]
                norm_q = x_norms[j]
                X[:,i] = (temp_xp-tan*temp_xq)*cos
                X[:,j] = (temp_xq+tan*temp_xp)*cos
                x_norms[i] = norm_p*np.sqrt(1-tan*angle*(norm_q/norm_p))
                x_norms[j] = norm_q*np.sqrt(max(0,1+tan*angle*norm_p/norm_q))
                
    for i in range(k,n-1):
        for j in range(i+1,n):   
            if x_norms[i] < x_norms[j]:
                temp_xp = np.copy(X[:,i])
                temp_xq = np.copy(X[:,j])
                norm_p = x_norms[i]
                norm_q = x_norms[j]
                X[:,i] = temp_xq
                X[:,j] = temp_xp
                x_norms[i] = norm_q
                x_norms[j] = norm_p
                
            angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
            if abs(angle) > tol:
                cos, tan = determine_jacobi(x_norms[i], x_norms[j], angle)
                temp_xp = np.copy(X[:,i])
                temp_xq = np.copy(X[:,j])
                norm_p = x_norms[i]
                norm_q = x_norms[j]
                X[:,i] = (temp_xp-tan*temp_xq)*cos
                X[:,j] = (temp_xq+tan*temp_xp)*cos
                x_norms[i] = norm_p*np.sqrt(1-tan*angle*(norm_q/norm_p))
                x_norms[j] = norm_q*np.sqrt(max(0,1+tan*angle*norm_p/norm_q))

    return X, x_norms

def determine_jacobi(norm_ap, norm_aq, angle):
    
    cot = (norm_aq/norm_ap - norm_ap/norm_aq)/(2*angle)
    tan = np.sign(cot)/(abs(cot) + np.sqrt(1+cot**2))
    cos = 1/np.sqrt(1 + tan**2)

    return cos, tan

def stable_angle(norm_ap, norm_aq, ap, aq):

      if norm_aq >= 1:
        too_small = False
        too_big = (norm_ap >= 1.0e+308/norm_aq)
        illcond = (1.0e-307*norm_ap > norm_aq)
    else:
        too_big = False
        too_small = (norm_ap <= 1.0e-307/norm_aq)
        illcond = (norm_ap > norm_aq/1.0e-307)
    if too_big:
        norm_apq = (np.inner(ap/norm_ap,aq))/norm_aq
    elif too_small:
        norm_apq = (np.inner(aq/norm_aq,ap))/norm_ap
    else:
        norm_apq = (np.inner(ap,aq)/norm_aq)/norm_ap

    return norm_apq, illcond
