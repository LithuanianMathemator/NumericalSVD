import numpy as np
from scipy import linalg
import math
import time

def JacobiSVD(A, eps=1.0e-16, compute='USV', mode='real', tau_A=False, conditioning='QR'):
    '''
    function to compute the SVD of a matrix A using the Jacobi algorithm
    input: matrix A, tol tolerance for iteration
    output: sg_values array of singular values of A, U and V_t unitary such that
    USV_t = A, where S = diag(sg_values)
    '''
    m, n = np.shape(A)
    norm_A = np.linalg.norm(A)

    if compute == 'S':
        R, P = linalg.qr(A,pivoting=True,mode='r')
        rank = int(sum(abs(np.diag(R)) > eps*n*norm_A))
        x_norms = np.zeros(rank)
        R_1, P_1 = linalg.qr(R[:rank].T,mode='r',pivoting=True)
        X = R_1[:rank,:rank].T
        for i in range(rank):
            x_norms[i] = np.linalg.norm(X[:,i])
        X, x_norms = FirstSweep(X, x_norms)
        S, sg_values = JacobiSweep(X, x_norms)
        if rank < n:
            sg_values = np.concatenate((sg_values,np.zeros(n-rank)))
        return sg_values
    
    if compute == 'SV':
        R, P = linalg.qr(A,pivoting=True,mode='r')
        rank = int(sum(abs(np.diag(R)) > eps*n))
        X = R[:rank].T
        x_norms = np.zeros(rank)
        for i in range(rank):
            x_norms[i] = np.linalg.norm(X[:,i])
        X, x_norms = FirstSweep(X, x_norms)
        S, sg_values = JacobiSweep(X, x_norms)
        U_x = np.eye(n)
        for i in range(rank):
            U_x[:,i] = 1/sg_values[i]*S[:,i]
        V_t = U_x[P,:].T
        if rank < n:
            sg_values = np.concatenate((sg_values,np.zeros(n-rank)))
        return sg_values, V_t

    if compute == 'US':
        
        if tau_A:
            X = A
            x_norms = np.zeros(n)
            for i in range(n):
                x_norms[i] = np.linalg.norm(X[:,i])
            X, x_norms = FirstSweep(X, x_norms)
            S, sg_values = JacobiSweep(X, x_norms)
            U = np.zeros((n,n))
            for i in range(n):
                U[:,i] = 1/sg_values[i]*S[:,i]
            if rank < n:
                sg_values = np.concatenate((sg_values,np.zeros(n-rank)))
            return U, sg_values

        Q, R, P = linalg.qr(A,pivoting=True)
        rank = int(sum(abs(np.diag(R)) > eps*n))
        R_1, P_1 = linalg.qr(R[:rank].T,mode='r',pivoting=True)
        X = R_1[:rank,:rank].T
        x_norms = np.zeros(rank)
        for i in range(rank):
            x_norms[i] = np.linalg.norm(X[:,i])
        X, x_norms = FirstSweep(X, x_norms)
        S, sg_values = JacobiSweep(X, x_norms)
        if rank < n:
            sg_values = np.concatenate((sg_values,np.zeros(n-rank)))
        U_x = np.zeros((rank,rank))
        for i in range(rank):
            U_x[:,i] = 1/sg_values[i]*S[:,i]
        U = np.zeros(np.shape(Q))
        U[:,:rank] = Q[:,:rank]@U_x[P_1,:]
        U[:,rank:] = Q[:,rank:]
        return U, sg_values

    if compute == 'USV':
        Q, R, P = linalg.qr(A,pivoting=True)
        P_inv = np.zeros_like(P)
        P_inv[P] = np.arange(len(P))
        rank = int(sum(abs(np.diag(R)) > eps*n))
        if conditioning=='QR':
            Q_1, R_1 = linalg.qr(R[:rank].T)
            X = np.copy(R_1[:rank].T)
        if conditioning=='LQ':
            Q_1, R_1, P_1 = linalg.qr(R[:rank].T,pivoting=True)
            P_1_inv = np.zeros_like(P_1)
            P_1_inv[P_1] = np.arange(len(P_1))
            R_2 = linalg.qr(R_1[:rank].T,mode='r')[0]
            X = np.copy(R_2.T)
        x_norms = np.zeros(rank)
        for i in range(rank):
            x_norms[i] = np.linalg.norm(X[:,i])
        X, x_norms = FirstSweep(X, x_norms)
        S, sg_values = JacobiSweep(X, x_norms)
        U_x = np.zeros((rank,rank))
        for i in range(rank):
            U_x[:,i] = 1/sg_values[i]*S[:,i]
        if conditioning=='QR':
            if rank == n:
                W = np.zeros((n,n))
                for i in range(n):
                    W[:,i] = linalg.solve_triangular(R, S[:,i])
                V = W
            else:
                V_x = np.zeros((rank,rank))
                for i in range(n):
                    V_x[:,i] = linalg.solve_triangular(R_1[:rank].T, S[:,i],lower=True)
                V = np.zeros((n,n))
                V[:,:rank] = Q_1[:,:rank]@V_x
                V[:,rank:] = Q_1[:,rank:]
            V_t = V.T[:,P_inv]
            U = np.zeros((m,m))
            U[:,:rank] = Q[:,:rank]@U_x
            U[:,rank:] = Q[:,rank:]
        if conditioning=='LQ':
            W = np.zeros((rank,rank))
            for i in range(rank):
                W[:,i] = linalg.solve_triangular(R_1, S[:,i])
            V = np.zeros((n,n))
            V[:,:rank] = Q_1[:,:rank]@U_x
            V[:,rank:] = Q_1[:,rank:]
            V_t = V.T[:,P_inv]
            U = np.zeros((m,m))
            U[:,:rank] = Q[:,:rank]@W[P_1_inv,:]
            U[:,rank:] = Q[:,rank:]
        
        return U, sg_values, V_t

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

