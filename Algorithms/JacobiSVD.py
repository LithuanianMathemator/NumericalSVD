import numpy as np
from scipy import linalg
import math
import time
import scipy.linalg as sl

eps_sys = np.finfo(float).eps

def JacobiSVD(A, eps=eps_sys, compute='USV', tau_A=False, conditioning='QR', simple=False):
    '''
    function to compute the SVD of a matrix A using the Jacobi algorithm
    input: matrix A, tol tolerance for iteration
    output: sg_values array of singular values of A, U and V_t unitary such that
    USV_t = A, where S = diag(sg_values)
    '''
    m, n = np.shape(A)

    if simple:

        X = np.copy(A)
        x_norms = np.zeros(n)
        for i in range(n):
            x_norms[i] = np.linalg.norm(X[:,i])
        if compute == 'SV' or compute == 'USV':
            V = np.eye(n)
            acc = True
            S, sg_values, V = JacobiSweep(X, x_norms, eps, acc, V)
            V_t = V.T
        else:
            acc=False
            S, sg_values = JacobiSweep(X, x_norms, eps)

        for i in range(n):
            sg_values[i] = np.linalg.norm(S[:,i])

        if compute == 'US' or compute == 'USV':
            U = np.zeros((m,n))
            for i in range(n):
                U[:,i] = 1/sg_values[i]*S[:,i]

        if compute == 'S':
            return sg_values
        if compute == 'US':
            return U, sg_values
        if compute == 'SV':
            return sg_values, V_t
        if compute == 'USV':
            return U, sg_values, V_t

    if compute == 'S':

        R, P = linalg.qr(A, pivoting=True, mode='r')
        rank = determine_rank(R, eps)
        x_norms = np.zeros(rank)
        R_1, P_1 = linalg.qr(R[:rank].T,mode='r',pivoting=True)
        X = R_1[:rank,:rank].T
        
        for i in range(rank):
            x_norms[i] = np.linalg.norm(X[:,i])

        X, x_norms = FirstSweep(X, x_norms)
        S, sg_values = JacobiSweep(X, x_norms)

        for i in range(rank):
            sg_values[i] = np.linalg.norm(S[:,i])

        if rank < n:
            sg_values = np.concatenate((sg_values,np.zeros(n-rank)))

        return sg_values
    
    if compute == 'SV':

        R, P = linalg.qr(A,pivoting=True,mode='r')
        rank = determine_rank(R, eps)
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
        rank = determine_rank(R, eps)
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
        P_inv[P] = np.arange(P.size)
        rank = determine_rank(R, eps)

        if conditioning=='QR':
            Q_1, R_1 = linalg.qr(R[:rank].T)
            X = np.copy(R_1[:rank].T)

        if conditioning=='LQ':
            Q_1, R_1, P_1 = linalg.qr(R[:rank].T,pivoting=True)
            P_1_inv = np.zeros_like(P_1)
            P_1_inv[P_1] = np.arange(P_1.size)
            R_2 = linalg.qr(R_1[:rank].T,mode='r')[0]
            X = np.copy(R_2.T)

        if conditioning=='ACC':
            Q_1, R_1, P_1 = linalg.qr(R[:rank].T,pivoting=True)
            P_1_inv = np.zeros_like(P_1)
            P_1_inv[P_1] = np.arange(P_1.size)
            Q_2, R_2 = linalg.qr(R_1[:rank].T)
            X = np.copy(R_2.T)
        
        # Jacobi sweeps
        x_norms = np.zeros(rank)
        for i in range(rank):
            x_norms[i] = np.linalg.norm(X[:,i])
        if conditioning=='ACC':
            acc = True
            V_x = np.eye(rank)
            X, x_norms, V_x = FirstSweep(X, x_norms, eps, acc, V_x)
            S, sg_values, V_x = JacobiSweep(X, x_norms, eps, acc, V_x)
        else:
            X, x_norms = FirstSweep(X, x_norms)
            S, sg_values = JacobiSweep(X, x_norms)
        if rank < n:
            sg_values = np.concatenate((sg_values,np.zeros(n-rank)))
        U_x = np.zeros((rank,rank))
        for i in range(rank):
            U_x[:,i] = 1/sg_values[i]*S[:,i]
        
        if conditioning=='QR':

            if rank == n:
                W = np.zeros((n,n))
                for i in range(n):
                    W[:,i] = linalg.solve_triangular(R[:rank,:rank], S[:,i])
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
                W[:,i] = linalg.solve_triangular(R_1[:rank], S[:,i])
            V = np.zeros((n,n))
            V[:,:rank] = Q_1[:,:rank]@U_x
            V[:,rank:] = Q_1[:,rank:]
            V_t = V.T[:,P_inv]

            U = np.zeros((m,m))
            U[:,:rank] = Q[:,:rank]@W[P_1_inv,:]
            U[:,rank:] = Q[:,rank:]

        if conditioning=='ACC':
            
            V = np.zeros((n,n))
            V[:,:rank] = Q_1[:,:rank]@U_x
            V[:,rank:] = Q_1[:,rank:]
            V_t = V.T[:,P_inv]

            U = np.zeros((m,m))
            U[:,:rank] = Q[:,:rank]@Q_2[P_1_inv,:]@V_x
            U[:,rank:] = Q[:,rank:]
        
        return U, sg_values, V_t

def JacobiSweep(X, x_norms, eps=1.0e-16, acc=False, V=None):
    '''
    Jacobi sweep until convergence
    Input: matrix X and its column norms x_norms
    '''
    m, n = np.shape(X)
    tol = n*eps
    first = True
    s = 0
    maxiter = 100
    it = 0

    while first or s > tol:

        s = 0
        first = False

        for i in range(n-1):

            k = np.argmax(x_norms[i:])

            # de Rijks pivoting
            if k != 0:

                norm_p = x_norms[i]
                norm_q = x_norms[i+k]
                x_norms[i] = norm_q
                x_norms[i+k] = norm_p
                temp_xp = np.copy(X[:,i])
                temp_xq = np.copy(X[:,i+k])
                X[:,i] = temp_xq
                X[:,i+k] = temp_xp
                if acc:
                    temp_vp = np.copy(V[:,i])
                    temp_vq = np.copy(V[:,i+k])
                    V[:,i] = temp_vq
                    V[:,i+k] = temp_vp

            for j in range(i+1,n):
                    
                angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
                if abs(angle) > tol:
                    JacobiRotation(X, x_norms, i, j, angle, illcond, acc, V)

                if abs(angle) > s:
                    s = abs(angle)
        
        it += 1

        if it >= maxiter:
            break
    
    if acc:
        return X, x_norms, V
    else:
        return X, x_norms

def FirstSweep(X, x_norms, eps=1.0e-16, acc=False, V=None):
    '''
    First Jacobi Sweep on lower triangular matrix using the matrix structure
    Input: lower triangular matrix X and its column norms x_norms
    '''
    m, n = np.shape(X)
    tol = m*eps

    if n <= 10:
        for i in range(n-1):

            k = np.argmax(x_norms[i:])

            # de Rijks pivoting
            if k != 0:

                norm_p = x_norms[i]
                norm_q = x_norms[i+k]
                x_norms[i] = norm_q
                x_norms[i+k] = norm_p
                temp_xp = np.copy(X[:,i])
                temp_xq = np.copy(X[:,i+k])
                X[:,i] = temp_xq
                X[:,i+k] = temp_xp
                if acc:
                    temp_vp = np.copy(V[:,i])
                    temp_vq = np.copy(V[:,i+k])
                    V[:,i] = temp_vq
                    V[:,i+k] = temp_vp

            for j in range(i+1,n):
                    
                angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
                if abs(angle) > tol:
                    JacobiRotation(X, x_norms, i, j, angle, illcond, acc, V)

        if acc:
            return X, x_norms, V
        else:
            return X, x_norms
    
    k = n//2
    if not acc:
        X_1, x_1_norms = FirstSweep(X[:,:k], x_norms[:k], eps=eps)
        X_2, x_2_norms = FirstSweep(X[k:,k:], x_norms[k:], eps=eps)
        X[:,:k] = X_1
        X[k:,k:] = X_2
        x_norms[:k] = x_1_norms
        x_norms[k:] = x_2_norms
    else:
        X_1, x_1_norms, V_1 = FirstSweep(X[:,:k], x_norms[:k], eps=eps, acc=True, V=V[:,:k])
        X_2, x_2_norms, V_2 = FirstSweep(X[k:,k:], x_norms[k:], eps=eps, acc=True, V=V[:,k:])
        X[:,:k] = X_1
        X[k:,k:] = X_2
        x_norms[:k] = x_1_norms
        x_norms[k:] = x_2_norms
        V[:,:k] = V_1
        V[:,k:] = V_2

    for i in range(k):

        l = np.argmax(x_norms[i:])

        # de Rijks pivoting
        if l != 0:

            norm_p = x_norms[i]
            norm_q = x_norms[i+l]
            x_norms[i] = norm_q
            x_norms[i+l] = norm_p
            temp_xp = np.copy(X[:,i])
            temp_xq = np.copy(X[:,i+l])
            X[:,i] = temp_xq
            X[:,i+l] = temp_xp
            if acc:
                temp_vp = np.copy(V[:,i])
                temp_vq = np.copy(V[:,i+l])
                V[:,i] = temp_vq
                V[:,i+l] = temp_vp

        for j in range(k,n):

            angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
            if abs(angle) > tol:
                JacobiRotation(X, x_norms, i, j, angle, illcond, acc, V)

    for i in range(k-1):

        l = np.argmax(x_norms[i:])

        # de Rijks pivoting
        if l != 0:

            norm_p = x_norms[i]
            norm_q = x_norms[i+l]
            x_norms[i] = norm_q
            x_norms[i+l] = norm_p
            temp_xp = np.copy(X[:,i])
            temp_xq = np.copy(X[:,i+l])
            X[:,i] = temp_xq
            X[:,i+l] = temp_xp
            if acc:
                temp_vp = np.copy(V[:,i])
                temp_vq = np.copy(V[:,i+l])
                V[:,i] = temp_vq
                V[:,i+l] = temp_vp

        for j in range(i+1,k):   
                
            angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
            if abs(angle) > tol:
                JacobiRotation(X, x_norms, i, j, angle, illcond, acc, V)

    for i in range(k,n-1):
        l = np.argmax(x_norms[i:])

        # de Rijks pivoting
        if l != 0:

            norm_p = x_norms[i]
            norm_q = x_norms[i+l]
            x_norms[i] = norm_q
            x_norms[i+l] = norm_p
            temp_xp = np.copy(X[:,i])
            temp_xq = np.copy(X[:,i+l])
            X[:,i] = temp_xq
            X[:,i+l] = temp_xp
            if acc:
                temp_vp = np.copy(V[:,i])
                temp_vq = np.copy(V[:,i+l])
                V[:,i] = temp_vq
                V[:,i+l] = temp_vp

        for j in range(i+1,n):   
                
            angle, illcond = stable_angle(x_norms[i], x_norms[j], X[:,i], X[:,j])
            if abs(angle) > tol:
                JacobiRotation(X, x_norms, i, j, angle, illcond, acc, V)

    if acc:
        return X, x_norms, V
    else:
        return X, x_norms

def determine_jacobi(norm_ap, norm_aq, angle):
    '''
    Computation of cotangent, tangent and cosine for rotation
    '''
    
    cot = (norm_aq/norm_ap - norm_ap/norm_aq)/(2*angle)
    tan = np.sign(cot)/(abs(cot) + np.sqrt(1+cot**2))
    cos = 1/np.sqrt(1 + tan**2)

    return cos, tan

def stable_angle(norm_ap, norm_aq, ap, aq):
    '''
    Stable computation of cosine between ap and aq
    Input: vectors ap and aq and their norms norm_ap and norm_aq
    Output: cosine norm_apq and boolean illcond to detect ill-conditioning
    '''
    m = np.size(ap)
    eps = np.finfo(float).eps
    nu = np.finfo(float).smallest_normal/(eps*2*m)
    if norm_aq >= 1:
        too_small = False
        too_big = (norm_ap >= 1.0e+308/norm_aq)
        illcond = (nu*norm_ap > norm_aq)
    else:
        too_big = False
        too_small = (norm_ap <= 1.0e-307/norm_aq)
        illcond = (norm_ap > (norm_aq/nu))
    if too_big:
        norm_apq = (np.inner(ap/norm_ap,aq))/norm_aq
    elif too_small:
        norm_apq = (np.inner(aq/norm_aq,ap))/norm_ap
    else:
        norm_apq = (np.inner(ap,aq)/norm_aq)/norm_ap

    return norm_apq, illcond

def determine_rank(R, eps):
    '''
    Determine rank of triangular matrix R
    '''
    m, n = np.shape(R)
    return n
    for k in range(n-1):
        if abs(R[k,k])*eps >= abs(R[k+1,k+1]):
            return k+1
        
    return n

def CacheSweep(X, x_norms, eps=1.0e-16):
    '''
    Jacobi sweep with cache aware strategy
    Input: matrix X with its column norms x_norms
    '''
    m, n = np.shape(X)
    tol = n*eps
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

                        if abs(angle) > s:
                            s = abs(angle)

    return X, x_norms

def JacobiRotation(X, x_norms, i, j, angle, illcond, acc=False, V=None):
    '''
        Stable implementation of Jacobi rotations
        Input: matrix X with its column norms x_norms, columns i, j to rotate, angle is cosine
        of the columns, illcond is true if modified Jacobi rotation needs to be used
    '''
    if not illcond:
        cos, tan = determine_jacobi(x_norms[i], x_norms[j], angle)
        temp_xp = np.copy(X[:,i])
        temp_xq = np.copy(X[:,j])
        norm_p = x_norms[i]
        norm_q = x_norms[j]
        X[:,i] = (temp_xp-tan*temp_xq)*cos
        X[:,j] = (temp_xq+tan*temp_xp)*cos
        x_norms[i] = np.linalg.norm(X[:,i])
        x_norms[j] = np.linalg.norm(X[:,j])

        if acc:
            temp_vp = np.copy(V[:,i])
            temp_vq = np.copy(V[:,j])
            V[:,i] = (temp_vp-tan*temp_vq)*cos
            V[:,j] = (temp_vq+tan*temp_vp)*cos
    else:
        temp_xp = np.copy(X[:,i])
        temp_xq = np.copy(X[:,j])
        norm_p = x_norms[i]
        norm_q = x_norms[j]
        X[:,j] = (temp_xq/norm_q - angle*temp_xp/norm_p)*norm_q
        x_norms[j] = norm_q*np.sqrt(1-angle**2)
        if acc:
            temp_vp = np.copy(V[:,i])
            temp_vq = np.copy(V[:,j])
            V[:,j] = temp_vq - angle*temp_vp
