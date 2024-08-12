import numpy as np
from scipy.linalg import lapack
import GolubKahan_SVD as gks

def GuEisenstat_SVD(alpha, beta, eps=1.0e-16, compute='USV', side=None):
    '''
    function to compute the SVD of a bidiagonal matrix using divide and conquer, the SVD
    gets computed by the Golub-Kahan algorithm once the matrix is small enough
    input: alpha = diagonal, beta = superdiagonal of bidiagonal matrix
    output: array of descending singular values, U, V_t matrices of left and right
    singular vectors, respectively
    '''
    n = np.size(alpha)
    alpha = np.copy(alpha)
    beta = np.copy(beta)
    if np.size(beta) != n:
        beta = np.append(beta, 0)

    if n < 25:

        if compute == 'US' or compute == 'USV':
            U = np.eye(n, dtype=float)
        if compute == 'S':
            V_t = np.zeros((n+1,2))
            V_t[n,1] = 1
            V_t[0,0] = 1
            W_t = np.zeros((n+1,2))
        else:
            V_t = np.eye(n+1, dtype=float)
            W_t = np.zeros((n+1,n+1))
        temp_alpha = np.zeros(n+1, dtype=float)
        temp_alpha[:n] = alpha
        if beta[n-1] != 0:
            gks.zero_column(temp_alpha, beta, V_t)
        alpha = temp_alpha[:n]
        beta = beta[:n-1]
        W_t[n,:] = V_t[n,:]

        if compute == 'US' or compute == 'USV':
            Q, sg_values, temp_W_t = gks.GolubKahan_SVD(alpha, beta, U, V_t[:n], eps = eps)
        else:
            sg_values, temp_W_t = gks.GolubKahan_SVD(alpha, beta, None, V_t[:n], eps=eps, compute='SV')

        W_t[:n,:] = temp_W_t

        if compute == 'US' or compute == 'USV':
            return Q, sg_values, W_t
        else:
            return sg_values, W_t
    
    k = n//2
    alpha_k = alpha[k-1]
    beta_k = beta[k-1]
    if compute == 'US' or compute == 'USV':
        Q1, sg_values_1, W1_t = GuEisenstat_SVD(alpha[:k-1],beta[:k-1], eps, compute)
        Q2, sg_values_2, W2_t = GuEisenstat_SVD(alpha[k:],beta[k:], eps, compute)
    elif compute == 'SV':
        sg_values_1, W1_t = GuEisenstat_SVD(alpha[:k-1],beta[:k-1], eps, compute)
        sg_values_2, W2_t = GuEisenstat_SVD(alpha[k:],beta[k:], eps, compute)
    else:
        sg_values_1, W1_t = GuEisenstat_SVD(alpha[:k-1],beta[:k-1], eps=eps, compute='S', side=1)
        sg_values_2, W2_t = GuEisenstat_SVD(alpha[k:],beta[k:], eps=eps, compute='S', side=2)

    if compute == 'S':
        c, s = gks.determine_givens(alpha_k*W1_t[k-1,1], beta_k*W2_t[n-k,0])
        z = np.zeros(n)
        z[0] = c*alpha_k*W1_t[k-1,1]-s*beta_k*W2_t[n-k,0]
        z[1:k] = alpha_k*W1_t[:k-1,1]
        z[k:] = beta_k*W2_t[:n-k,0]
        d = np.zeros(n)
        d[1:k] = sg_values_1
        d[k:] = sg_values_2
    else:   
        c, s = gks.determine_givens(alpha_k*W1_t[k-1,k-1], beta_k*W2_t[n-k,0])
        z = np.zeros(n)
        z[0] = c*alpha_k*W1_t[k-1,k-1]-s*beta_k*W2_t[n-k,0]
        z[1:k] = alpha_k*W1_t[:k-1,k-1]
        z[k:] = beta_k*W2_t[:n-k,0]
        d = np.zeros(n)
        d[1:k] = sg_values_1
        d[k:] = sg_values_2

    if compute == 'US' or compute == 'USV':
        U_m, sg_values, V_m_t = M_SVD(np.copy(z), np.copy(d), eps, compute)
    else:
        sg_values, V_m_t = M_SVD(np.copy(z), np.copy(d), eps, compute)
    
    if compute == 'US' or compute == 'USV':
        U = np.zeros((n,n))
        U[:k-1,:] = Q1@U_m[1:k,:]
        U[k-1,:] = U_m[0,:]
        U[k:,:] = Q2@U_m[k:]

    if compute == 'S':
        V_t = np.zeros((n+1,2))
        V_t[:n,0] = c*W1_t[k-1,0]*V_m_t[:,0] + V_m_t[:,1:k]@W1_t[:k-1,0]
        V_t[n,0] = s*W1_t[k-1,0]
        V_t[:n,1] = -s*W2_t[n-k,1]*V_m_t[:,0] + V_m_t[:,k:]@W2_t[:n-k,1]
        V_t[n,1] = c*W2_t[n-k,1]
    else:
        V_t = np.zeros((n+1,n+1))
        V_t[:n,:k] = V_m_t[:,0].reshape(n,1)@(c*W1_t[k-1,:].reshape(1,k))\
            + V_m_t[:,1:k]@W1_t[:k-1,:]
        V_t[:n,k:] = V_m_t[:,0].reshape(n,1)@(-s*W2_t[n-k,:].reshape(1,n-k+1))\
            + V_m_t[:,k:]@W2_t[:n-k,:]
        V_t[n,:k] = s*W1_t[k-1,:]
        V_t[n,k:] = c*W2_t[n-k,:]

    if compute == 'US' or compute == 'USV':
        return U, sg_values, V_t
    elif side in [1,2] or compute == 'SV':
        return sg_values, V_t
    else:
        return sg_values

def M_SVD(z, d, eps=1.0e-16, compute='USV'):
    '''
    function to compute the SVD of M where M has non-zero elements only in its first 
    row and diagonal
    input: z first row of M, d diagonal of M
    output: SVD of M
    '''
    n = np.size(d)
    G = np.diag(d)
    G[0,:] = z
    if compute == 'US' or compute == 'USV':
        U = np.eye(n, dtype=float)
        V_t = np.eye(n, dtype=float)
    else:
        V_t = np.eye(n, dtype=float)
        U = None
    z, d, U, V_t, a = deflate_M(z, d, U, V_t, eps, compute)
    sg_values_temp = np.zeros(n-a)
    diff = np.zeros((n-a,n-a))
    sums = np.zeros((n-a,n-a))
    for i in range(n-a):
        diff[i], sg_values_temp[i], sums[i], info = lapack.dlasd4(i, d[:n-a], z[:n-a])
        if info != 0:
            raise RuntimeError('The root finder did not converge.')
    U_temp, V_temp = compute_vec(d[:n-a], np.sign(z[:n-a]), diff, sums, compute)
    if compute == 'US' or compute == 'USV':
        U[:,:n-a] = U[:,:n-a]@U_temp
    V_t[:n-a,:] = V_temp@V_t[:n-a,:]
    sg_values = np.zeros(n)
    sg_values[:n-a] = sg_values_temp
    sg_values[n-a:] = d[n-a:]
    idx = np.flip(np.argsort(abs(sg_values)))
    sg_values = sg_values[idx]
    if compute == 'US' or compute == 'USV':
        U = (U.T[idx]).T
    V_t = V_t[idx]

    if compute == 'US' or compute == 'USV':
        return U, sg_values, V_t
    else:
        return sg_values, V_t

def deflate_M(z, d, U, V_t, eps=1.0e-16, compute='USV'):
    '''
    deflate the matrix M so d is ascending and no elements are in it more than once
    z is made large enough
    input: z first row of M, d diagonal of M, U and V_t are the identity
    output: updated z and d and U and V_t such that UM_newV_t is M, a such that the first
    n-a elements of d are ascending and unique
    '''
    a = 0
    n = np.size(d)
    
    # sort diagonal
    idx = np.argsort(d[1:])
    d[1:] = d[1:][idx]
    z[1:] = z[1:][idx]
    if compute == 'US' or compute == 'USV':
        U[:,1:] = U.T[1:][idx].T
        V_t[1:] = V_t[1:][idx]
    else:
        V_t[1:] = V_t[1:][idx]

    norm_M = d[n-1] + np.linalg.norm(z)
    tM = 4*eps*norm_M

    # deflate M if z has elements that are too small
    # if abs(z[0]) < tM:
    #     z[0] = tM

    idx = np.zeros(n-1, dtype=int)
    j = 0
    k = n-2
    for i in range(1,n):
        if abs(z[i]) < tM:
            idx[k] = i-1
            k -= 1
            z[i] = 0
            a += 1
        else:
            idx[j] = i-1
            j += 1

    d[1:] = d[1:][idx]
    z[1:] = z[1:][idx]
    if compute == 'US' or compute == 'USV':
        U[:,1:] = U.T[1:][idx].T
        V_t[1:] = V_t[1:][idx]
    else:
        V_t[1:] = V_t[1:][idx]

    # deflate M if elements in d are too small
    idx = np.zeros(j, dtype=int)
    l = j+1
    k = j-1
    j = 0
    for i in range(1,l):
        if abs(d[i]) < tM:
            idx[k] = i-1
            k -= 1
            c, s = gks.determine_givens(z[0], z[i])
            temp_1 = z[0]
            z[0] = c*temp_1 - s*z[i]
            z[i] = 0
            temp_1 = np.copy(V_t[0,:])
            temp_2 = np.copy(V_t[i,:])
            V_t[0,:] = c*temp_1-s*temp_2
            V_t[i,:] = s*temp_1+c*temp_2
            a += 1
        else:
            idx[j] = i-1
            j += 1

    # deflate M if elements in d are almost the same
    d[1:l] = d[1:l][idx]
    z[1:l] = z[1:l][idx]
    if compute == 'US' or compute == 'USV':
        U[:,1:l] = U.T[1:l][idx].T
        V_t[1:l] = V_t[1:l][idx]
    else:
        V_t[1:l] = V_t[1:l][idx]

    idx = np.zeros(j, dtype=int)
    l = j+1
    k = j-1
    j = 1
    i = 1
    m = 1
    if l-1 > 0:
        idx[0] = 0
    while i < l and i+m < l:
        if abs(d[i]-d[i+m]) < tM:
            d[i+m] = d[i]
            idx[k] = i+m-1
            k -= 1
            c, s = gks.determine_givens(z[i], z[i+m])
            z[i] = c*z[i] - s*z[i+m]
            z[i+m] = 0
            temp_1 = np.copy(V_t[i,:])
            temp_2 = np.copy(V_t[i+m,:])
            V_t[i,:] = c*temp_1-s*temp_2
            V_t[i+m,:] = s*temp_1+c*temp_2
            if compute == 'US' or compute == 'USV':
                temp_1 = np.copy(U[:,i])
                temp_2 = np.copy(U[:,i+m])
                U[:,i] = c*temp_1-s*temp_2
                U[:,i+m] = s*temp_1+c*temp_2
            m += 1
            a += 1
        else:
            idx[j] = i+m-1
            i = i+m
            j += 1
            m = 1

    d[1:l] = d[1:l][idx]
    z[1:l] = z[1:l][idx]
    if compute == 'US' or compute == 'USV':
        U[:,1:l] = U.T[1:l][idx].T
        V_t[1:l] = V_t[1:l][idx]
    else:
        V_t[1:l] = V_t[1:l][idx]

    return z, d, U, V_t, a

def compute_vec(d, sign_z, diff, sums, compute):
    
    n = np.size(d)
    z = np.ones(n)

    for i in range(0,n):
        z[i] = diff[n-1,i]*sums[n-1,i]
        for j in range(0,i):
            z[i] = z[i]*(diff[j,i]*sums[j,i] / (d[i] - d[j]) / (d[i] + d[j]))
        for j in range(i,n-1):
            z[i] = z[i]*(diff[j,i]*sums[j,i] / (d[i] - d[j+1]) / (d[i] + d[j+1]))
    z = np.sqrt(abs(z))*sign_z

    if compute == 'US' or compute == 'USV':
        U = np.zeros((n,n))
    else:
        U = None
    V = np.zeros((n,n))

    for i in range(0,n):
        V[0,i] = z[0] / diff[i,0] / sums[i,0]
        if compute == 'US' or compute == 'USV':
            U[0,i] = -1
        for j in range(1,n):
            V[j,i] = z[j] / diff[i,j] / sums[i,j]
            if compute == 'US' or compute == 'USV':
                U[j,i] = d[j]*V[j,i]
    for i in range(n):
        V[:,i] = V[:,i]/np.linalg.norm(V[:,i])
        if compute == 'US' or compute == 'USV':
            U[:,i] = U[:,i]/np.linalg.norm(U[:,i])

    V_t = V.T

    if compute == 'US' or compute == 'USV':
        if n == 1:
            if np.sign(U[0,0]) != np.sign(V[0,0]):
                U[0,0] = U[0,0]*(-1)

    return U, V_t
