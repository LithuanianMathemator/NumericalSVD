import numpy as np
import GolubKahan_SVD as gks

def GuEisenstat_SVD(alpha, beta, tol=1.0e-16, compute='USV'):
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

    if n < 25:

        U = np.eye(n, dtype=float)
        V_t = np.eye(n+1, dtype=float)
        temp_alpha = np.zeros(n+1, dtype=float)
        temp_alpha[:n] = alpha
        if beta[n-1] != 0:
            gks.zero_column(temp_alpha, beta, V_t)
        alpha = temp_alpha[:n]
        beta = beta[:n-1]
        W_t = np.zeros((n+1,n+1))
        W_t[n,:] = V_t[n,:]
        Q, sg_values, temp_W_t = gks.GolubKahan_SVD(alpha, beta, U, V_t[:n])
        W_t[:n,:] = temp_W_t

        return Q, sg_values, W_t
    
    k = n//2
    alpha_k = alpha[k-1]
    beta_k = beta[k-1]
    Q1, sg_values_1, W1_t = GuEisenstat_SVD(alpha[:k-1],beta[:k-1])
    Q2, sg_values_2, W2_t = GuEisenstat_SVD(alpha[k:],beta[k:])

    c, s = gks.determine_givens(alpha_k*W1_t[k-1,k-1], beta_k*W2_t[n-k,0])
    z = np.zeros(n)
    z[0] = c*alpha_k*W1_t[k-1,k-1]-s*beta_k*W2_t[n-k,0]
    z[1:k-1] = alpha_k*W1_t[:k-1,-1]
    z[k-1:] = beta_k*W2_t[:n-k,0]
    d = np.zeros(n)
    d[1:k-1] = sg_values_1
    d[k-1:] = sg_values_2

    U_m, sg_values, V_m_t = M_SVD(z, d, tol, compute)

    U = np.zeros((n,n))
    V_t = np.zeros((n+1,n+1))

    U[:k-1,:] = Q1@U_m[1:k-1,:]
    U[k-1,:] = U_m[0,:]
    U[k:,:] = Q2@U_m[k:]

    V_t[:n,:k] = V_m_t@(c*W1_t[k-1,:].reshape(1,k)) + V_m_t@W1_t[:k-1]
    V_t[:n,k:] = V_m_t@(s*W2_t[n-k,:].reshape(1,n-k+1)) + V_m_t@W2_t[:n-k]
    V_t[n,:k] = -s*W1_t[k-1,:]
    V_t[n,k:] = c*W2_t[n-k,:]

    return U, sg_values, V_t

def M_SVD(z, d, tol=1.0e-16, compute='USV'):
    '''
    function to compute the SVD of M where M has non-zero elements only in its first 
    row and diagonal
    input: z first row of M, d diagonal of M
    output: SVD of M
    '''
    n = np.size(d)
    G = np.diag(d)
    G[0,:] = z
    U = np.eye(n, dtype=float)
    V_t = np.eye(n, dtype=float)
    z, d, U, V_t, a = deflate_M(z, d, U, V_t, tol)
    sg_values_temp = solve_secular(z[:n-a], d[:n-a], tol)
    U_temp, V_temp = compute_vec(d[:n-a], sg_values_temp, z[:n-a])
    U[:,:n-a] = U[:,:n-a]@U_temp
    V_t[:n-a,:] = V_temp@V_t[:n-a,:]
    sg_values = np.zeros(n)
    sg_values[:n-a] = sg_values_temp
    sg_values[n-a:] = d[n-a:]
    # print(U@np.diag(sg_values)@V_t-G)
    idx = np.flip(np.argsort(abs(sg_values)))
    sg_values = sg_values[idx]
    if compute == 'US' or compute == 'USV':
        U = (U.T[idx]).T
    if compute == 'SV' or compute == 'USV':
        V_t = V_t[idx]

    return U, sg_values, V_t

def deflate_M(z, d, U, V_t, tol=1.0e-16):
    '''
    deflate the matrix M so d is ascending and no elements are in it more than once
    z is made large enough
    input: z first row of M, d diagonal of M, U and V_t are the identity
    output: updated z and d and U and V_t such that UM_newV_t is M, a such that the first
    n-a elements of d are ascending and unique
    '''
    a = 0
    n = np.size(d)
    idx = np.argsort(d[1:])

    d[1:] = d[1:][idx]
    z[1:] = z[1:][idx]
    U[:,1:] = U.T[1:][idx].T
    V_t[1:] = V_t[1:][idx]

    norm_M = np.sqrt(np.inner(d,d) + np.inner(z,z))
    tM = tol*norm_M

    if abs(z[0]) < tM:
        z[0] = tM

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
    U[:,1:] = U.T[1:][idx].T
    V_t[1:] = V_t[1:][idx]

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

    d[1:l] = d[1:l][idx]
    z[1:l] = z[1:l][idx]
    U[:,1:l] = U.T[1:l][idx].T
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
    U[:,1:l] = U.T[1:l][idx].T
    V_t[1:l] = V_t[1:l][idx]

    return z, d, U, V_t, a

def solve_secular(z, d, tol=1.0e-16):
    
    n = np.size(d)
    sg_values = np.zeros(n)
    
    for i in range(n):

        if i < n-1:
            mu = (d[i]-d[i+1])/2
            closest_d = d[i+1]
            psi, phi = evaluate_secular(z, d, mu, closest_d, i+1)

            if 1 + psi+ phi > 0:
                closest_d = d[i]
                a = 0
                b = (d[i+1]-d[i])/2
                case = 0

            else:
                closest_d = d[i+1]
                a = (d[i]-d[i+1])/2
                b = 0
                case = 1

        else:
            closest_d = d[i]
            a = 0
            b = np.linalg.norm(z)
            psi = 1
            phi = 1
            case = 0

        k = 0

        while abs(1+ psi + phi) > tol*n*(1 + abs(psi) + abs(phi)) and k < 10*n:
            
            mu = (a+b)/2
            psi, phi = evaluate_secular(z, d, mu, closest_d, i+case)
            if 1 + psi+ phi > 0:
                b = mu
            else:
                a = mu
            k += 1

        mu = (a+b)/2
        sg_values[i] = closest_d + mu
    
    return sg_values

def evaluate_secular(z, d, mu, closest_d, i):
    
    psi = 0
    phi = 0
    n = np.size(d)

    for j in range(i):
        delta_j = d[j]-closest_d
        psi += z[j]**2/((delta_j-mu)*(d[j]+d[i]+mu))

    for j in range(i,n):
        delta_j = d[j]-closest_d
        phi += z[j]**2/((delta_j-mu)*(d[j]+d[i]+mu))

    return psi, phi

def compute_vec(d, sg, sign_z):
    
    n = np.size(d)
    z = np.ones(n)

    for i in range(n):
        for j in range(n):
            if j == i:
                z[j] *= (sg[i]+d[j])*(sg[i]-d[j])
            else:
                z[j] *= (sg[i]+d[j])*(sg[i]-d[j])/((d[i]+d[j])*(d[i]-d[j]))
    z = np.sqrt(z)*np.sign(sign_z)

    U = np.zeros((n,n))
    V = np.zeros((n,n))

    for j in range(n):
        for i in range(n):
            if i == 0:
                U[i, j] = -1
            else:
                U[i, j] = d[i]*z[i]/((d[i]+sg[j])*(d[i]-sg[j])) 
            V[i, j] = z[i]/((d[i]+sg[j])*(d[i]-sg[j]))
        U[:,j] = U[:,j]/np.linalg.norm(U[:,j])
        V[:,j] = V[:,j]/np.linalg.norm(V[:,j])

    V_t = V.T

    return U, V_t
