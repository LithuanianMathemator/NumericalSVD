import numpy as np

def GolubKahan_SVD(alpha, beta, U=None, V_t=None, eps=1.0e-16, compute='USV'):
    '''
    algorithm for the computation of the SVD of a bidiagonal matrix using implicit QR
    input: alpha = diagonal, beta = superdiagonal of bidiagonal matrix, U and V_t 
    bidiagonalize initial matrix
    output: array of descending singular values, U, V_t matrices of left and right
    singular vectors, respectively
    '''
    alpha = np.copy(alpha)
    beta = np.copy(beta)
    n = np.size(alpha)
    q = 0
    if n == 1:
        if compute == 'S':
            return alpha
        if compute == 'US':
            return U, alpha
        if compute == 'SV':
            return alpha, V_t
        if compute == 'USV':
            return U, alpha, V_t
        
    it = 0
    maxiter = 101*n

    while q < n:

        it += 1
        
        # set upper diagonal elements to zero
        for j in range(n-q-1):
            if abs(beta[j]) <= eps*(abs(alpha[j]) + abs(alpha[j+1])):
                beta[j] = 0

        # determine p and q
        p, q_d = count_blocks(alpha[:n-q], beta[:n-q-1])
        q += q_d
        if alpha[n-q-1] == 0:
            if compute == 'S' or compute == 'US':
                zero_column(alpha[p:n-q], beta[p:n-q-1], None, compute)
            else:
                zero_column(alpha[p:n-q], beta[p:n-q-1], V_t[p:n-q,:], compute)
            q += 1

        if q < n and n-q-p > 1:
            if 0 in alpha[p:n-q]:
                j = np.where(alpha[p:n-q] == 0)[0][-1]
                if compute == 'S' or compute == 'SV':
                    zero_rotate(alpha[p+j:n-q], beta[p+j:n-q-1], None, compute)
                if compute == 'US' or compute == 'USV':
                    zero_rotate(alpha[p+j:n-q], beta[p+j:n-q-1], U[:,p+j:n-q], compute)
            else:
                if compute == 'S':
                    SVD_step(alpha[p:n-q], beta[p:n-q-1], None, None, compute)
                if compute == 'US':
                    SVD_step(alpha[p:n-q], beta[p:n-q-1], U[:,p:n-q], None, compute)
                if compute == 'SV':
                    SVD_step(alpha[p:n-q], beta[p:n-q-1], None, V_t[p:n-q,:], compute)
                if compute == 'USV':
                    SVD_step(alpha[p:n-q], beta[p:n-q-1], U[:,p:n-q], V_t[p:n-q,:], compute)

        if it > maxiter:
            break

    # make negative diagonal elements positive, adjust U
    neg_sg = np.where(alpha < 0)
    for j in neg_sg:
        alpha[j] = abs(alpha[j])
        if compute == 'USV' or compute == 'US':
            U[:,j] = -1*U[:,j]

    # sort diagonal
    idx = np.flip(np.argsort(abs(alpha)))
    alpha = alpha[idx]
    if compute == 'US' or compute == 'USV':
        U[:,:n] = (U.T[idx]).T
    if compute == 'SV' or compute == 'USV':
        V_t[:n] = V_t[idx]

    if compute == 'S':
        return alpha
    if compute == 'US':
        return U, alpha
    if compute == 'SV':
        return alpha, V_t
    if compute == 'USV':
        return U, alpha, V_t

def SVD_step(alpha, beta, U, V_t, compute='USV'):
    '''
    function to implicitly apply one step of the QR algorithm on B^TB for a bidiagonal
    matrix B
    input: alpha = diagonal of B, beta = superdiagonal of B, U and V_t unitary matrices
    used for bidiagonalization of initial matrix
    '''
    n = np.size(alpha)
    y, z = initialize_QR(alpha, beta)
    for k in range(n-1):
        
        c, s = determine_givens(y, z)
        a_k = c*alpha[k] - s*beta[k]
        a_l = c*alpha[k+1]
        b_k = s*alpha[k] + c*beta[k]
        if k > 0:
            b_j = c*beta[k-1] - s*z
            beta[k-1] = b_j
        beta[k] = b_k
        alpha[k] = a_k
        z = -s*alpha[k+1]
        alpha[k+1] = a_l
        y = alpha[k]

        if compute == 'SV' or compute == 'USV':
            temp_1 = np.copy(V_t[k,:])
            temp_2 = np.copy(V_t[k+1,:])
            V_t[k,:] = c*temp_1-s*temp_2
            V_t[k+1,:] = s*temp_1+c*temp_2
        
        c, s = determine_givens(y, z)
        a_k = c*alpha[k] - s*z
        a_l = s*beta[k] + c*alpha[k+1]
        b_k = c*beta[k] - s*alpha[k+1]
        if k < n-2:
            b_l = c*beta[k+1]
            z = -s*beta[k+1]
            beta[k+1] = b_l
        beta[k] = b_k
        alpha[k] = a_k
        alpha[k+1] = a_l
        y = beta[k]

        if compute == 'US' or compute == 'USV':
            temp_1 = np.copy(U[:,k])
            temp_2 = np.copy(U[:,k+1])
            U[:,k] = c*temp_1-s*temp_2
            U[:,k+1] = s*temp_1+c*temp_2

def count_blocks(alpha, beta):
    '''
    find partition of B with a maximal diagonal block and a block with no zero on the
    superdiagonal
    input: alpha = diagonal, beta = superdiagonal of B
    output: size p and q_0 such that q_0 is the size of the maximal diagonal block, p is
    the size of the block that is left
    '''
    if np.size(beta) == 0:
        return 0, 1
    n = np.size(alpha)
    q_0 = 0
    b = beta[n-2]
    while b == 0:
        q_0 += 1
        if q_0 == n-1:
            return 0, n
        else:
            b = beta[n-2-q_0]
    p = n-q_0-2
    if p == 0:
        return 0, q_0
    b = beta[n-3-q_0]
    while b != 0:
        p -= 1
        if p == 0:
            return 0, q_0
        else:
            b = beta[p-1]

    return p, q_0

def zero_column(alpha, beta, V_t, compute='USV'):
    '''
    function to zero the last element of beta if the last element of alpha is zero
    input: alpha = diagonal, beta = superdiagonal, V_t unitary matrix which bidiagonalizes
    initial matrix alongside U
    '''
    n = np.size(alpha)
    y = beta[n-2]
    for k in range(n-2, -1, -1):
        x = alpha[k]
        c, s = determine_givens(x, y)
        a_k = c*alpha[k] - s*y
        if k > 0:
            b_j = c*beta[k-1]
            y = s*beta[k-1]
            beta[k-1] = b_j
        alpha[k] = a_k
        if compute == 'SV' or compute == 'USV':
            temp_1 = np.copy(V_t[k,:])
            temp_2 = np.copy(V_t[n-1,:])
            V_t[k,:] = c*temp_1-s*temp_2
            V_t[n-1,:] = s*temp_1+c*temp_2
    beta[n-2] = 0

def zero_rotate(alpha, beta, U, compute='USV'):
    '''
    function to zero a row of the bidiagonal matrix if the diagonal element in that row
    is zero
    input: alpha = diagonal, beta = superdiagonal, U unitary matrix which bidiagonalizes
    initial matrix alongside V_t
    '''
    n = np.size(alpha)
    y = beta[0]
    for k in range (1, n):
        c, s = determine_givens(alpha[k], -y)
        a_k = s*y + c*alpha[k]
        if k < n-1:
            b_k = c*beta[k]
            y = -s*beta[k]
            beta[k] = b_k
        alpha[k] = a_k
        if compute == 'US' or compute == 'USV':
            temp_1 = np.copy(U[:,0])
            temp_2 = np.copy(U[:,k])
            U[:,0] = c*temp_1-s*temp_2
            U[:,k] = s*temp_1+c*temp_2
    beta[0] = 0

def initialize_QR(alpha, beta):
    '''
    function to compute the first values for QR in the SVD QR step by computing
    the eigenvalues of the trailing 2x2 submatrix of B^TB
    input: alpha = diagonal of B, beta = superdiagonal of B
    output: values to compute initial Givens rotation of QR with
    '''
    n = np.size(alpha)
    a = alpha[n-2]**2
    b = alpha[n-1]**2+beta[n-2]**2
    c = alpha[n-2]*beta[n-2]
    if n > 2:
        a += beta[n-3]**2
    mu_1 = ((a+b) + np.sqrt(4*c**2 + (a-b)**2))/2
    mu_2 = ((a+b) - np.sqrt(4*c**2 + (a-b)**2))/2
    mu = min([mu_1, mu_2], key=lambda x:abs(x-b))
    y = alpha[0]**2-mu
    z = alpha[0]*beta[0]

    return y, z

def determine_givens(y,z):
    '''
    function to determine Givens rotation eliminating z
    input: elements y and z of 2-dim vector, z to be eliminated
    output: c and s for Givens rotation [[c,s],[-s,c]]
    '''
    if z == 0:
        c = 1; s = 0
    else:
        if abs(z) > abs(y):
            t = -y/z; s = 1/np.sqrt(1+t**2); c = s*t
        else:
            t = -z/y; c = 1/np.sqrt(1+t**2); s = c*t

    return c, s
