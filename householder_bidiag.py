import numpy as np

def householder_bidiag(A, U, V_t, mode='real', compute='USV'):
    '''
    function to determine a bidiagonal decomposition of A using
    Householder transformations
    input: matrix A to be bidiagonalized, U and V_t such that UAV_t is
    some initial matrix, usually U and V_t will be the identity
    output: alpha = main diagonal, beta = superdiagonal, U and V_t are updated
    '''
    B = np.copy(A)
    m_B, n_B = np.shape(B)
    m_U, n_U = np.shape(U)
    m_V, n_V = np.shape(V_t)

    for j in range(min(m_B, n_B)):
        
        v, b = determine_householder(B[j:,j],mode)
        B[j:,j:] = B[j:,j:] - \
            b*v.reshape(m_B-j,1)@(np.conjugate(v).T@B[j:,j:]).reshape(1,n_B-j)
        if compute == 'US' or compute == 'USV':
            U[:,j:] = U[:,j:] - \
                b*(U[:,j:]@v).reshape(m_U,1)@np.conjugate(v).reshape(1,m_B-j)
        
        if mode == 'complex':
            
            theta = np.angle(B[j,j])
            B[j,j:] = np.exp(-1j*theta)*B[j,j:]
            if compute == 'US' or compute == 'USV':
                U[:,j] = np.exp(1j*theta)*U[:,j]
        
        if j <= n_B-2 or n_B > m_B:

            v, b = determine_householder(np.conjugate(B[j, j+1:]),mode)
            B[j:,j+1:] = B[j:,j+1:] - \
                b*(B[j:,j+1:]@v).reshape(m_B-j,1)@np.conjugate(v).reshape(1,n_B-j-1)
            if compute == 'UV' or compute == 'USV':
                V_t[j+1:,:] = V_t[j+1:,:] - \
                    b*v.reshape(n_B-j-1,1)@(np.conjugate(v).T@V_t[j+1:,:]).reshape(1,n_V)
            
            if mode == 'complex':
                
                theta = np.angle(B[j,j+1])
                B[j:,j+1] = np.exp(-1j*theta)*B[j:,j+1]
                if compute == 'UV' or compute == 'USV':
                    V_t[j+1,:] = np.exp(1j*theta)*V_t[j+1,:]

    return abs(np.diag(B)), abs(np.diag(B,1))

def determine_householder(x, mode='real'):
    '''
    function to determine the householder vector onto which P reflects,
    P = I - beta*v*v_t unitary
    input: vector x
    output: v and beta such that P reflects onto scalar multiple of e_1
    '''

    if mode == 'real':

        m = np.size(x)
        sigma = np.inner(x[1:m], x[1:m])
        v = np.copy(x)
        v[0] = 1

        if sigma == 0 and x[0] >= 0:
            beta = 0
        elif sigma == 0 and x[0] < 0:
            beta = 2
        else:
            mu = np.sqrt(x[0]**2 + sigma)
            if x[0] <= 0:
                v[0] = x[0] - mu
            else:
                v[0] = -sigma/(x[0] + mu)
            beta = 2*v[0]**2/(sigma + v[0]**2)
            v = v/v[0]

        return v, beta

    if mode == 'complex':

        m = np.size(x)
        sigma = np.inner(np.conjugate(x[1:m]), x[1:m])
        v = np.copy(x) + 0j
        v[0] = 1

        r, theta = abs(x[0]), np.angle(x[0])

        if sigma == 0:
            return v, 0
        else:
            mu = np.sqrt(r**2+sigma)
            a = -sigma/(r + mu)
            v[0] = a*np.exp(1j*theta)
            beta = 2*a**2/(sigma+a**2)
            v = v/v[0]
            
            return v, beta
