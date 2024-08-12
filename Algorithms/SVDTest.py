import numpy as np
import scipy.linalg as sl
from scipy.stats import ortho_group
from scipy.stats import loguniform
import matplotlib.pyplot as plt
from JacobiSVD import JacobiSVD
from householder_bidiag import householder_bidiag
from GolubKahan_SVD import GolubKahan_SVD
from JacobiSVD_simple import jacobisvd_simple
from GuEisenstat_SVD import GuEisenstat_SVD
from tqdm import tqdm
from tqdm.contrib import itertools

eps_sys = np.finfo(float).eps

def FullSVDTest(test='rel_max_error', ref='GEJSV'):

    # arrays for errors
    qr_error_array = np.zeros(0)
    dac_error_array = np.zeros(0)
    jac_error_array = np.zeros(0)
    gesvd_error_array = np.zeros(0)
    gejsv_error_array = np.zeros(0)
    bound_array = np.zeros(0)
    sens_array = np.zeros(0)
    
    # matrix generation and testing 
    dims = [50]
    cond_s = [i for i in range(1,9)]
    cond_d = [i for i in range(1,9)]
    s_modes = [1,2,5]
    d_modes = [1,3,4]

    for n in dims:
        
        m = n + n//5

        for i, j, mode_s, mode_d in itertools.product(cond_s, cond_d, s_modes, d_modes):

            k_S = 10**(2*i)
            k_D = 10**(2*j)

            if test == 'reference':
                A, s = BendelMickey(k_S, k_D, mode_s, mode_d, m, n, test)
            else:
                A = BendelMickey(k_S, k_D, mode_s, mode_d, m, n, test)

            if test == 'rel_max_error' or test == 'full_sg_norm' or test == 'sensible':
                
                if ref == 'GESVD':
                    sg_values = sl.svd(A, compute_uv=False, lapack_driver='gesvd')
                    bound = k_S*eps_sys*1/10
                elif ref == 'GEJSV':
                    sg_values = sl.lapack.dgejsv(A, joba=0, jobu=3, jobv=3)[0]
                    bound = k_S*eps_sys*n

                alpha, beta = householder_bidiag(A, U=None, V_t=None, compute='S')

                qr = GolubKahan_SVD(alpha, beta, U=None, V_t=None, eps=eps_sys, compute='S')

                dac = GuEisenstat_SVD(alpha, beta, compute='S')

                jac = JacobiSVD(A, compute='S', conditioning='LQ', eps=eps_sys, simple=False)

                if test == 'sensible':
                    eig = np.sort(np.sqrt(sl.eig(A.T@A, right=False)))[::-1]

            elif test == 'reference':

                gesvd_values = sl.svd(A, compute_uv=False, lapack_driver='gesvd')
                gejsv_values = sl.lapack.dgejsv(A, joba=0, jobu=3, jobv=3)[0]
            
            else:
                
                U_qr = np.eye(m)
                V_qr = np.eye(n)
                U_qr, alpha, beta, V_qr = householder_bidiag(A, U_qr, V_qr, compute='USV')
                Q_dac, W_dac = np.copy(U_qr), np.copy(V_qr)

                U_qr, qr, V_qr = GolubKahan_SVD(alpha, beta, U_qr, V_qr, eps=eps_sys, compute='USV')

                beta_2 = np.zeros(n)
                beta_2[:n-1] = beta
                U_dac_temp, dac, V_dac_temp = GuEisenstat_SVD(alpha, beta, compute='USV')
                U_dac = Q_dac
                U_dac[:,:n] = Q_dac[:,:n]@U_dac_temp
                V_dac = V_dac_temp[:n,:n]@W_dac

                U_jac, jac, V_jac = JacobiSVD(A, compute='USV', conditioning='ACC', eps=eps_sys, simple=False)

            if test == 'rel_max_error':
                qr_error = np.max(abs((qr - sg_values)/sg_values))
                dac_error = np.max(abs((dac - sg_values)/sg_values))
                jac_error = np.max(abs((jac - sg_values)/sg_values))
                qr_error_array = np.append(qr_error_array, qr_error)
                dac_error_array = np.append(dac_error_array, dac_error)
                jac_error_array = np.append(jac_error_array, jac_error)
                bound_array = np.append(bound_array, bound)
            elif test == 'max_error':
                qr_error = np.max(abs((qr - sg_values)/sg_values[0]))
                dac_error = np.max(abs((dac - sg_values)/sg_values[0]))
                jac_error = np.max(abs((jac - sg_values)/sg_values[0]))
                qr_error_array = np.append(qr_error_array, qr_error)
                dac_error_array = np.append(dac_error_array, dac_error)
                jac_error_array = np.append(jac_error_array, jac_error)
            elif test == 'full_sg_norm':
                qr_error = np.linalg.norm(qr - sg_values)/np.linalg.norm(A)
                dac_error = np.linalg.norm(dac - sg_values)/np.linalg.norm(A)
                jac_error = np.linalg.norm(jac - sg_values)/np.linalg.norm(A)
                qr_error_array = np.append(qr_error_array, qr_error)
                dac_error_array = np.append(dac_error_array, dac_error)
                jac_error_array = np.append(jac_error_array, jac_error)
            elif test == 'full_svd':
                qr_error = np.linalg.norm(A - U_qr[:,:n]@np.diag(qr)@V_qr)/np.linalg.norm(A)
                dac_error = np.linalg.norm(A - U_dac[:,:n]@np.diag(dac)@V_dac)/np.linalg.norm(A)
                jac_error = np.linalg.norm(A - U_jac[:,:n]@np.diag(jac)@V_jac)/np.linalg.norm(A)
                qr_error_array = np.append(qr_error_array, qr_error)
                dac_error_array = np.append(dac_error_array, dac_error)
                jac_error_array = np.append(jac_error_array, jac_error)
            elif test == 'u_vectors':
                qr_error = np.linalg.norm(U_qr.T@U_qr - np.eye(m))
                dac_error = np.linalg.norm(U_dac.T@U_dac - np.eye(m))
                jac_error = np.linalg.norm(U_jac.T@U_jac - np.eye(m))
                qr_error_array = np.append(qr_error_array, qr_error)
                dac_error_array = np.append(dac_error_array, dac_error)
                jac_error_array = np.append(jac_error_array, jac_error)
            elif test == 'v_vectors':
                qr_error = np.linalg.norm(V_qr.T@V_qr - np.eye(n))
                dac_error = np.linalg.norm(V_dac.T@V_dac - np.eye(n))
                jac_error = np.linalg.norm(V_jac.T@V_jac - np.eye(n))
                qr_error_array = np.append(qr_error_array, qr_error)
                dac_error_array = np.append(dac_error_array, dac_error)
                jac_error_array = np.append(jac_error_array, jac_error)
            elif test == 'reference':
                # gesvd_error = np.max(abs((gesvd_values - np.sort(s)[::-1])/np.sort(s)[::-1]))
                # gejsv_error = np.max(abs((gejsv_values - np.sort(s)[::-1])/np.sort(s)[::-1]))
                # gesvd_error_array = np.append(gesvd_error_array, gesvd_error)
                # gejsv_error_array = np.append(gejsv_error_array, gejsv_error)
                gesvd_error = np.max(np.linalg.norm(gesvd_values - np.sort(s)[::-1])/np.linalg.norm(A))
                gejsv_error = np.max(np.linalg.norm(gejsv_values - np.sort(s)[::-1])/np.linalg.norm(A))
                gesvd_error_array = np.append(gesvd_error_array, gesvd_error)
                gejsv_error_array = np.append(gejsv_error_array, gejsv_error)
            elif test == 'sensible':
                qr_error = np.max(abs((qr - sg_values)/sg_values))
                dac_error = np.max(abs((dac - sg_values)/sg_values))
                jac_error = np.max(abs((jac - sg_values)/sg_values))
                eig_error = np.max(abs((eig - sg_values)/sg_values))
                qr_error_array = np.append(qr_error_array, qr_error)
                dac_error_array = np.append(dac_error_array, dac_error)
                jac_error_array = np.append(jac_error_array, jac_error)
                bound_array = np.append(bound_array, bound)
                sens_array = np.append(sens_array, eig_error)

    if test == 'rel_max_error':
        x = np.arange(np.size(qr_error_array))
        plt.semilogy(x, qr_error_array, label='Golub-Reinsch', linewidth=0.5)
        plt.semilogy(x, dac_error_array, label='Divide-And-Conquer', linewidth=0.5)
        plt.semilogy(x, jac_error_array, label='One-sided Jacobi', linewidth=0.5)
        plt.semilogy(x, bound_array, label='Bound (4.1)', linewidth=0.5)
        plt.title(f'Error f for each algorithm, reference {ref}')
        plt.legend()
        plt.show()
    elif test == 'reference':
        x = np.arange(np.size(gesvd_error_array))
        plt.semilogy(x, gesvd_error_array, label='GESVD', linewidth=0.5)
        plt.semilogy(x, gejsv_error_array, label='GEJSV', linewidth=0.5)
        plt.title('Error n for each matrix')
        plt.xlabel('Test matrices G')
        plt.ylabel('Measured error for G')
        plt.legend()
        plt.show()
    elif test == 'full_svd':
        x = np.arange(np.size(qr_error_array))
        plt.semilogy(x, qr_error_array, label='Golub-Reinsch', linewidth=0.5)
        plt.semilogy(x, dac_error_array, label='Divide-And-Conquer', linewidth=0.5)
        plt.semilogy(x, jac_error_array, label='One-sided Jacobi', linewidth=0.5)
        plt.title(f'Error r for each algorithm')
        plt.legend()
        plt.show()
    elif test == 'u_vectors':
        x = np.arange(np.size(qr_error_array))
        plt.semilogy(x, qr_error_array, label='Golub-Reinsch', linewidth=0.5)
        plt.semilogy(x, dac_error_array, label='Divide-And-Conquer', linewidth=0.5)
        plt.semilogy(x, jac_error_array, label='One-sided Jacobi', linewidth=0.5)
        plt.title(f'Error o_U for each algorithm')
        plt.legend()
        plt.show()
    elif test == 'v_vectors':
        x = np.arange(np.size(qr_error_array))
        plt.semilogy(x, qr_error_array, label='Golub-Reinsch', linewidth=0.5)
        plt.semilogy(x, dac_error_array, label='Divide-And-Conquer', linewidth=0.5)
        plt.semilogy(x, jac_error_array, label='One-sided Jacobi', linewidth=0.5)
        plt.title(f'Error o_V for each algorithm')
        plt.legend()
        plt.show()
    elif test == 'sensible':
        x = np.arange(np.size(qr_error_array))
        plt.semilogy(x, qr_error_array, label='Golub-Reinsch', linewidth=0.5)
        plt.semilogy(x, dac_error_array, label='Divide-And-Conquer', linewidth=0.5)
        plt.semilogy(x, jac_error_array, label='One-sided Jacobi', linewidth=0.5)
        plt.semilogy(x, bound_array, label='Bound (4.1)', linewidth=0.5)
        plt.semilogy(x, sens_array, label='Eigenvalue solver', linewidth=0.5)
        plt.title(f'Error f for each algorithm, reference {ref}')
        plt.legend()
        plt.show()

def BendelMickey(k_S, k_D, mode_s, mode_d, m, n, test=None):
    
    s = np.zeros(n)

    if mode_s == 1:
        s[0] = 1
        s[1:] = np.ones(n-1)/k_S
    elif mode_s == 2:
        s[:n-1] = np.ones(n-1)
        s[n-1] = 1/k_S
    elif mode_s == 3:
        for i in range(n):
            s[i] = k_S**(-i/(n-1))
    elif mode_s == 4:
        for i in range(n):
            s[i] = 1 - i/(n-1)*(1-1/k_S)
    elif mode_s == 5:
        s = loguniform.rvs(1/k_S, 1, size=n)
    else:
        raise RuntimeError('No available mode for matrix generation selected.')
    
    eps_sys = np.finfo(float).eps
    tol = n*eps_sys
    
    W_1 = ortho_group.rvs(m)[:,:n]
    W_2 = ortho_group.rvs(n)

    if test == 'reference':
        return W_1@np.diag(s)@W_2, s

    a = np.sqrt(np.sum(s**2)/n)
    comp_array = np.ones(n)
    X = 1/a*W_1@np.diag(s)@W_2
    x_norms = np.zeros(n)
    for i in range(n):
        x_norms[i] = np.linalg.norm(X[:,i])
    close = np.isclose(x_norms, comp_array, tol, tol)
    while False in close:

        if np.size(np.where(x_norms < 1-tol)[0]) == 0\
              or np.size(np.where(x_norms > 1+tol)[0]) == 0:
            break

        # compute inner products where the columns are not equal to trace/n
        i_1 = np.where(x_norms < 1-tol)[0][0]
        i_2 = np.where(x_norms > 1+tol)[0][0]
        inner_1 = x_norms[i_1]**2
        inner_2 = x_norms[i_2]**2
        inner_mix = np.inner(X[:,i_1], X[:,i_2])

        # determine Givens rotations
        t = inner_mix + np.sign(inner_mix)*np.sqrt(inner_mix**2 - (inner_1-1)*(inner_2-1))
        t = t/(inner_2-1)
        cv = 1/np.sqrt(1+t**2)
        sv = cv*t

        # update X and x_norms
        temp = np.copy(X[:,i_1])
        X[:,i_1] = cv*temp - sv*X[:,i_2]
        X[:,i_2] = sv*temp + cv*X[:,i_2]
        x_norms[i_1] = 1
        x_norms[i_2] = np.linalg.norm(X[:,i_2])

        close = np.isclose(x_norms, comp_array, tol, tol)

    equil_X = X

    d = np.zeros(n)

    if mode_d == 1:
        d[0] = 1
        d[1:] = np.ones(n-1)/k_D
    elif mode_d == 2:
        d[:n-1] = np.ones(n-1)
        d[n-1] = 1/k_D
    elif mode_d == 3:
        for i in range(n):
            d[i] = k_D**(-i/(n-1))
    elif mode_d == 4:
        for i in range(n):
            d[i] = 1 - i/(n-1)*(1-1/k_D)
    elif mode_d == 5:
        d = loguniform.rvs(1, k_D, size=n)
    else:
        raise RuntimeError('No available mode for matrix generation selected.')
    
    A = equil_X@np.diag(d)

    return A

FullSVDTest()
