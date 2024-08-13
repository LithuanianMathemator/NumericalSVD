import numpy as np
import scipy.linalg as sl

d = np.array([0, 200.0, 300.0, 400.0, 500.0])
z = np.array([0.0000002, 0.0000001, 0.0000004, 0.0000005, 0.0000006])

diffs = np.zeros(5)
sums = np.zeros(5)
sg_values = np.zeros(5)

# DLASD4 outputs: 1) array of differences, 2) singular value, 3) array of sums
# 4) info about convergence

for i in range(5):

    diff_array, sg_values[i], sum_array, info = sl.lapack.dlasd4(i, d, z)
    diffs[i] = diff_array[i]
    sums[i] = sum_array[i]

    if info != 0:
        raise RuntimeError('The root finder did not converge.')
    
print(f'The computed singular values are: {sg_values}.')
print(f'The computed differences are: {diffs}.')
print(f'The computed sums are: {sums}.')
