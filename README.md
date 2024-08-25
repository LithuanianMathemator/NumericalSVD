# NumericalSVD

My bachelor's thesis is concerned with the numerical computation of the SVD. It goes over the theory of both the SVD and its numerical computation. Besides theoretical results, some applications also get shown. The most important part, however, are the algorithms and their testing.

This repository has all the implementations of the numerical algorithms that get discussed in the thesis, alongside the scripts used to generate the figures and examples in it.

The algorithms and tests can be found in the folder 'Algorithms'.

## Algorithms:
- Householder bidiagonalization: householder_bidiag.py
- Golub-Reinsch SVD: GolubKahan_SVD.py
- Divide-And-Conquer SVD: GuEisenstat_SVD.py
- One-sided Jacobi algorithm: JacobiSVD.py

## Tests:
All the tests are found in the file FullSVDTest.py. The two inputs for the function FullSVDTest() are test and reference.
- test = 'rel_max_error': error f from the thesis
- test = 'max_error': same as rel_max_error, but relative to the largest singular value
- test = 'full_sg_norm': error n from the thesis
- test = 'full_svd': error r from the thesis
- test = 'u_vectors': error o_U from the thesis
- test = 'v_vectors': error o_V from the thesis
- test = 'reference': test for the reference singular values
- test = 'sensible': same as 'rel_max_error with added test of eigenvalue algorithm
- test = 'time': runtime test
- test = 'vec_norm': vector accuracy test, error v in thesis
- reference = 'GESVD' for LAPACK driver DGESVD and 'GEJSV' for LAPACK driver DGEJSV

The examples and functions used to compute them are in the folder 'Examples'.

All the result graphs, even the ones that are not shown in the thesis, can be found in the folder 'ResultGraphs'.
