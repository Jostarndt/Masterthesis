import scipy
from scipy import linalg
import numpy as np


print('start')


A = np.array([[-1, 1],[-0.5,-0.5]])
B = np.array([[0, 0],[0,0]])
Q = np.array([[1, 0],[0,1]])
R = np.array([[1, 0],[0,1]])


P = linalg.solve_continuous_are(A,B,Q,R)

print(P)

print('Testing:')


test = np.matmul(A.transpose(), P)+ np.matmul(P, A) + Q


print(test)


print('done')




