import scipy
from scipy import linalg
import numpy as np


print('start')


A = np.array([[-1, 1],[-0.5,-0.5]])
B = np.array([[3, 2],[1,0]])
Q = np.array([[1, 0],[0,1]])
R = np.array([[2, 0],[0,2]])


P = linalg.solve_continuous_are(A,B,Q,R)

print(P)

print('Testing:')


test = np.matmul(A.transpose(), P)+ np.matmul(P, A) + Q - np.matmul(P, np.matmul(B, np.matmul( np.linalg.inv(R), np.matmul(B.transpose(), P))))


print(test)


print('done')



#print('BR^ -1 B^t = ', np.matmul(B, np.matmul( np.linalg.inv(R), B.transpose())))
#print('PBR^ -1 B^t P = ', np.matmul(P, np.matmul(B, np.matmul( np.linalg.inv(R), np.matmul(B.transpose(), P)))))
print('optimal control: R^ -1 B^t P = ', np.matmul(np.linalg.inv(R), np.matmul(B.transpose(), P)))

