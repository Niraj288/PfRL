import sys
import numpy as np
from numpy import *

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
       #print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    #print t

    return R, t

def alin(a,b):
	# give coordinates in the form of numpy array of shape (n, 3)
	a = mat(a)
	b = mat(b)

	# R is rotation matrix
	R, t = rigid_transform_3D(a, b)

	T = a

	n = a.shape[0]

	# Rotate T to alighn mobile
	T = R*T.T + tile(t, (1, n))
	T = T.T 

	err = T - b

	err = multiply(err, err)
	err = sum(err)
	rmse = sqrt(err/n);

	return np.array(T), rmse

if __name__=='__main__':
	b = np.array([[0,0,0], [1,1,1], [2,2,2]]) # original molecule
	a = np.array([[0,0,0], [-1,-1,-1], [-2,-2,-2]]) # fake molecule
	#print alin(a, b)





