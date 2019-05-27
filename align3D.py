import sys
import numpy as np
import rmsd

def alin(A,B):
	A -= rmsd.centroid(A)
	B -= rmsd.centroid(B)
	#print (B)
	U = rmsd.kabsch(A, B)
	A = np.dot(A, U)

	return A, B, rmsd.rmsd(A,B)

if __name__=='__main__':
	b = np.array([[0,0,0], [1,1,1], [2,2,2]], dtype = 'float') # original molecule
	a = np.array([[0,0,0], [-1,-1,-1], [-2,-2,-2]], dtype = 'float') # fake molecule
	print (alin(a, b))





