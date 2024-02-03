from math import sqrt
import numpy as np

def euclidean(array_x, array_y):
	n = array_x.shape[0]
	ret = 0.
	for i in range(n):
		ret += (array_x[i]-array_y[i])**2
	return sqrt(ret)


def hausdorff(XA, XB):
	XA = np.array(XA)
	XB = np.array(XB)
	
	nA = XA.shape[0]
	nB = XB.shape[0]
	cmax = 0.
	for i in range(nA):
		cmin = np.inf
		for j in range(nB):
			d = euclidean(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and np.inf>cmin:
			cmax = cmin
	for j in range(nB):
		cmin = np.inf
		for i in range(nA):
			d = euclidean(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and np.inf>cmin:
			cmax = cmin
	return cmax



if __name__ == "__main__":
    # C = np.array([[1, 1], [2, 1]])
    # Q = np.array([[1, 1], [2, 1], [2, 1]])
    C = [[1, 1], [2, 1]]
    Q = [[1, 1], [2, 1], [2, 1]]
    dis = hausdorff(C, Q)
    print(dis)