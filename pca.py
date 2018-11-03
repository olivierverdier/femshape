import numpy as np
import matplotlib.pyplot as pl

import scipy.optimize as spo

class PCA:
	def __init__(self, pe, x, y):
		"""
		PCA plot.

		Parameters
		----------
		x, y : (C x P) arrays, where C is the number of curves, P is the number of points.
		pe: (C x F) array, where F is the number of features.
		"""
		self.pe = pe
		self.x = x
		self.y = y
		self.run()

	def run(self):
		# Centre the pe matrix
		pe = self.pe
		a = pe.mean(axis=0)
		a = (a*np.ones((1,np.shape(pe)[1]))).T
		a = np.tile(a,(1,np.shape(pe)[0]))
		pe = pe - a.T

		# Compute the PCA
		D,V = np.linalg.eig(np.cov(pe.T))
		order = np.argsort(D)
		V = np.real(V)

		self.pr = np.dot(pe,V[:,order[-2:]])

	def plot(self, scaling=20):
		# for i in range(ncurves):
		for i, (pr, x, y) in enumerate(zip(self.pr, self.x, self.y)):
			pl.plot(-pr[0]+ x/scaling,pr[1] + y/scaling)
			pl.plot(-pr[0],pr[1],'.')
			pl.text(-pr[0],pr[1],i)
		pl.xlabel('PC1')
		pl.ylabel('PC2')
		pl.axis('tight')
		#a = pl.gca()
		#a.axes.xaxis.set_ticklabels([])
		#a.axes.yaxis.set_ticklabels([])

def distance_matrix(pe):
	ncurves = len(pe)
	d = np.zeros((ncurves,ncurves))    
	for i in range(ncurves-1):
		for j in range(i,ncurves):
			d[i,j] = np.linalg.norm(pe[i,:] - pe[j,:])
	d = d + d.T
	return d

def pca_opt(pca, scaling=20):
	pr = pca.pr
	x = pca.x
	y = pca.y
	d = distance_matrix(pca.pe)
	ncurves = len(pr)
	# Optimisation
	pr2 = np.reshape(pr, [np.shape(pr)[0]*np.shape(pr)[1]])
	# N = #params in pr2, M = # datapoints
	pr3,dummy = spo.leastsq(objfun,pr2,args=d)
	o = objfun(pr3,d)
	o = np.reshape(o,[ncurves-1,ncurves])
	p = objfun(pr2,d)
	p = np.reshape(p,[ncurves-1,ncurves])
	print (np.sqrt(np.linalg.norm(p)**2/(ncurves*(ncurves-1)/2.)))
	print (np.sqrt(np.linalg.norm(o,'fro')**2/(ncurves*(ncurves-1)/2.)))
	pr3 = np.reshape(pr3, np.shape(pr))

	# Centre the pr3 matrix -- optimiser makes it wander
	a = pr3.mean(axis=0)
	a = (a*np.ones((1,np.shape(pr3)[1]))).T
	a = np.tile(a,(1,np.shape(pr3)[0]))
	pr3 = pr3 - a.T
	pr3 = 2*pr3
	
	pl.figure()
	for i in range(ncurves):
		pl.plot(-pr3[i,0]+x[i,:]/(scaling/2.5),pr3[i,1]+y[i,:]/(scaling/2.5))
		pl.plot(-pr3[i,0],pr3[i,1],'.')
		pl.text(-pr3[i,0],pr3[i,1],i)
	pl.axis('equal')
	pl.axis('tight')
	
	#a = pl.gca()
	#a.axes.xaxis.set_ticklabels([])
	#a.axes.yaxis.set_ticklabels([])
	
# Objective function for optimisation of the curve placement
def objfun(pl,d):
	# Compute how well the 2D positions match between pl and d
	pl = np.reshape(pl,[np.shape(pl)[0]//2,2])
	ncurves = np.shape(d)[0]
	e = np.zeros((ncurves-1,ncurves))
	for i in range(ncurves-1):
		for j in range(i+1,ncurves):
			e[i,j] = np.sqrt((pl[i,0]-pl[j,0])**2 + (pl[i,1]-pl[j,1])**2) - d[i,j]
	return np.reshape(e,[(ncurves-1)*ncurves])



from femshape import Current, Representer

def fem_current_pca(currents, x, y):
	"""
	Prepare PCA data using FEM current invariants.
	"""
	#x[0,:] = 0.5*np.cos(np.linspace(0,2*np.pi,npoints,endpoint=False))
	#y[0,:] = 0.5*np.sin(2*np.linspace(0,2*np.pi,npoints,endpoint=False))
	
	
	#U,V,H1,H2, G, invx, invy = rep_fem(x,y)
	#G = np.squeeze(G[0,:,:])
	rep = Representer(currents[0])
	G = rep.inertia.array()
	invs = np.array([current.invariants for current in currents])
	invx = invs[...,0]
	invy = invs[...,1]
	
	# For the H^1 metric use L = np.linalg.cholesky(G)
	L = np.linalg.cholesky(G)
	# For the H^2 metric use L = G
	#L=G
	
	Linv = np.linalg.inv(L)
	from scipy.linalg import block_diag
	Gihalf = block_diag(Linv,Linv)
	curr2 = np.hstack((invx,invy))

	# representer in Euclidean basis
	pe = np.zeros(np.shape(curr2))
	for i in range(len(x)):
		pe[i,:] = np.dot(Gihalf,curr2[i,:].T).T
	
	return pe
