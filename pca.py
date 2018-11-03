import numpy as np
import matplotlib.pyplot as pl

import scipy.optimize as spo

def compute_pca(pe, x, y):
	"""
	Take a representer in either form and compute PCA.
	"""
	#TODO Need to fix the plotting a little
	D,Evec = np.linalg.eig(np.cov(pe.T))
	order = np.argsort(D)
	Evec = np.real(Evec)

	pr = np.dot(pe,Evec[:,order[-2:]])
	#pl.plot(pr[:,0],pr[:,1],'.')
	pl.figure()
	for i in range(np.shape(pr)[0]):
		pl.plot(pr[i,1],pr[i,0],'.')
		pl.text(pr[i,1],pr[i,0],i)
	pl.axis('equal')

	pl.figure()
	for i in range(np.shape(pr)[0]):
		pl.plot(pr[i,1] + 0.5*x[i,:],pr[i,0] + 0.5*y[i,:]/5,'.')
		pl.text(pr[i,1] + x[i,0],pr[i,0] + y[i,0],i)
	pl.axis('equal')


# PCA
def pca_opt(pe,x,y,name='',scaling=20):
	ncurves = np.shape(pe)[0]
	# Centre the pe matrix
	a = pe.mean(axis=0)
	a = (a*np.ones((1,np.shape(pe)[1]))).T
	a = np.tile(a,(1,np.shape(pe)[0]))
	pe = pe - a.T

	# Compute the PCA
	D,V = np.linalg.eig(np.cov(pe.T))
	order = np.argsort(D)
	V = np.real(V)

	pr = np.dot(pe,V[:,order[-2:]])
	
	for i in range(ncurves):
		pl.plot(-pr[i,0]+ x[i,:]/scaling,pr[i,1] + y[i,:]/scaling)
		pl.plot(-pr[i,0],pr[i,1],'.')
		pl.text(-pr[i,0],pr[i,1],i)
	pl.xlabel('PC1')
	pl.ylabel('PC2')
	pl.axis('tight')
	if name!='':
		pl.savefig(name+'.pdf',dpi=600)
	#a = pl.gca()
	#a.axes.xaxis.set_ticklabels([])
	#a.axes.yaxis.set_ticklabels([])
		
	d = np.zeros((ncurves,ncurves))    
	for i in range(ncurves-1):
		for j in range(i,ncurves):
			d[i,j] = np.linalg.norm(pe[i,:] - pe[j,:])
	d = d + d.T

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
	pl.xlabel('PC1')
	pl.ylabel('PC2')
	pl.axis('tight')
	if name!='':
		pl.savefig(name+'_opt.pdf',dpi=600)
	
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
