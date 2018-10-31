import numpy as np
import matplotlib.pyplot as pl

import scipy.optimize as spo


def getc(x,y,ncurves,cl,N):
	curr = np.zeros((ncurves,N,N,2))
	# Make the curves periodic
	x = np.concatenate((x,(x[:,0]*np.ones((1,np.shape(x)[0]))).T),axis=1)
	y = np.concatenate((y,(y[:,0]*np.ones((1,np.shape(y)[0]))).T),axis=1)
	for i in range(ncurves):
		for m in range(N):
			for n in range(N):
				curr[i,m,n,:] = current(x[i,:],y[i,:],m,n)

	curr2 = np.reshape(curr,[ncurves,N*N*2],order='F').copy()
	return x,y,curr,curr2

def current(x,y,m,n):
	npoints = np.shape(x)[0]-1
	w = ((x[:npoints]+x[1:])/2.0)**m * ((y[:npoints]+y[1:])/2.0)**n
	c = np.array([np.sum(w*np.diff(x)),np.sum(w*np.diff(y))])
	index = np.where(np.abs(c)<1e-14)
	if np.shape(index)[1]>0:
		c[index] = 0
	return c

def dualnorm(N):
	G = np.zeros((N,N,N,N))
	for m1 in range(N):
		for m2 in range(N):
			for n1 in range(N):
				for n2 in range(N):
					G[m1,m2,n1,n2] = 10.0*(1.0/(m1+n1+1.0) - (-1.0)**(m1+n1+1.0)/(m1+n1+1.0)) * (1.0/(m2+n2+1.0) - (-1.0)**(m2+n2+1.0)/(m2+n2+1.0))
					if (m1*n1 > 0):
						G[m1,m2,n1,n2] = G[m1,m2,n1,n2] + (1.0+(-1.0)**(m1+n1)) * (1.0+(-1.0)**(m2+n2)) * m1*n1/(m1+n1-1.0)/(m2+n2+1.0)
					if (m2*n2 > 0):
						G[m1,m2,n1,n2] = G[m1,m2,n1,n2] + (1.0+(-1.0)**(m1+n1)) * (1.0+(-1.0)**(m2+n2)) * m2*n2/(m1+n1+1.0)/(m2+n2-1.0)

	G = np.reshape(G,(N*N,N*N),order='F').copy()
	# next 4 commented out?
	Gi = np.linalg.inv(G)
	from scipy.linalg import block_diag
	Gi = block_diag(Gi,Gi)
	G = block_diag(G,G)
	return [Gi,G]

def representer(x,y,ncurves,cl,npoints,N):
	"""
	Representer for the monomiials.
	"""
	[x,y,curr,curr2] = getc(x,y,ncurves,cl,N)
	[Gi,G] = dualnorm(N)
	
	u = np.zeros((ncurves,N,N,2))
	for i in range(ncurves):
		u1 = np.linalg.solve(G,curr2[i,:].T)
		u1 = np.reshape(u1,[N,N,2],order='F').copy()
		u[i,:,:,:] = u1
		
	#meshing = 2/0.05
	#Fix size of U if wish to use!
	#X,Y = np.meshgrid(np.linspace(-1,1,meshing),np.linspace(-1,1,meshing))
	#U = np.zeros((ncurves,np.shape(X)))
	#V = np.zeros((ncurves,np.shape(Y)))
	#print np.shape(U)

	#for i in range(ncurves):
	#    for m in range(N-1):
	#        for n in range(N-1):
	#            U[i] += X**m*Y**n*u[i,m+1,n+1,0]
	#            V[i] += X**m*Y**n*u[i,m+1,n+1,1]

	#print X, Y, U, V
	#print (np.max(np.sqrt(U**2+V**2), np.sqrt(np.dot(u1.T,np.dot(G,u1)))))

	return u
	#pl.figure()
	#pl.plot(x,y,'r')
	#pl.quiver(X,Y,U.astype("float64"),V.astype("float64"),'b')
	#pl.axis('equal')

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


