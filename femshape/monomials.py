import numpy as np
import matplotlib.pyplot as pl



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


def monomial_current_pca(x, y, N=10):
	"""
	Prepare PCA data using monomial current invariants.
	"""
	ncurves, npoints = x.shape
	cl = (npoints-1)*np.ones((1,ncurves),dtype=int)
	#x,y = randshapes(ncurves,npoints)
	#x[0,:] = 0.5*np.cos(np.linspace(0,2*np.pi,npoints,endpoint=False))
	#y[0,:] = 0.5*np.sin(2*np.linspace(0,2*np.pi,npoints,endpoint=False))
	x,y,curr,curr2 = getc(x,y,ncurves,cl,N)

	[Gi,G] = dualnorm(N)
	L = np.linalg.cholesky(G)
	Linv = np.linalg.inv(L)
	from scipy.linalg import block_diag
	#Gihalf = block_diag(Linv,Linv)
	Gihalf = block_diag(Linv)
	
	#Ghalf = np.linalg.cholesky(G).T

	# representer in Euclidean basis
	#ue = np.dot(Ghalf,u1)
	#print (np.sqrt(np.dot(u1.T,np.dot(G,u1))),np.linalg.norm(ue))
	#Gihalf = np.linalg.cholesky(Gi).T

	pe = np.zeros(np.shape(curr2))
	for i in range(ncurves):
		pe[i,:] = np.dot(Gihalf,curr2[i,:].T).T

	return pe
	

	#curr2 = np.hstack((invx,invy))
