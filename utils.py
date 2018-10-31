import numpy as np

# Generate random Fourier shapes
def randshapes(ncurves, npoints, deviation=1):
	x = np.zeros((ncurves,npoints))
	y = np.zeros((ncurves,npoints))
	t = np.arange(0,2*np.pi,2*np.pi/npoints)
	t = t*np.ones((1,np.shape(t)[0]))
	nfourier = 6
	fr = np.concatenate((np.arange(nfourier+1),np.arange(-nfourier,0)),axis=0)
	fr = fr*np.ones((1,np.shape(fr)[0]))

	for shape in range(ncurves):
		a = (np.random.randn(2*nfourier+1) + 1j*np.random.randn(2*nfourier+1))*deviation
		a = a*np.ones((1,np.shape(a)[0]))
		a = a/(1+np.abs(fr)**3)

		a[0,0] = 0
		a[0,1] /= np.abs(a[0,1])
		a[0,-1] = 0;

		z = np.dot(a,np.exp(1j*np.tile(fr.T,(1,npoints))*np.tile(t,(2*nfourier+1,1))))
		x[shape,:] = 0.5*np.real(z)
		y[shape,:] = 0.5*np.imag(z)
	return np.array([x, y])

def figure_of_eight(npoints):
	x = 0.5*np.cos(np.linspace(0,2*np.pi,npoints,endpoint=False))
	y = 0.5*np.sin(2*np.linspace(0,2*np.pi,npoints,endpoint=False))
	return np.array([x,y])
