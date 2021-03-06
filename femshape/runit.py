
import femshape2 as fem
reload(fem)
import pylab as pl
import numpy as np

def run():

	pl.ion()
	n = 5000

	# Circle
	#t=np.linspace(0,2*np.pi,n,endpoint=False)
	#gamma1 = np.vstack((np.cos(t),np.sin(t))).T

	# Ellipse
	#t=np.linspace(0,2*np.pi,n,endpoint=False)
	#gamma1 = np.vstack((0.7*np.cos(t),0.9*np.sin(t))).T
	
	# Vertical line
	t=np.linspace(-1,1,n,endpoint=True)
	gamma1 = np.vstack((0.0*t-0.5,t)).T
	
	# Circle only in the lower (left-hand) triangle to compare with the matlab code
	#t=np.linspace(0,2*np.pi,n,endpoint=False)
	#gamma1 = np.vstack((-0.5+0.25*np.cos(t),-0.5+0.25*np.sin(t))).T

	#pl.figure()
	#pl.plot(gamma1[:,0],gamma1[:,1])
	#pl.axis('equal')
	
	#for l in range(5,22,5):
	for o in [1]: #range(1,8):
		#shapecalc = fem.FEMShapeInvariant(order=1, meshsize=4,L=3)
		shapecalc = fem.FEMShapeInvariant(order=o, meshsize=8,L=2)
		#inv1 = shapecalc.compute_invariants(gamma1,closed=True)
		inv1 = shapecalc.compute_invariants(gamma1,closed=False)
		shapecalc.calcM()
	


