"""
Module for computing shape invariants of planar curves using FEniCS.

This module has been tested with FEniCS version 2018.01
"""

# the following import is is necessary for the 3D plot to work
from mpl_toolkits.mplot3d import Axes3D

import dolfin as fem
from dolfin import inner, dx, grad
from numpy import zeros, array, linspace, vstack, meshgrid, ascontiguousarray
from matplotlib import cm

class FEMShapeInvariant(object):
	"""
	Class for extracting shape invariants using FEM.
	"""

	def __init__(self, space=None, order=2, meshsize=64, L=1):
		"""
		Initialize.

		Parameters
		----------
		space : fenics.FunctionSpace or None
			FEniCS object of type `FunctionSpace` defining the FEM space.
			If equal to None, then a new `FunctionSpace` is created on the square domain [-1,1]x[-1,1].

		order : int
			The order of the FEM space. Only used if `space` is None.

		meshsize : int
			Size of the mesh underlying the FEM space. Only used if `space` is None.
		"""
		super(FEMShapeInvariant, self).__init__()

		self.order = order
		self.meshsize = meshsize
		# Initialize FEM space
		if space is not None:
			self.V = space
			self.mesh = self.V.mesh()
		else:
			self.mesh = fem.RectangleMesh(fem.Point(-L,-L), fem.Point(L,L), meshsize, meshsize, "left")
			#self.V = fem.FunctionSpace(self.mesh, "DG", order)
			self.V = fem.FunctionSpace(self.mesh, "CG", order)
		self.element = self.V.element() # Basic element type
		self.L = L
		# Build bounding box trees
		self.tree = fem.BoundingBoxTree()
		self.tree.build(self.mesh)

		# Create FEM functions for the invariants
		self.invariant_dx = fem.Function(self.V)
		self.invariant_dy = fem.Function(self.V)


	def compute_invariants(self, gamma, closed=True):
		"""
		Compute the FEM invariants associated with the curve `gamma`.

		Parameters
		----------
		gamma : ndarray, shape (n,2)
			Shape represented as `n` ordered points in the plane.

		closed : bool
			Specify if gamma is a closed curve.
		"""

		# Check shape of gamma
		if len(gamma.shape) is not 2:
			raise AttributeError("gamma has the wrong shape.")
		elif gamma.shape[1] is not 2 and gamma.shape[0] is not 2:
			raise AttributeError("gamma should be a sequence of planar points.")
		elif gamma.shape[1] is not 2:
			gamma = gamma.T

		self.gamma = gamma
		# Create output vectors (the invariants)
		invariants = zeros((self.V.dim(),2),dtype=float, order='F')

		# Extend with one point if gamma is closed
		if closed:
			gamma = vstack((gamma,gamma[0]))

		# Loop over points on the curve
		for (xk,xkp1,yk,ykp1) in zip(gamma[:-1,0],gamma[1:,0],gamma[:-1,1],gamma[1:,1]):

			xmid = (xk+xkp1)/2
			ymid = (yk+ykp1)/2
			midpoint = fem.Point(xmid, ymid)

			# Compute which cells in mesh collide with point
			collisions = self.tree.compute_entity_collisions(midpoint)

			# Skip if midpoint does not collide with any cell
			if len(collisions) == 0:
				# print "Skipping point, no collisions found: (%g, %g)" % (p.x(), p.y())
				continue

			# Pick first cell (may be several)
			cell_index = collisions[0]
			cell = fem.Cell(self.mesh, cell_index)

			# Evaluate basis functions associated with the selected cell
			values = zeros(self.element.space_dimension())
			vertex_coordinates = cell.get_vertex_coordinates()
			values = self.element.evaluate_basis_all(array([xmid,ymid]), vertex_coordinates, 0)

			# Find the global basis function indices associated with the selected cell
			global_dofs = self.V.dofmap().cell_dofs(cell_index)

			# Compute the invariant integrals
			invariants[global_dofs,0] += values*(xkp1-xk)
			invariants[global_dofs,1] += values*(ykp1-yk)

		# Store results in FEniCS functions
		self.invariant_dx.vector()[:] = invariants[:,0]
		self.invariant_dy.vector()[:] = invariants[:,1]

		return invariants

	def matrix_representation(self, size=256):
		(xx,yy,ux,uy) = self.mat_rep(self.invariant_dx, self.invariant_dy)
		return ux, uy

	def mat_rep(self, x, y, size=256):
		"""
		This function return a matrix representation of the
		dx and dy invariants thought of a FEniCS functions on
		the underlying FEM space.

		It is assumed that the mesh streches over [-1,1]x[-1,1].
		"""

		# Create matrix x and y coordinates
		[xx,yy] = meshgrid(linspace(-self.L,self.L,size), linspace(-self.L,self.L,size))
		coords = zeros((size**2,2), dtype=float)
		coords[:,0] = xx.reshape(size**2)
		coords[:,1] = yy.reshape(size**2)

		# Use this array to send into FEniCS.
		val = array([1.0],dtype=float)

		# Create the dx invariant matrix
		values = []
		for c in coords:
			# Evaluate the FEniCS function `invariant_dx` at the point `c`
			x.eval(val, ascontiguousarray(c))

			# Append the computed value in a vector
			values.append(val[0])

		# Reformat the vector of values
		values = array(values)
		ux = values.reshape((size,size))

		# Create the dy invariant matrix
		values = []
		for c in coords:
			# Evaluate the FEniCS function `invariant_dy` at the point `c`
			y.eval(val, ascontiguousarray(c))

			# Append the computed value in a vector
			values.append(val[0])

		# Reformat the vector of values
		values = array(values)
		uy = values.reshape((size,size))

		# Return the two matrices
		return (xx,yy,ux,uy)

	def calcM(self, show_plot=False, ret_inv=False, invariants=[], name=''):
		import matplotlib.pyplot as pl
		import numpy as np
		u = fem.TrialFunction(self.V)
		v = fem.TestFunction(self.V)

		# Choice of metric
		# H^1 metric with length scale c^2 = 1/10
		m = 1./10*inner(grad(u),grad(v))*dx() + u*v*dx()
		# L^2
		mL2 = u*v*dx

		M = fem.PETScMatrix()
		fem.assemble(m,tensor=M)
		#M = fem.assemble(m)
		ML2 = fem.assemble(mL2)

		x = fem.Function(self.V)
		y = fem.Function(self.V)
		x2 = fem.Function(self.V)
		y2 = fem.Function(self.V)

		if invariants != []:
			self.invariant_dx.vector()[:] = invariants[:,0]
			self.invariant_dy.vector()[:] = invariants[:,1]

		fem.solve(M,x2.vector(),self.invariant_dx.vector())
		fem.solve(M,y2.vector(),self.invariant_dy.vector())

		# H^2 metric
		x3 = x2*v*dx()
		y3 = y2*v*dx()
		M3x = fem.assemble(x3)
		M3y = fem.assemble(y3)
		fem.solve(M,x.vector(),M3x)
		fem.solve(M,y.vector(),M3y)

		if show_plot:
			# Plot the representer
			(xrep,yrep,ux,uy) = self.mat_rep(x,y,size=41)

			pl.figure()
			pl.quiver(xrep,yrep,ux,uy)
			pl.plot(self.gamma[:,0],self.gamma[:,1],linewidth=4)
			pl.axis('tight')
			pl.axis('equal')
			if name!='':
				name = name+str(self.order)+'_'+str(self.meshsize)+'rep.pdf'
				pl.savefig(name,dpi=600)

			fig = pl.figure()
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(xrep,yrep,np.sqrt(ux**2+uy**2), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
			pl.title('Order '+str(self.order)+' Meshsize '+str(self.meshsize))
			if name!='':
				name = name+str(self.order)+'_'+str(self.meshsize)+'rep3.pdf'
				pl.savefig(name,dpi=600)

			#pl.figure()
			#plot(self.mesh)

		# Print the norm
		H1 = x2.vector().inner(self.invariant_dx.vector())
		H1 += y2.vector().inner(self.invariant_dy.vector())
		H2 = x.vector().inner(self.invariant_dx.vector())
		H2 += y.vector().inner(self.invariant_dy.vector())

		# H1 = np.inner(x2.vector().array(),self.invariant_dx.vector().array()) + np.inner(y2.vector().array(),self.invariant_dy.vector().array())
		# H2 = np.inner(x.vector().array(),self.invariant_dx.vector().array()) + np.inner(y.vector().array(),self.invariant_dy.vector().array())

		if ret_inv:
			return x2.vector()[:], y2.vector()[:], H1, H2, M.array(), self.invariant_dx.vector()[:], self.invariant_dy.vector()[:]
		else:
			return x.vector()[:], y.vector()[:], H1, H2
