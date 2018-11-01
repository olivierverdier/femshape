"""
Module for computing shape invariants of planar curves using FEniCS.

This module has been tested with FEniCS version 2018.01
"""

import dolfin as fem
from dolfin import inner, dx, grad
import numpy as np
from numpy import zeros, array, linspace, vstack, meshgrid, ascontiguousarray

import matplotlib.pyplot as pl

class Space:
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

	def grid_evaluation(self, x, y, size=256):
		"""
		Evaluates the FEniCS functions x and y on a grid of given size.

		Return
		------
		The grid points, and the corresponding function evaluations.
		"""

		# Create matrix x and y coordinates
		L = self.L
		[xx,yy] = meshgrid(linspace(-L,L,size), linspace(-L,L,size))
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


def compute_invariants(space, gamma):
	"""
	Compute the FEM invariants associated with the curve `gamma`.

	Parameters
	----------
	space : FEM space
	gamma : ndarray, shape (n,2)
		Shape represented as `n` ordered points in the plane.

	closed : bool
		Specify if gamma is a closed curve.

	Return
	------
	Two functions defined on the given space.
	"""

	# Check shape of gamma
	if len(gamma.shape) is not 2:
		raise AttributeError("gamma has the wrong shape.")
	elif gamma.shape[1] is not 2 and gamma.shape[0] is not 2:
		raise AttributeError("gamma should be a sequence of planar points.")
	elif gamma.shape[1] is not 2:
		gamma = gamma.T

	space.gamma = gamma
	# Create output vectors (the invariants)
	invariants = zeros((space.V.dim(),2),dtype=float, order='F')

	# Loop over points on the curve
	for (xk,xkp1,yk,ykp1) in zip(gamma[:-1,0],gamma[1:,0],gamma[:-1,1],gamma[1:,1]):

		xmid = (xk+xkp1)/2
		ymid = (yk+ykp1)/2
		midpoint = fem.Point(xmid, ymid)

		# Compute which cells in mesh collide with point
		collisions = space.tree.compute_entity_collisions(midpoint)

		# Skip if midpoint does not collide with any cell
		if len(collisions) == 0:
			# print "Skipping point, no collisions found: (%g, %g)" % (p.x(), p.y())
			continue

		# Pick first cell (may be several)
		cell_index = collisions[0]
		cell = fem.Cell(space.mesh, cell_index)

		# Evaluate basis functions associated with the selected cell
		values = zeros(space.element.space_dimension())
		vertex_coordinates = cell.get_vertex_coordinates()
		values = space.element.evaluate_basis_all(array([xmid,ymid]), vertex_coordinates, 0)

		# Find the global basis function indices associated with the selected cell
		global_dofs = space.V.dofmap().cell_dofs(cell_index)

		# Compute the invariant integrals
		invariants[global_dofs,0] += values*(xkp1-xk)
		invariants[global_dofs,1] += values*(ykp1-yk)

	return invariants


class Current:
	"""
	Compute the discrete invariants associated to a curve, regarded as a current.
	"""
	def __init__(self, space, curve, closed=True):
		self.space = space
		# Extend with one point if gamma is closed
		if closed:
			curve = vstack((curve, curve[0]))
		self.curve = curve
		invariants = compute_invariants(space, curve)
		self.invariants = invariants
		# Create FEM functions for the invariants
		self.invariant_dx = fem.Function(space.V)
		self.invariant_dy = fem.Function(space.V)
		# Store results in FEniCS functions
		self.invariant_dx.vector()[:] = invariants[:,0]
		self.invariant_dy.vector()[:] = invariants[:,1]

class Representer:
	def __init__(self, current, scale=1/np.sqrt(10)):
		self.current = current
		self.scale = scale
		self.compute_representers()

	def compute_representers(self):
		V = self.current.space.V
		u = fem.TrialFunction(V)
		v = fem.TestFunction(V)

		# Choice of metric
		# H^1 metric with length scale c^2 = 1/10
		m = self.scale**2*inner(grad(u),grad(v))*dx() + u*v*dx()
		# L^2
		# mL2 = u*v*dx

		M = fem.PETScMatrix()
		fem.assemble(m,tensor=M)
		#M = fem.assemble(m)
		# ML2 = fem.assemble(mL2)

		x = fem.Function(V)
		y = fem.Function(V)
		x2 = fem.Function(V)
		y2 = fem.Function(V)

		invariant_dx, invariant_dy = self.current.invariant_dx, self.current.invariant_dy

		fem.solve(M,x2.vector(), invariant_dx.vector())
		fem.solve(M,y2.vector(), invariant_dy.vector())

		# H^2 metric
		x3 = x2*v*dx()
		y3 = y2*v*dx()
		M3x = fem.assemble(x3)
		M3y = fem.assemble(y3)
		fem.solve(M,x.vector(),M3x)
		fem.solve(M,y.vector(),M3y)


		# Compute the norm
		H1 = x2.vector().inner(invariant_dx.vector())
		H1 += y2.vector().inner(invariant_dy.vector())
		H2 = x.vector().inner(invariant_dx.vector())
		H2 += y.vector().inner(invariant_dy.vector())

		# H1 = np.inner(x2.vector().array(),self.invariant_dx.vector().array()) + np.inner(y2.vector().array(),self.invariant_dy.vector().array())
		# H2 = np.inner(x.vector().array(),self.invariant_dx.vector().array()) + np.inner(y.vector().array(),self.invariant_dy.vector().array())

		self.H1 = x2, y2
		self.H1_sq_norm = H1
		self.H2 = x, y
		self.H2_sq_norm = H2
		self.M = M

	def plot(self, order=2, size=64, name=None):
		"""
		Plot the shape and its representer.
		"""
		if order == 1:
			x, y = self.H1
		elif order ==2:
			x, y = self.H2
		else:
			raise ValueError()
		xrep, yrep, ux, uy = self.current.space.grid_evaluation(x, y, size=size)
		lengths = np.sqrt(np.square(ux) + np.square(uy))
		pl.quiver(xrep,yrep,ux,uy, lengths)
		pl.plot(self.current.curve[:,0],self.current.curve[:,1],linewidth=4, alpha=.5)
		pl.axis('tight')
		pl.axis('equal')
		pl.colorbar()
		if name is not None:
			name = name+str(self.current.space.order)+'_'+str(self.current.space.meshsize)+'rep.pdf'
			pl.savefig(name,dpi=600)


		#pl.figure()
		#plot(self.mesh)

