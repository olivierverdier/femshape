"""
Module for computing shape invariants of planar curves using FEniCS.

This module requires FEniCS version 1.5.

Klas Modin, 2015-03-12
"""

import fenics as fem
from numpy import zeros, array, linspace, sin, cos, pi, vstack, hstack, meshgrid, ascontiguousarray

class FEMShapeInvariant(object):
	"""
	Class for extracting shape invariants using FEM.
	"""

	def __init__(self, space=None, order=2, meshsize=64):
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
		
		# Initialize FEM space
		if space is not None:
			self.V = space
			self.mesh = self.V.mesh()
		else:
			self.mesh = fem.RectangleMesh(-1, -1, 1, 1, meshsize, meshsize, "left")
			self.V = fem.FunctionSpace(self.mesh, "CG", order)
		self.element = self.V.element() # Basic element type

		# Build bounding box trees
		self.tree = fem.BoundingBoxTree()
		self.tree.build(self.mesh)

		# Create FEM functions for the invariants
		self.invariant_dx = fem.Function(self.V)
		self.invariant_dy = fem.Function(self.V)

	def compute_invariants(self, gamma, closed = True):
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
			self.element.evaluate_basis_all(values, array([xmid,ymid]), vertex_coordinates, 0)

			# Find the global basis function indices associated with the selected cell
			global_dofs = self.V.dofmap().cell_dofs(cell_index)

			# Compute the invariant integrals
			invariants[global_dofs,0] += values*(xkp1-xk)
			invariants[global_dofs,1] += values*(ykp1-yk)

		# Store results in FEniCS functions
		self.invariant_dx.vector()[:] = invariants[:,0]
		self.invariant_dy.vector()[:] = invariants[:,1]

		return invariants

	def matrix_representation(self,size=256):
		"""
		This function return a matrix representation of the 
		dx and dy invariants thought of a FEniCS functions on
		the underlying FEM space.

		It is assumed that the mesh streches over [-1,1]x[-1,1].
		"""

		# Create matrix x and y coordinates
		[xx,yy] = meshgrid(linspace(-1,1,size), linspace(-1,1,size))
		coords = zeros((size**2,2), dtype=float)
		coords[:,0] = xx.reshape(size**2)
		coords[:,1] = yy.reshape(size**2)

		# Use this array to send into FEniCS.
		val = array([1.0],dtype=float)

		# Create the dx invariant matrix
		values = []
		for c in coords:
			# Evaluate the FEniCS function `invariant_dx` at the point `c`
			self.invariant_dx.eval(val, ascontiguousarray(c))

			# Append the computed value in a vector
			values.append(val[0])

		# Reformat the vector of values
		values = array(values)
		xx[:,:] = values.reshape((size,size))

		# Create the dy invariant matrix
		values = []
		for c in coords:
			# Evaluate the FEniCS function `invariant_dy` at the point `c`
			self.invariant_dy.eval(val, ascontiguousarray(c))

			# Append the computed value in a vector
			values.append(val[0])

		# Reformat the vector of values
		values = array(values)
		yy[:,:] = values.reshape((size,size))

		# Return the two matrices
		return (xx,yy)





