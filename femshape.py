"""
Module for computing shape invariants of planar curves using FEniCS.

This module has been tested with FEniCS version 2018.01
"""

import dolfin as fem
from dolfin import inner, dx, grad
import numpy as np
from numpy import zeros, array, linspace, vstack, meshgrid, ascontiguousarray

def batch_eval(f, pts):
	"""
	Evaluate a FEniCS function on points.
	The points should be an array of shape (2,N,M)
	"""

	# Use this array to send into FEniCS.
	out = np.zeros(1)

	def gen():
		for pt in pts.reshape(2, -1).T:
			f.eval(out, pt)
			yield out[0]

	values = list(gen())
	avalues = np.array(values).reshape(pts.shape[1:])
	return avalues


class Space:
	"""
	Finite element discretisation.
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
		[xx, yy] = meshgrid(linspace(-L, L, size),  linspace(-L, L, size))
		pts = np.array([xx, yy])
		ux = batch_eval(x, pts)
		uy = batch_eval(y, pts)
		return xx, yy, ux, uy

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

def compute_representers(V, inertia, rhs):
	"""
	Auxilliary function to solve the equations
	M u = f
	and
	(M w, v) = (u, v) forall v
	where f is the right hand side (rhs), and M is the inertia.
	One also computes (u, f) and (w, v)
	"""
	M = inertia

	x = fem.Function(V)
	x2 = fem.Function(V)

	fem.solve(M, x2.vector(), rhs.vector())

	# H^2 metric
	v = fem.TestFunction(V)
	x3 = x2*v*dx()
	M3x = fem.assemble(x3)
	fem.solve(M,x.vector(),M3x)


	# Compute the norm
	H1 = x2.vector().inner(rhs.vector())
	H2 = x.vector().inner(rhs.vector())

	return x2, x, H1, H2


class Representer:
	def __init__(self, current, scale=1/np.sqrt(10)):
		self.current = current
		self.scale = scale

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
		self.inertia = M
		#M = fem.assemble(m)
		# ML2 = fem.assemble(mL2)

		x1, x2, H1x, H2x = compute_representers(V, self.inertia, self.current.invariant_dx)
		y1, y2, H1y, H2y = compute_representers(V, self.inertia, self.current.invariant_dy)
		self.H1 = x1, y1
		self.H2 = x2, y2
		self.H1_sq_norm = H1x + H1y
		self.H2_sq_norm = H2x + H2y
		# self.compute_representers(V, )

	def square_distance(self, rep):
		rdiffx = rep.H2x.vector() - self.H2x.vector()
		rdiffy = rep.H2y.vector() - self.H2y.vector()
		diffx = rep.current.invariant_dx.vector() - self.current.invariant_dx.vector()
		diffy = rep.current.invariant_dy.vector() - self.current.invariant_dy.vector()
		return rdiffx.inner(diffx) + rdiffy.inner(diffy)


import matplotlib.pyplot as plt

def plot_representer(representer, order=2, size=32):
	"""
	Plot the H1 or H2 representer of a given current.
	"""
	if order == 2:
		x,y = representer.H2
	else:
		x,y = representer.H1
	xx, yy, ux, uy = representer.current.space.grid_evaluation(x, y, size=size)
	lengths = np.sqrt(np.square(ux) + np.square(uy))
	plt.pcolormesh(xx, yy, lengths, alpha=.2)
	#plt.quiver(xx, yy, ux, uy, lengths, alpha=.3)
	max_length = np.max(lengths)
	line_width_function = (lengths/max_length)**2*4
	plt.streamplot(xx, yy, ux, uy, color=lengths, linewidth=line_width_function, arrowstyle='->', density=1.2)
	plt.plot(representer.current.curve[:,0], representer.current.curve[:,1], linewidth=5, alpha=.5, color='k')
	plt.axis('tight')
	plt.axis('equal')
	plt.colorbar()

def plot_invariants(current, threshold=1e-3):
	"""
	Plot the invariants of a current as follows.
	Each invariant is a pair $x_i:= ∫γ^* (φ_i dx), yi := ∫γ^* (φ_i dy)$
	We plot the points of coordinates $(x_i, y_i)$.
	Each non-zero point roughly corresponds to a gradient of the curve at the location
	of the basis function $φ_i$.
	"""
	inv_lengths = np.sqrt(np.sum(np.square(current.invariants), axis=1))
	mask = inv_lengths > threshold
	invs = current.invariants[mask]
	plt.plot(invs[:,0], invs[:,1],'.',alpha=.2)
	plt.axis('equal')
