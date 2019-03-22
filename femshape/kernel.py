import numpy as np

def get_velocities(points):
	"""
	First order approximation of tangent vectors of the curve.
	The points must clearly be in the right order.
	"""
	shape = (2, len(points.T)+1)
	fig8_ = np.zeros(shape)
	fig8_[:,:-1] = points
	fig8_[:,-1] = points[:,0]
	vels = fig8_[:,1:] - fig8_[:,:-1]
	return vels

import torch

def point_features(curve, points, kernel):
	"""
	curve: [2 x N] array
	points: [2 x M] array
	returns: [M x 2] feature array
	Compute the features of a curve with respect to a set of points.
	The curve has to be a curve (i.e., with ordered points),
	but the points are not ordered (they are typically on a grid).

	If P is a point, γ is the discretised current corresponding to the curve,
	we compute α(K(P)dx).

	"""
	velocities = get_velocities(curve)
	features = kernel.convolve(torch.from_numpy(points.T), torch.from_numpy(curve.T), torch.from_numpy(velocities.T))
	return np.asarray(features)
