# README #

This is a Python library for computing shape invariants of planar curves using FEM and the FEniCS package. To run the module, [FEniCS version 1.5](http://fenicsproject.org) must be installed.

## Getting started ##

The simplest way to calculate the invariants of a curve are as follows.

1. Create a space

```python
space = Space()
```
Check the documentation of the `Space` class to see the available options.

2. Create a curve

This is just an arbitrary array of size P x 2, where P is the number of points.
For instance,
```python
ts = np.arange(0, 2*np.pi, 200)
curve = .7*np.array([np.cos(ts), np.sin(3*ts)]).T
```

3. Compute the associated current

```python
current = Current(space, curve)
```

You have now access to the property `invariants` which contains the result of the current evaluated on the basis of one-forms.

You can inspect the invariants using
```python
plot_invariants(current)
```

<img alt="invariants" src="https://raw.githubusercontent.com/olivierverdier/femshape/master/invariants.png"/>

4. Compute the representer

Givne an underlying Hilbert space structure on the one forms, one can compute the associated representer of the current:

```python
representer = Representer(current, scale=.2)
```

You can plot the result using

```python
plot_representer(representer)
```

<img alt="representer" src="https://raw.githubusercontent.com/olivierverdier/femshape/master/representer.png"/>

## Further examples

The notebooks contain further examples, especially using PCA to differentiate shapes.


