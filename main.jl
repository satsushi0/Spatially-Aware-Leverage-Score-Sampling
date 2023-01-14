# main.jl
# Multi-dimensional polynomial regression with spatially-aware leverage score sampling.

nPerDim = 100
ndim = 2
dpoly = 5
n = nPerDim^ndim
d = binomial(dpoly + ndim, ndim)