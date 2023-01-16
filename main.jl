# main.jl
# Multi-dimensional polynomial regression with spatially-aware leverage score sampling.

include("./data.jl")
using Plots

nPerDim = 100
ndim = 2
dpoly = 5
n = nPerDim^ndim
d = binomial(dpoly + ndim, ndim)
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, "Legendre")

heatmap(1 : nPerDim, 1 : nPerDim, b_0)

