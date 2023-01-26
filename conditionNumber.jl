# conditionNumber.jl
# Condition number study.

include("./data.jl")
using Plots
using LinearAlgebra

ndim = 2
nPerDim = 100
dpoly = 10
d = binomial(dpoly + ndim, ndim) 

ntrial = 100
inits = ["grid", "Gaussian_nomanip", "Gaussian", "Gaussian_range", "uniform", "ChebyshevNodes"]
polys = ["Legendre", "Chebyshev", "None"]

kappa = zeros(Float64, length(inits), length(polys))

for i in eachindex(inits)
    init = inits[i]
    for j in eachindex(polys)
        poly = polys[j]
        cond = zeros(Float64, ntrial)
        for t in 1 : ntrial
            A = data._generateA(ndim, nPerDim, dpoly, init, poly)
            eigenvalues = sort(eigvals(A' * A))
            cond[t] = eigenvalues[d] / eigenvalues[1]
        end
        kappa[i, j] = mean(cond)
    end
end