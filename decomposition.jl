# decomposition.jl

using Plots

include("./data.jl")
nPerDim = 100                       # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 5                          # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.
init = "grid"                       # How to generate initial data points for the matrix A. 
polyType = "Legendre"
A = data._generateA(ndim, nPerDim, dpoly, init, polyType)
tau = data._leverageScore(A)

unique = unique!(A[:, 2])

ls_sum = zeros(Float64, nPerDim)
for i in 1 : nPerDim
    ls_sum[i] = sum(tau[A[:, 2] .== unique[i]])
end
ls_sum = ls_sum / d

A1 = data._generateA(1, nPerDim, dpoly, init, polyType)
tau1 = data._leverageScore(A1)
tau1 = tau1 / size(A, 2)
ls_sum

approx = 1 ./ sqrt.(1 .- LinRange(-0.99, 0.99, nPerDim) .^ 2)
approx ./= sum(approx)

plot(A1[:, 2], ls_sum, label="sum")
plot!(A1[:, 2], tau1 * 21 / 6, label="1d")
plot!(A1[:, 2], approx, label="approx")
savefig("decom.png")