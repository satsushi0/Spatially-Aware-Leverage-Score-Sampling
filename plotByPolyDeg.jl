# plotByPolyDeg.jl
# Visualize the changes in error with different polynomial degree for regression.

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics

nPerDim = 100                       # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 5                           # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.
init = "ChebyshevNodes"             # How to generate initial data points for the matrix A. 
target = "heat"                     # Target function.
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, "grid", "Legendre", target)
sampleSize = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
ntrial = 100                        # Repeat the sampling and regression for ntrial times and take the median error.

b = reshape(b_0, :, 1)
b_norm = mean(b .^ 2)

result_med = Dict{String, Vector{Float64}}()

function eval(sample, prob)
    A_tilde = A[sample, :] ./ (prob .^ (1 / 2))
    b_tilde = b[sample] ./ (prob .^ (1 / 2))
    X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde
    error = mean((A * X_tilde - b) .^ 2) / b_norm
    return error
end

# Do the simulation.
dpolys = [3, 5, 7, 9, 11, 13, 15]
ds = ones(Int64, length(dpolys))
for i in eachindex(dpolys)
    dpoly = dpolys[i]
    result_med["$dpoly"] = zeros(length(sampleSize))
    A = data._generateA(ndim, nPerDim, dpoly, "grid", "Legendre")
    ds[i] = size(A, 2)
    for j in eachindex(sampleSize)
        nsample = sampleSize[j]
        binaryTree = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "coordwise")
        errors = zeros(ntrial)
        for t in 1 : ntrial
            sample, prob = sampling.btPivotalSampling(binaryTree, tau, nsample)
            errors[t] = eval(sample, prob)
        end
        result_med["$dpoly"][j] = median(errors)
    end
end

# Plot the result.
plot(title="Heat Equation, n=$n, leverage score, coordwise", xlabel="# samples", ylabel="median normalized error", 
     yaxis=:log, ylims=(1e-4, 1.0), legendfontsize=7)
for i in eachindex(dpolys)
    dpoly = dpolys[i]
    d = ds[i]
    plot!(sampleSize, result_med["$dpoly"], label="dpoly=$dpoly, d=$d", lw=(dpoly - 1) / 2)
end
plot!()
savefig("plot_heat_polydeg2.png")