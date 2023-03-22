# p_target_approx.jl

# Section 1.1: Our Contribution / Section 5: Experiments

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics
using VegaLite, DataFrames, FileIO

nPerDim = 100                       # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 12                          # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.
init = "grid"           # How to generate initial data points for the matrix A. 
target = "heat_matlab"                  # Target function.
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, init, "Legendre", target)
uniform_prob = zeros(n, 1) .+ 1.0 / n   # Use this even inclusion probabilities to compare with the leverage score.
b = reshape(b_0, :, 1)
b_norm = mean(b .^ 2)

nsample = 125
sample_method = "PCA"
incl_prob = tau
ntrial = 100

# Without sampling. Using all data points.
X_star = (A' * A) \ A' * b
b_star = A * X_star

binaryTree = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], incl_prob, nsample, bt_method)
errors = zeros(Float64, ntrial)
b_1s = zeros(Float64, ntrial, n)

for t in 1 : ntrial
    if sample_method == "PCA"
        sample, prob = sampling.btPivotalSampling(binaryTree, incl_prob, nsample)
    elseif sample_method == "bernoulli"
        sample, prob = sampling.bernoulliSampling(incl_prob, nsample)
    end
    A_tilde = A[sample, :] ./ (prob .^ (1 / 2))
    b_tilde = b[sample] ./ (prob .^ (1 / 2))
    X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde
    b_1 = A * X_tilde
    error = mean((b_1 - b) .^ 2) / b_norm
    errors[t] = error
    b_1s[t, :] = b_1
end

b_1s = b_1s[sortperm(errors), :]
b_1_mean = b_1s[ntrial ÷ 2, :]

plot(xlabel="time", xlabelfontsize=12, 
     ylabel="starting frequency", ylabelfontsize=12, 
     size=(450, 400))
heatmap!(LinRange(-1, 1, nPerDim), LinRange(-1, 1, nPerDim), reshape(b_0, nPerDim, nPerDim)' , c=:turbo, clim=(0.0, 2.3))
savefig("sec5_heat_target.png")
