# p_sample_vis.jl

# Section 5: Experiments

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics

nPerDim = 50                       # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 12                          # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.
init = "uniform"           # How to generate initial data points for the matrix A. 
target = "spring"                   # Target function.
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, init, "Legendre", target)
uniform_prob = zeros(n, 1) .+ 1.0 / n   # Use this even inclusion probabilities to compare with the leverage score.

nsample = 100
method = "bernoulli"
incl_prob = uniform_prob
binaryTree_PCA = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], incl_prob, nsample, "PCA")
if method == "bernoulli"
    sample, prob = sampling.bernoulliSampling(incl_prob, nsample)
    while length(sample) != nsample
        sample, prob = sampling.bernoulliSampling(incl_prob, nsample)
    end
elseif method == "PCA"
    sample, prob = sampling.btPivotalSampling(binaryTree_PCA, incl_prob, nsample)
end
sampled_A = A[sample, :]

plot(size=(400, 400))
# plot!(A[:, 2], A[:, 3], markerstrokewidth=0, ms=1, seriestype=:scatter, legend=false, color="gray")
plot!(sampled_A[:, 2], sampled_A[:, 3], markerstrokewidth=0, ms=4, 
      seriestype=:scatter, legend=false, color="black", grid=false, alpha=0.5)
savefig("sec5_scatter_bernoulli_uniform.png")