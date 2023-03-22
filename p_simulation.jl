# p_simulation.jl

# Section 5: Experiments

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics

nPerDim = 100                       # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 12                          # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.
init = "uniform"           # How to generate initial data points for the matrix A. 
target = "heat_matlab"                   # Target function.
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, init, "Legendre", target)
uniform_prob = zeros(n, 1) .+ 1.0 / n   # Use this even inclusion probabilities to compare with the leverage score.


sampleSize = collect(70 : 10 : 400)
ntrial = 1000                            # Repeat the sampling and regression for ntrial times and take the median error.
sampleMethods = ["bernoulli", "btPivotalCoordwise", "btPivotalPCA"]

# Keep the median and std of errors. For each setting, we have a key-value pair and the vector stores the value for each sample size.
result_med = Dict{String, Vector{Float64}}()
result_std = Dict{String, Vector{Float64}}()

b = reshape(b_0, :, 1)
b_norm = mean(b .^ 2)

# A function that computes the normalized squared l2 error given the indices of sampled points and corresponding inclusion probability.
function eval(sample, prob)
    A_tilde = A[sample, :] ./ (prob .^ (1 / 2))
    b_tilde = b[sample] ./ (prob .^ (1 / 2))
    X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde
    error = mean((A * X_tilde - b) .^ 2) / b_norm
    return error
end

# Do the simulation.
for method in sampleMethods
    result_med[method * "_uniform"] = zeros(length(sampleSize))
    result_std[method * "_uniform"] = zeros(length(sampleSize))
    result_med[method * "_leverage"] = zeros(length(sampleSize))
    result_std[method * "_leverage"] = zeros(length(sampleSize))
    for i in eachindex(sampleSize)
        nsample = sampleSize[i]  # With Gaussian initialization, tau could be bigger than 1.0.
        binaryTree_uniform_coordwise = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample, "coordwise")
        binaryTree_leverage_coordwise = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "coordwise")
        binaryTree_uniform_PCA = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample, "PCA")
        binaryTree_leverage_PCA = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "PCA")
        errors_uniform, errors_leverage = zeros(ntrial), zeros(ntrial)
        for t in 1 : ntrial
            errorCount = 0
            while true          # Some sampling results cause SingularException in computing the inverse i.e. (A^T A)^-1.
                try             # In this case, redo the sampling.
                    if method == "bernoulli"
                        sample, prob = sampling.bernoulliSampling(uniform_prob, nsample)
                        errors_uniform[t] = eval(sample, prob)
                        sample, prob = sampling.bernoulliSampling(tau, nsample)
                        errors_leverage[t] = eval(sample, prob)
                    elseif method == "btPivotalCoordwise"
                        sample, prob = sampling.btPivotalSampling(binaryTree_uniform_coordwise, uniform_prob, nsample)
                        errors_uniform[t] = eval(sample, prob)
                        sample, prob = sampling.btPivotalSampling(binaryTree_leverage_coordwise, tau, nsample)
                        errors_leverage[t] = eval(sample, prob)
                    elseif method == "btPivotalPCA"
                        sample, prob = sampling.btPivotalSampling(binaryTree_uniform_PCA, uniform_prob, nsample)
                        errors_uniform[t] = eval(sample, prob)
                        sample, prob = sampling.btPivotalSampling(binaryTree_leverage_PCA, tau, nsample)
                        errors_leverage[t] = eval(sample, prob)
                    elseif method == "distPivotal"
                        sample, prob = sampling.distPivotalSampling(dist_mat, uniform_prob, nsample)
                        errors_uniform[t] = eval(sample, prob)
                        sample, prob = sampling.distPivotalSampling(dist_mat, tau, nsample)
                        errors_leverage[t] = eval(sample, prob)
                    end
                    break
                catch
                    errorCount = errorCount + 1
                    if errorCount > 10
                        throw(ErrorException("Error Count: $errorCount, method=$method, nsample=$nsample, t=$t"))
                    end
                end
            end
        end
        result_med[method * "_uniform"][i] = median(errors_uniform)
        result_std[method * "_uniform"][i] = std(errors_uniform)
        result_med[method * "_leverage"][i] = median(errors_leverage)
        result_std[method * "_leverage"][i] = std(errors_leverage)
    end
end

# Compute the error when using the entire data points.
error_full = eval(collect(1 : n), uniform_prob)

# Plot the results.
plot(title="", xlabel="Number of Samples", xlabelfontsize=10, ylabel="Median Relative Error", ylabelfontsize=10, 
     yaxis=:log, ylims=(2e-4, 2e-1), legend=:topright, size=(500, 400))
plot!(sampleSize, result_med["bernoulli_uniform"], label="Bernoulli, Uniform", lw=1, ls=:dash, lc=:orange, alpha=0.5)
plot!(sampleSize, result_med["bernoulli_leverage"], label="Bernoulli, Leverage", lw=3, ls=:dash, lc=:orange, alpha=0.5)
plot!(sampleSize, result_med["btPivotalCoordwise_uniform"], label="Pivotal Coordinate, Uniform", lw=1, lc=:blue, alpha=0.5)
plot!(sampleSize, result_med["btPivotalCoordwise_leverage"], label="Pivotal Coordinate, Leverage", lw=3, lc=:blue, alpha=0.5)
plot!(sampleSize, result_med["btPivotalPCA_uniform"], label="Pivotal PCA, Uniform", lw=1, lc=:green, alpha=0.5)
plot!(sampleSize, result_med["btPivotalPCA_leverage"], label="Pivotal PCA, Leverage", lw=3, lc=:green, alpha=0.5)
plot!(sampleSize, ones(Float64, length(sampleSize)) .* error_full, label="Optimal Error", lw=1, lc=:magenta, alpha=0.5)
savefig("sec5_heat.png")