# 3Dsimulation.jl
# Fit a polynomial function on the 3D spring distance target using sampling.

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics

nPerDim = 50                        # Number of points to generate for each coordinate.
ndim = 3                            # Dimensionality of the target function.
dpoly = 8                           # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.

initMethod = "grid"       # Specify how to generate initial data points from "grid", "Gaussian", "uniform", or "ChebyshevNodes".
# Generate the data matrix A, leverage score tau, and the target b_0.
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, initMethod, "Legendre", "spring")

# Visualization of the target function.
p = fill(plot(), 25, 1)
l = @layout [a{0.01h}; grid(4, 6)]
p[1] = plot(title="Spring Distance 3D", showaxis=false, grid=false)
fvals = sort(unique(A[:, 3]))
b = (b_0 .- minimum(b_0)) ./ (maximum(b_0) - minimum(b_0))
for iter in 1 : nPerDim
    fval = fvals[iter]
    _b = b[A[:, 3] .== fval]
    fval = round(fval + 1.0, digits=3)
    p[iter + 1] = plot(title="fval=$fval", titlefontsize=10, showaxis=false, legend=false)
    Plots.heatmap!(reshape(_b, nPerDim, nPerDim)', colormap=:turbo, clim=(0.0, 1.0))
end
plot(p..., layout=l, size=(1000, 750))
savefig("qoi_spring_3D.png")

# Visualization of the leverage score.
p = fill(plot(), 25, 1)
l = @layout [a{0.01h}; grid(4, 6)]
p[1] = plot(title="grid, n=$n, leverage score", showaxis=false, grid=false)
fvals = sort(unique(A[:, 3]))
for iter in 1 : nPerDim
    fval = fvals[iter]
    _tau = tau[A[:, 3] .== fval]
    fval = round(fval + 1.0, digits=3)
    p[iter + 1] = plot(title="fval=$fval", titlefontsize=10, showaxis=false, legend=false)
    Plots.heatmap!(reshape(_tau, nPerDim, nPerDim)', colormap=:turbo, clim=(0.0, maximum(tau)))
end
plot(p..., layout=l, size=(1000, 750))
savefig("init_grid_3D.png")


uniform_prob = zeros(n, 1) .+ 1.0 / n   # Use this even inclusion probabilities to compare with the leverage score.
sampleSize = collect(50 : 10 : 250)
ntrial = 100                            # Repeat the sampling and regression for ntrial times and take the median error.
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
    # dist_mat = sampling.distance(A[:, 2 : 2 + ndim - 1])
    for i in eachindex(sampleSize)
        nsample = sampleSize[i]
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
                    if errorCount > 5
                        throw(ErrorException("Error"))
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
title, name = "Spring3D | $initMethod", "plot_spring3D_$initMethod" * "_$dpoly.png"

plot(title="$title, n=$n, polydeg=$dpoly, d=$d", xlabel="# samples", ylabel="median normalized error", yaxis=:log, ylims=(1e-3, 1e1), legend=:left)
plot!(sampleSize, result_med["bernoulli_uniform"], label="bernoulli_uniform", lw=1, ls=:dash, lc=:orange)
plot!(sampleSize, result_med["bernoulli_leverage"], label="bernoulli_leverage", lw=4, ls=:dash, lc=:orange)
plot!(sampleSize, result_med["btPivotalCoordwise_uniform"], label="btPivotalCoordwise_uniform", lw=1, lc=:blue)
plot!(sampleSize, result_med["btPivotalCoordwise_leverage"], label="btPivotalCoordwise_leverage", lw=4, lc=:blue)
plot!(sampleSize, result_med["btPivotalPCA_uniform"], label="btPivotalPCA_uniform", lw=1, lc=:red)
plot!(sampleSize, result_med["btPivotalPCA_leverage"], label="btPivotalPCA_leverage", lw=4, lc=:red)
# plot!(sampleSize, result_med["distPivotal_uniform"], label="distPivotal_uniform", lw=1, lc=:green)
# plot!(sampleSize, result_med["distPivotal_leverage"], label="distPivotal_leverage", lw=4, lc=:green)
plot!(sampleSize, ones(Float64, length(sampleSize)) .* error_full, label="full_data", lw=1, lc=:magenta)
savefig("$name")