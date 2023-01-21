# main.jl
# Multi-dimensional polynomial regression with spatially-aware leverage score sampling.

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics

nPerDim = 100
ndim = 2
dpoly = 5
n = nPerDim ^ ndim
d = binomial(dpoly + ndim, ndim)
target = "heat"
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, "grid", "Legendre", target)
uniform_prob = zeros(n, 1) .+ 1.0 / n

if target == "spring"
    heatmap(LinRange(-1, 1, nPerDim), LinRange(-1, 1, nPerDim), b_0, 
            xlabel="spring constant, \$k\$", 
            ylabel="driving frequency, \$\\omega\$")
    savefig("qoi_spring.png")
elseif target == "heat"
    heatmap(LinRange(0, 1, nPerDim), LinRange(0, 5, nPerDim), b_0, 
            xlabel="time, \$t\$", 
            ylabel="starting frequency, \$f\$")
    savefig("qoi_heat.png")
end

sampleSize = [30, 35, 40, 45, 50, 60, 70]
ntrial = 100
sampleMethods = ["bernoulli", "btPivotalCoordwise", "btPivotalPCA"]

result_med = Dict{String, Vector{Float64}}()
result_std = Dict{String, Vector{Float64}}()

b = reshape(b_0, :, 1)
b_norm = mean(b .^ 2)

function eval(sample, prob)
    A_tilde = A[sample, :] ./ (prob .^ (1 / 2))
    b_tilde = b[sample] ./ (prob .^ (1 / 2))
    X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde
    error = mean((A * X_tilde - b) .^ 2) / b_norm
    return error
end

for method in sampleMethods
    result_med[method * "_uniform"] = zeros(length(sampleSize))
    result_std[method * "_uniform"] = zeros(length(sampleSize))
    result_med[method * "_leverage"] = zeros(length(sampleSize))
    result_std[method * "_leverage"] = zeros(length(sampleSize))
    for i in eachindex(sampleSize)
        nsample = sampleSize[i]
        binaryTree_uniform_coordwise = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample, "coordwise")
        binaryTree_leverage_coordwise = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "coordwise")
        binaryTree_uniform_PCA = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample, "PCA")
        binaryTree_leverage_PCA = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "PCA")
        errors_uniform, errors_leverage = zeros(ntrial), zeros(ntrial)
        for t in 1 : ntrial
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
            end
        end
        result_med[method * "_uniform"][i] = median(errors_uniform)
        result_std[method * "_uniform"][i] = std(errors_uniform)
        result_med[method * "_leverage"][i] = median(errors_leverage)
        result_std[method * "_leverage"][i] = std(errors_leverage)
    end
end

if target == "spring"
    title, name = "Spring Distance", "plot_spring.png"
elseif target == "heat"
    title, name = "Heat Equation", "plot_heat.png"
end
plot(sampleSize, result_med["bernoulli_uniform"], label="bernoulli_uniform", lw=1, ls=:dash, lc=:orange, yaxis=:log, 
     title="$title, n=$n, polydeg=$dpoly, d=$d", xlabel="# samples", ylabel="median normalized error")
plot!(sampleSize, result_med["bernoulli_leverage"], label="bernoulli_leverage", lw=4, ls=:dash, lc=:orange)
plot!(sampleSize, result_med["btPivotalCoordwise_uniform"], label="btPivotalCoordwise_uniform", lw=1, lc=:blue)
plot!(sampleSize, result_med["btPivotalCoordwise_leverage"], label="btPivotalCoordwise_leverage", lw=4, lc=:blue)
plot!(sampleSize, result_med["btPivotalPCA_uniform"], label="btPivotalPCA_uniform", lw=1, lc=:green)
plot!(sampleSize, result_med["btPivotalPCA_leverage"], label="btPivotalPCA_leverage", lw=4, lc=:green)
savefig("$name")

# nsample = 50
# sample, _ = sampling.bernoulliSampling(uniform_prob, nsample)
# sample, _ = sampling.bernoulliSampling(tau, nsample)
# binaryTree_uniform = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample)
# binaryTree_leverage = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample)
# sample, prob = sampling.btPivotalSampling(binaryTree_uniform, uniform_prob, nsample)
# sample, prob = sampling.btPivotalSampling(binaryTree_leverage, tau, nsample)
# plot(A[sample, 2], A[sample, 3], seriestype=:scatter, label="btPivotal_uniform")
# savefig("btPivotal_uniform.png")
