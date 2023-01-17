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
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, "Legendre")
uniform_prob = zeros(n, 1) .+ 1.0 / n

heatmap(1 : nPerDim, 1 : nPerDim, b_0)

sampleSize = [30, 35, 40, 45, 50, 60, 70]
ntrial = 100
sampleMethods = ["bernoulli", "btPivotal"]

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
    for i in 1 : length(sampleSize)
        nsample = sampleSize[i]
        binaryTree_uniform = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample)
        binaryTree_leverage = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample)
        errors_uniform, errors_leverage = zeros(ntrial), zeros(ntrial)
        for t in 1 : ntrial
            if method == "bernoulli"
                sample, prob = sampling.bernoulliSampling(uniform_prob, nsample)
                errors_uniform[t] = eval(sample, prob)
                sample, prob = sampling.bernoulliSampling(tau, nsample)
                errors_leverage[t] = eval(sample, prob)
            elseif method == "btPivotal"
                sample, prob = sampling.btPivotalSampling(binaryTree_uniform, uniform_prob, nsample)
                errors_uniform[t] = eval(sample, prob)
                sample, prob = sampling.btPivotalSampling(binaryTree_leverage, tau, nsample)
                errors_leverage[t] = eval(sample, prob)
            end
        end
        result_med[method * "_uniform"][i] = median(errors_uniform)
        result_std[method * "_uniform"][i] = std(errors_uniform)
        result_med[method * "_leverage"][i] = median(errors_leverage)
        result_std[method * "_leverage"][i] = std(errors_leverage)
    end
end

plot(sampleSize, result_med["bernoulli_uniform"], label="bernoulli_uniform", lw=1, ls=:dash, lc=:orange, yaxis=:log)
plot!(sampleSize, result_med["bernoulli_leverage"], label="bernoulli_leverage", lw=4, ls=:dash, lc=:orange)
plot!(sampleSize, result_med["btPivotal_uniform"], label="btPivotal_uniform", lw=1, lc=:blue)
plot!(sampleSize, result_med["btPivotal_leverage"], label="btPivotal_leverage", lw=4, lc=:blue)

# nsample = 50
# sample, _ = sampling.bernoulliSampling(uniform_prob, nsample)
# sample, _ = sampling.bernoulliSampling(tau, nsample)
# binaryTree_uniform = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample)
# binaryTree_leverage = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample)
# sample, prob = sampling.btPivotalSampling(binaryTree_uniform, uniform_prob, nsample)
# sample, prob = sampling.btPivotalSampling(binaryTree_leverage, tau, nsample)
# plot(A[sample, 2], A[sample, 3], seriestype=:scatter, label="btPivotal_uniform")
# savefig("btPivotal_uniform.png")