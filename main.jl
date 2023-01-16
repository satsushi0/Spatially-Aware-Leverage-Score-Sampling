# main.jl
# Multi-dimensional polynomial regression with spatially-aware leverage score sampling.

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics

nPerDim = 100
ndim = 2
dpoly = 5
n = nPerDim^ndim
d = binomial(dpoly + ndim, ndim)
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, "Legendre")
uniform_prob = zeros(n, 1) .+ 1.0 / n

heatmap(1 : nPerDim, 1 : nPerDim, b_0)

sampleSize = [30, 35, 40, 45, 50, 60, 70]
ntrial = 100
result_med = Dict{String, Vector{Float64}}()
result_std = Dict{String, Vector{Float64}}()

b = reshape(b_0, :, 1)
b_norm = mean(b.^2)

function eval(sample, prob)
    A_tilde = A[sample, :] ./ (prob.^(1 / 2))
    b_tilde = b[sample] ./ (prob.^(1 / 2))
    X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde
    error = mean((A * X_tilde - b).^2) / b_norm
    return error
end


# Bernoulli sampling

result_med["bernoulli_uniform"] = zeros(length(sampleSize))
result_std["bernoulli_uniform"] = zeros(length(sampleSize))
result_med["bernoulli_leverage"] = zeros(length(sampleSize))
result_std["bernoulli_leverage"] = zeros(length(sampleSize))

for i in 1 : length(sampleSize)
    nsample = sampleSize[i]
    error_uniform = zeros(ntrial)
    error_leverage = zeros(ntrial)
    for j in 1 : ntrial
        sample, prob = sampling.bernoulliSampling(uniform_prob, nsample)
        error_uniform[j] = eval(sample, prob)
        sample, prob = sampling.bernoulliSampling(tau, nsample)
        error_leverage[j] = eval(sample, prob)
    end
    result_med["bernoulli_uniform"][i] = median(error_uniform)
    result_std["bernoulli_uniform"][i] = std(error_uniform)
    result_med["bernoulli_leverage"][i] = median(error_leverage)
    result_std["bernoulli_leverage"][i] = std(error_leverage)
end

plot(sampleSize, [result_med["bernoulli_uniform"], 
                  result_med["bernoulli_leverage"]], yaxis=:log)