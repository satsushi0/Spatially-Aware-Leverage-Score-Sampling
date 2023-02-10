# studyFullGaussian.jl

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics
using VegaLite, DataFrames, FileIO

nPerDim = 100                       # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 16                          # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.
init = "Gaussian_nomanip"           # How to generate initial data points for the matrix A. 
target = "heat_matlab"                  # Target function.
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, init, "Legendre", target)
uniform_prob = zeros(n, 1) .+ 1.0 / n   # Use this even inclusion probabilities to compare with the leverage score.

sampleSize = collect(30 : 5 : 100)
ntrial = 100                            # Repeat the sampling and regression for ntrial times and take the median error.
sampleMethods = ["btPivotalCoordwise", "btPivotalPCA"]

errors = Dict{String, Matrix{Float64}}()

b = reshape(b_0, :, 1)
b_norm = mean(b .^ 2)

function eval(sample, prob)
    A_tilde = A[sample, :] ./ (prob .^ (1 / 2))
    b_tilde = b[sample] ./ (prob .^ (1 / 2))
    X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde
    error = mean((A * X_tilde - b) .^ 2) / b_norm
    return error
end

# Do the simulation.
checker = zeros(Int64, size(sampleSize, 1), ntrial)
for method in sampleMethods
    errors[method * "_leverage"] = zeros(Float64, size(sampleSize, 1), ntrial)
    for i in eachindex(sampleSize)
        nsample = sampleSize[i]
        binaryTree_leverage_coordwise = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "coordwise")
        binaryTree_leverage_PCA = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "PCA")
        for t in 1 : ntrial
            errorCount = 0
            while true          # Some sampling results cause SingularException in computing the inverse i.e. (A^T A)^-1.
                try             # In this case, redo the sampling.
                    if method == "btPivotalCoordwise"
                        sample, prob = sampling.btPivotalSampling(binaryTree_leverage_coordwise, tau, nsample)
                        errors[method * "_leverage"][i, t] = eval(sample, prob)
                        checker[i, t] = length(sample)
                    elseif method == "btPivotalPCA"
                        sample, prob = sampling.btPivotalSampling(binaryTree_leverage_PCA, tau, nsample)
                        errors[method * "_leverage"][i, t] = eval(sample, prob)
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
    end
end

# Compute the error when using the entire data points.
error_full = eval(collect(1 : n), uniform_prob)

# plot(title="Heat (M) | $init, polydeg=$dpoly", xlabel="# samples", ylabel="median normalized error", yaxis=:log, ylims=(1e-3, 1e3), legend=:right)
# plot!(sampleSize, median(errors["btPivotalCoordwise_leverage"], dims=2), label="btPivotalCoordwise_leverage", lw=4, lc=:blue)
# plot!(sampleSize, median(errors["btPivotalPCA_leverage"], dims=2), label="btPivotalPCA_leverage", lw=4, lc=:red)
# plot!(sampleSize, ones(Float64, length(sampleSize)) .* error_full, label="full_data", lw=1, lc=:magenta)
# savefig("studyFullGaussian_mean.png")

plot_index = collect(2 : 15)
plot(title="Error Distribution | btPivotalCoordwise, dpoly=$dpoly", xlabel="# samples", ylabel="normalized error", yaxis=:log, legend=:top)
plot!(sampleSize[plot_index], mean(errors["btPivotalCoordwise_leverage"], dims=2)[plot_index], label="mean", lw=1, lc=:orange)
plot!(sampleSize[plot_index], median(errors["btPivotalCoordwise_leverage"], dims=2)[plot_index], label="median", lw=1, lc=:blue)
for i in eachindex(plot_index)
    scatter!(ones(ntrial) * sampleSize[plot_index[i]], errors["btPivotalCoordwise_leverage"][plot_index[i], :], label="", markeralpha=0.5, markerstrokewidth=0, mc=:purple)
end
plot!()
savefig("studyFullGaussian_dist_$dpoly.png")

nsample = 250
binaryTree_leverage_coordwise = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "coordwise")
sample, prob = sampling.btPivotalSampling(binaryTree_leverage_coordwise, tau, nsample)
A_tilde = A[sample, :] ./ (prob .^ (1 / 2))
b_tilde = b[sample] ./ (prob .^ (1 / 2))
X_tilde = (A_tilde' * A_tilde) \ A_tilde' * b_tilde
b_1 = A * X_tilde
X_star = (A' * A) \ A' * b
b_star = A * X_star
error = mean((b_1 - b) .^ 2) / b_norm
df = DataFrame(x=A[:, 2], y=A[:, 3], qoi=b_0)
df |> @vlplot(mark={type=:point, filled=true, size=20, opacity=0.5}, x=:x, y=:y, color={:qoi, scale={scheme=:turbo, domain=[0.0, 2.0]}}, 
              width=400, height=400, title="True Values") |> FileIO.save("studyFullGaussian_true_$dpoly.png")
df = DataFrame(x=A[:, 2], y=A[:, 3], est=b_star[:, 1])
df |> @vlplot(mark={type=:point, filled=true, size=20, opacity=0.5}, x=:x, y=:y, color={:est, scale={scheme=:turbo, domain=[0.0, 2.0]}}, 
            width=400, height=400, title="dpoly=$dpoly, x_star")|> FileIO.save("studyFullGaussian_eststar_$dpoly" * "_$nsample.png")
df = DataFrame(x=A[:, 2], y=A[:, 3], est=clamp.(b_1, 0.0, 2.0))
df |> @vlplot(mark={type=:point, filled=true, size=20, opacity=0.5}, x=:x, y=:y, color={:est, scale={scheme=:turbo, domain=[0.0, 2.0]}}, 
              width=400, height=400, title="dpoly=$dpoly, k=$nsample, Error=$error")|> FileIO.save("studyFullGaussian_est_$dpoly" * "_$nsample.png")
df = DataFrame(x=A[:, 2], y=A[:, 3], error=clamp.(b_1-b_0, -1.0, 1.0))
df |> @vlplot(mark={type=:point, filled=true, size=20, opacity=0.5}, x=:x, y=:y, color={:error, scale={scheme=:redblue, domain=[-1.0, 1.0]}}, 
              width=400, height=400, title="Estimation Error | dpoly=$dpoly, k=$nsample")|> FileIO.save("studyFullGaussian_diff_$dpoly" * "_$nsample.png")

# The number of data points vs leverage score.
ndim = 2                            # Dimensionality of the target function.
init = "Gaussian_nomanip"           # How to generate initial data points for the matrix A. 
polyType = "Legendre"
dpolys = [5, 10, 15, 20]
nPerDims = [20, 50, 100, 200, 300]
threshold = 0.9
rates_mean = zeros(Float64, length(dpolys), length(nPerDims))
rates_std = zeros(Float64, length(dpolys), length(nPerDims))
nTrial = 50
for i in eachindex(dpolys)
    dpoly = dpolys[i]
    for j in eachindex(nPerDims)
        nPerDim = nPerDims[j]
        result = zeros(Float64, nTrial)
        for t in 1 : nTrial
            if nPerDim > 150
                n = nPerDim ^ ndim
                println("dpoly = $dpoly | n = $n | t = $t")
            end
            A = data._generateA(ndim, nPerDim, dpoly, init, polyType)
            tau = data._leverageScore(A)
            result[t] = sum(tau .> threshold)
        end
        rates_mean[i, j] = mean(result)
        rates_std[i, j] = std(result)
    end
end

plot(xlabel="# data points (n)", ylabel="# leverage score > $threshold", title="Gaussian and leverage score, $polyType", xaxis=:log)
for i in eachindex(dpolys)
    dpoly = dpolys[i]
    plot!(nPerDims .^ ndim, rates_mean[i, :], ribbon=rates_std[i, :], label="dpoly=$dpoly")
end
plot!()
savefig("n_vs_highlev_$polyType.png")

dpoly = 15
nPerDim = 50
n = nPerDim ^ ndim
A = data._generateA(ndim, nPerDim, dpoly, init, polyType)
tau = data._leverageScore(A)
highlev = sum(tau .> threshold)
df = DataFrame(x=A[:, 2], y=A[:, 3], tau=tau[:, 1])
df |> @vlplot(mark={type=:point, filled=true, size=20, opacity=0.5}, x=:x, y=:y, color={:tau, scale={scheme=:turbo, domain=[0.0, 1.0]}}, 
              width=400, height=400, title="Leverage Score | n=$n, dpoly=$dpoly") |> FileIO.save("FullGaussian_leverge_score_$dpoly" * "_$n" * ".png")