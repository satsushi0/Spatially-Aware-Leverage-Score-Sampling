# QR_check.jl

using Plots
using Combinatorics
using LinearAlgebra


# 1D case.

n = 100
dpoly = 5
x = collect(LinRange(-1.0, 1.0, n))
A = ones(Float64, n)

for i in 1 : dpoly
    # A = [A A[:, size(A, 2)] .* x]
    if i == 1
        A = [A A[:, size(A, 2)] .* x]
    else
        A = [A A[:, size(A, 2)] .* A[:, 2]]
    end
    Q, _ = qr(A)
    A = Q[:, 1 : size(A, 2)]
end

plot(xlabel="x value (original polynomial degree 1 values)", ylabel="polynomial value", title="QR factorization")
for i in 0 : dpoly
    plot!(x, A[:, i + 1], label="p_$i", lw=2)
end
plot!()
savefig("QR_$dpoly.png")

heatmap(A' * A, yflip=true, colormap=:grays, title="A^T * A | dpoly=$dpoly", size=(420, 400))
savefig("QR_orthogonality.png")


# Multi-dimensional case.

ndim = 3
nPerDim = 25
dpoly = 5
n = nPerDim ^ ndim
d = binomial(dpoly + ndim, ndim) 
base = zeros(ndim, n) 
P = zeros(Int64, n, ndim)
for i in 1 : ndim
    for j in 1 : nPerDim
        for k = 1 : nPerDim ^ (ndim - i)
            P[(j - 1) * nPerDim ^ (ndim - i) + k, i] = j
        end    
    end
    for l in 1 : nPerDim ^ (i - 1)
        P[(l - 1) * nPerDim * nPerDim ^ (ndim - i) + 1 : l * nPerDim * nPerDim ^ (ndim - i), i] = P[1 : nPerDim * nPerDim ^ (ndim - i), i]
    end
end
for i in 1 : ndim
    base[i, :] = LinRange(-1, 1, nPerDim)[P[:, i]]
end

C = collect(combinations(1 : dpoly + ndim, ndim))
C = reduce(vcat, C')
for i in ndim : -1 : 2
    C[:, i] = C[:, i] .- C[:, i - 1]
end
order = sortperm(sum(C, dims=2)[:, 1])
C = C[order, :] .- 1

A = ones(Float64, n)

function decompose(i)
    _, base_ix = findmax(C[i, :])
    QR_ix = 0
    target = C[i, :]
    target[base_ix] -= 1
    for j in i : -1 : 1
        if C[j, :] == target
            QR_ix = j
            break
        end
    end
    return QR_ix, base_ix
end

for i in 2 : d
    QR_ix, base_ix = decompose(i)
    A = [A A[:, QR_ix] .* base[base_ix, :]]
    Q, _ = qr(A)
    A = Q[:, 1 : size(A, 2)]
end


# Check the implementation.
include("./data.jl")
A = data._generateA(ndim, nPerDim, dpoly, "grid", "QR")

p = fill(plot(), 22, 1)
l = @layout [a{0.01h}; grid(3, 7)]
p[1] = plot(title="QR Polynomial in 2D", showaxis=false, grid=false)
for iter in 1 : 21
    y, x = C[iter, :]
    p[iter + 1] = plot(title="x^$x, y^$y", titlefontsize=10, showaxis=false)
    disp = reshape(A[:, iter], 100, 100)'
    heatmap!(disp, yflip=true, legend=false, colormap=:redgreensplit, clim=(minimum(A), maximum(A)), axis=false, colorbar=false)
end
plot(p..., layout=l, size=(1000, 500))
savefig("QR2D.png")

heatmap(A' * A, yflip=true, colormap=:grays, title="A^T * A | dpoly=$dpoly", size=(420, 400))
savefig("QR2D_orthogonality.png")


# Compare the performance with different polynomial generators.

include("./data.jl")
include("./sampling.jl")
using Plots
using Statistics

nPerDim = 100                       # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 30                          # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.
init = "grid"                       # How to generate initial data points for the matrix A. 
target = "spring"              # Target function.
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, init, "Legendre", target)
uniform_prob = zeros(n, 1) .+ 1.0 / n   # Use this even inclusion probabilities to compare with the leverage score.

sampleSize = collect(350 : 10 : 500)
ntrial = 100                            # Repeat the sampling and regression for ntrial times and take the median error.
method = "bernoulli"
polyTypes = ["Legendre", "Chebyshev", "None", "QR"]

result_med = Dict{String, Vector{Float64}}()

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

dist_mat = sampling.distance(A[:, 2 : 2 + ndim - 1])
for polyType in polyTypes
    println("Simulating with " * polyType)
    A = data._generateA(ndim, nPerDim, dpoly, init, polyType)
    tau = data._leverageScore(A)
    result_med[polyType * "_uniform"] = zeros(length(sampleSize))
    result_med[polyType * "_leverage"] = zeros(length(sampleSize))
    for i in eachindex(sampleSize)
        nsample = sampleSize[i]  # With Gaussian initialization, tau could be bigger than 1.0.
        println("   Sample Size: " * nsample)
        # binaryTree_uniform_coordwise = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample, "coordwise")
        # binaryTree_leverage_coordwise = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "coordwise")
        # binaryTree_uniform_PCA = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, nsample, "PCA")
        # binaryTree_leverage_PCA = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], tau, nsample, "PCA")
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
        result_med[polyType * "_uniform"][i] = median(errors_uniform)
        result_med[polyType * "_leverage"][i] = median(errors_leverage)
    end
end

# Compute the error when using the entire data points.
error_full = eval(collect(1 : n), uniform_prob)

# Plot the results.
if target == "spring"
    title, name = "Spring | $init, $method", "plot_spring_$init" *"_$method" * "_$dpoly.png"
elseif target == "heat"
    title, name = "Heat | $init", "plot_heat_$init" * "_$dpoly.png"
elseif target == "heat_matlab"
    title, name = "Heat (M) | $init, $method", "plot_heatM_$init" *"_$method" * "_$dpoly.png"
end
plot(title="$title, n=$n, polydeg=$dpoly, d=$d", xlabel="# samples", ylabel="median normalized error", yaxis=:log, ylims=(1e-3, 1e1), legend=:topright)
plot!(sampleSize, result_med["None_uniform"], label="None_uniform", lw=1, ls=:dash, lc=:orange)
plot!(sampleSize, result_med["None_leverage"], label="None_leverage", lw=4, ls=:dash, lc=:orange)
plot!(sampleSize, result_med["Chebyshev_uniform"], label="Chebyshev_uniform", lw=1, lc=:blue)
plot!(sampleSize, result_med["Chebyshev_leverage"], label="Chebyshev_leverage", lw=4, lc=:blue)
plot!(sampleSize, result_med["Legendre_uniform"], label="Legendre_uniform", lw=1, lc=:green)
plot!(sampleSize, result_med["Legendre_leverage"], label="Legendre_leverage", lw=4, lc=:green)
plot!(sampleSize, result_med["QR_uniform"], label="QR_uniform", lw=1, lc=:red)
plot!(sampleSize, result_med["QR_leverage"], label="QR_leverage", lw=4, lc=:red)
plot!(sampleSize, ones(Float64, length(sampleSize)) .* error_full, label="full_data", lw=1, lc=:magenta)
savefig("$name")