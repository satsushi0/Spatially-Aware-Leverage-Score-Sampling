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