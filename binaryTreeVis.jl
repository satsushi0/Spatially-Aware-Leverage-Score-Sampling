# binaryTreeVis.jl

include("./data.jl")
include("./sampling.jl")
using Plots

nPerDim = 50
ndim = 2
dpoly = 5
n = nPerDim ^ ndim
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, "ChebyshevNodes", "Legendre", "spring")
uniform_prob = zeros(n, 1) .+ 1.0 / n

binaryTree = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, 1, "PCA", 30)
exponent = ceil(Int64, log2(n))
binaryTree = Int.(binaryTree[2 ^ (exponent) : 2 ^ (exponent + 1) - 1, 1])

function squeeze(vec)
    return vec[vec .> 0]
end

p = fill(plot(), 9, 1)
l = @layout [a{0.01h}; grid(2, 4)]
p[1] = plot(title="Binary Tree, Gaussian, PCA30", showaxis=false, grid=false)
for iter in 1 : 8
    step = 2 ^ (exponent - iter)
    p[iter + 1] = plot(title="depth=$iter", titlefontsize=10, showaxis=false)
    for i in 1 : 2 ^ iter
        A_frac = A[squeeze(binaryTree[(i - 1) * step + 1 : i * step]), 2 : 3]
        plot!(A_frac[:, 1], A_frac[:, 2], shape=:rect, markerstrokewidth=0, ms=1, seriestype=:scatter, legend=false, palette=:glasbey_category10_n256)
    end
end
plot(p..., layout=l, size=(1000, 500))
savefig("bt_Gaussian_pca30.png")