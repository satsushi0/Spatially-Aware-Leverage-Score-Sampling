# binaryTreeVis.jl
# Visualize the binary tree construction.

include("./data.jl")
include("./sampling.jl")
using Plots

nPerDim = 50                        # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 5                           # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, "ChebyshevNodes", "Legendre", "spring")
uniform_prob = zeros(n, 1) .+ 1.0 / n

# Create the binary tree.
binaryTree = sampling.createBinaryTree(A[:, 2 : 2 + ndim - 1], uniform_prob, 1, "PCA", 30)
# Compute the depth of the tree.
exponent = ceil(Int64, log2(n))
# Only focus on the leaves as all the data points are stored in the leaves.
binaryTree = Int.(binaryTree[2 ^ (exponent) : 2 ^ (exponent + 1) - 1, 1])

# A function to remove 0 from an array. 0 is from the empty leaves.
function squeeze(vec)
    return vec[vec .> 0]
end

# Plot the binary tree with leaves in the same subtree are plotted with the same color.
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