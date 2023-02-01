# surfaceReaction.jl
# As third qoi, consider a surface function model.
# https://arxiv.org/pdf/1410.1931.pdf Details are in section 4.3.

using Plots
using ODE

include("./data.jl")
nPerDim = 100                       # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 2                          # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.
d = binomial(dpoly + ndim, ndim)    # Number of features. The matrix A has size n by d.
init = "grid"                       # How to generate initial data points for the matrix A. 
polyType = "Legendre"
A = data._generateA(ndim, nPerDim, dpoly, init, polyType)
A = A[:, 2 : 3]

alpha = 0.1
gamma = 0.011
kappa = 10.0
tmax = 4.0
rho_init = 0.9

function surfaceReaction(t, rho)
    deriv = alpha * (1.0 - rho) - gamma * rho - kappa * (1.0 - rho) ^ 2 * rho
    return deriv
end

b_0 = zeros(Float64, n)
for i in 1 : n
    alpha = 0.1 + exp(0.05 * A[i, 1])
    gamma = 0.001 + 0.01 * exp(0.05 * A[i, 2])
    t, y = ode45(surfaceReaction, rho_init, [0.0, tmax])
    b_0[i] = y[length(y)]
end

b = reshape(b_0, 100, 100)'
alphas = 0.1 .+ exp.(0.05 .* LinRange(-1, 1, 100))
gammas = 0.001 .+ 0.01 .* exp.(0.05 .* LinRange(-1, 1, 100))
heatmap(alphas, gammas, b, colormap=:turbo, size=(460, 400),
        title="Surface Reaction at t=$tmax", xlabel="adsorption, \$\\alpha\$", ylabel="desorption, \$\\gamma\$")
savefig("qoi_surface.png")