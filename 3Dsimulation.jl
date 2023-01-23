# 3Dsimulation.jl

include("./data.jl")
include("./sampling.jl")
using Plots

nPerDim = 24
ndim = 3
dpoly = 5
n = nPerDim ^ ndim

initMethod = "grid"
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, initMethod, "Legendre", "spring")
b_0 = data.generateSpringDistance3D(A[:, 2 : 4])

function turbocolormap(x)
    x = clamp.(x, 0.0, 1.0)
    c1 = 0.13572137988692035 .+ x.*(4.597363719627905  .+ x.*(-42.327689751912274 .+ x.*( 130.58871182451415 .+ x.*(-150.56663492057857 .+ x.*58.137453451135656))))
    c2 = 0.09140261235958302 .+ x.*(2.1856173378635675 .+ x.*( 4.805204796477784  .+ x.*(-14.019450960349728 .+ x.*(  4.210856355081685 .+ x.*2.7747311504638876))))
    c3 = 0.10667330048674728 .+ x.*(12.592563476453211 .+ x.*(-60.10967551582361  .+ x.*( 109.07449945380961 .+ x.*( -88.50658250648611 .+ x.*26.818260967511673))))
    return [clamp.(c1, 0.0, 1.0) clamp.(c2, 0.0, 1.0) clamp.(c3, 0.0, 1.0)]
end

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

p = fill(plot(), 25, 1)
l = @layout [a{0.01h}; grid(4, 6)]
p[1] = plot(title="Spring Distance 3D", showaxis=false, grid=false)
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
