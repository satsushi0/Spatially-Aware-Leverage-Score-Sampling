# initDist.jl

include("./data.jl")
include("./sampling.jl")
using Plots
using VegaLite, DataFrames, FileIO

nPerDim = 50
ndim = 2
dpoly = 5
n = nPerDim ^ ndim

initMethods = ["grid", "uniform", "Gaussian", "ChebyshevNodes"]

for initMethod in initMethods
    A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, initMethod, "Legendre", "spring")
    df = DataFrame(x=A[:, 2], y=A[:, 3], tau=tau[:, 1])
    df |> @vlplot(mark={type=:point, filled=true, size=50, opacity=1.0}, x=:x, y=:y, color={:tau, scale={scheme=:turbo, domain=[0.0, 0.15]}}, 
                width=400, height=400, title="$initMethod, n=$n") |> FileIO.save("$initMethod.png")

    df = DataFrame(x=A[:, 2], y=A[:, 3], qoi=b_0)
    df |> @vlplot(mark={type=:point, filled=true, size=50, opacity=1.0}, x=:x, y=:y, color={:qoi, scale={scheme=:turbo, domain=[0.5, 1.0]}}, 
                width=400, height=400, title="$initMethod, n=$n") |> FileIO.save("qoi_spring_$initMethod.png")
end
