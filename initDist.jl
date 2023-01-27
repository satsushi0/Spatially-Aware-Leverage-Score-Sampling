# initDist.jl
# Visualize the initial data points.

include("./data.jl")
include("./sampling.jl")
using Plots
using VegaLite, DataFrames, FileIO

nPerDim = 50                        # Number of points to generate for each coordinate.
ndim = 2                            # Dimensionality of the target function.
dpoly = 5                           # Polynomial degree for the regression.
n = nPerDim ^ ndim                  # Number of total data points.

initMethods = ["grid", "uniform", "Gaussian", "ChebyshevNodes"]

for initMethod in initMethods
    # Generate the initial data points.
    A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, initMethod, "Legendre", "spring")
    # Plot the data points colored by the leverage score.
    df = DataFrame(x=A[:, 2], y=A[:, 3], tau=tau[:, 1])
    df |> @vlplot(mark={type=:point, filled=true, size=50, opacity=1.0}, x=:x, y=:y, color={:tau, scale={scheme=:turbo, domain=[0.0, 0.15]}}, 
                width=400, height=400, title="$initMethod, n=$n") |> FileIO.save("$initMethod.png")
    # Plot the data points colored by the target value.
    df = DataFrame(x=A[:, 2], y=A[:, 3], qoi=b_0)
    df |> @vlplot(mark={type=:point, filled=true, size=50, opacity=1.0}, x=:x, y=:y, color={:qoi, scale={scheme=:turbo, domain=[0.5, 1.0]}}, 
                width=400, height=400, title="$initMethod, n=$n") |> FileIO.save("qoi_spring_$initMethod.png")
end

initMethod = "Gaussian_nomanip"
A, tau, b_0 = data.generate(ndim, nPerDim, dpoly, initMethod, "Legendre", "heat_matlab")

# Plot the data points colored by the leverage score.
_tau = clamp.(tau, 0.0, 0.15)
df = DataFrame(x=A[:, 2], y=A[:, 3], _tau=_tau[:, 1])
df |> @vlplot(mark={type=:point, filled=true, size=50, opacity=1.0}, x=:x, y=:y, color={:_tau, scale={scheme=:turbo, domain=[0.0, 0.15]}}, 
            width=400, height=400, title="$initMethod, n=$n, ") |> FileIO.save("$initMethod" * "_updated.png")
# Plot the data points colored by the target value.
df = DataFrame(x=A[:, 2], y=A[:, 3], qoi=b_0)
df |> @vlplot(mark={type=:point, filled=true, size=50, opacity=1.0}, x=:x, y=:y, color={:qoi, scale={scheme=:turbo, domain=[0.5, 2.0]}}, 
            width=400, height=400, title="$initMethod, n=$n") 
            |> FileIO.save("qoi_heat_$initMethod" * ".png")

histogram(b_0, bins=range(0, 2, length=21), normalize=:pdf, xlabel="target value (y)", ylabel="Pr(y)", title="PDF of Heat Equation | Gaussian initialization", legend=false)
savefig("histogram_heat_gaussian.png")

heatmap(LinRange(0.0, 3.0, nPerDim), LinRange(0.0, 5.0, nPerDim), reshape(b_0, nPerDim, nPerDim)', 
        colormap=:turbo, size=(450, 400),
        title="Heat Equation in Matlab", xlabel="time, \$t\$", ylabel="starting frequency, \$f\$")
savefig("qoi_heat_matlab.png")