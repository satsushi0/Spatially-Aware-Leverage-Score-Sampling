# QR1d.jl

using Plots
using LinearAlgebra

n = 100
dpoly = 30
x = collect(LinRange(-1.0, 1.0, n))
A = ones(Float64, n)

for i in 1 : dpoly
    A = [A A[:, size(A, 2)] .* x]
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