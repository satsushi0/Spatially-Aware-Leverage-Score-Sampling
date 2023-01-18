# data.jl
# Generate dataset for multi-dimensional polynomial regression.

module data

    using Combinatorics
    using LinearAlgebra
    using ODE
    using DiffEqOperators
    using DifferentialEquations
    
    function generate(ndim, nPerDim, dpoly, polyType="Legendre", target="spring")

        A = _generateA(ndim, nPerDim, dpoly, polyType)
        tau = _leverageScore(A)
        if target == "spring"
            b_0 = _generateSpringDistance(nPerDim)
        elseif target == "heat"
            b_0 = _generateHeatEquation(nPerDim) 
        end
        return A, tau, b_0

    end

    function _generateA(ndim, nPerDim, dpoly, polyType)

        n = nPerDim ^ ndim
        d = binomial(dpoly + ndim, ndim)
        
        base = zeros(ndim, nPerDim)
        for dim in 1 : ndim
            base[dim, :] = LinRange(-1, 1, nPerDim)
        end
        
        polyStraight = ones(ndim, nPerDim, dpoly + 1)
        polyStraight[:, :, 2] = base
        for k in 3 : dpoly + 1
            if polyType == "Legendre"
                polyStraight[:, :, k] = (2 * k - 1) / k * polyStraight[:, :, k - 1] .* base - (k - 1) / k * polyStraight[:, :, k - 2]
            elseif polyType == "Chebyshev"
                polyStraight[:, :, k] = 2 * polyStraight[:, :, k - 1] .* base - polyStraight[:, :, k - 2]
            else
                polyStraight[:, :, k] = polyStraight[:, :, k - 1] .* base
            end
        end
        
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

        C = collect(combinations(1 : dpoly + ndim, ndim))
        C = reduce(vcat, C')
        for i in ndim : -1 : 2
            C[:, i] = C[:, i] - C[:, i - 1]
        end
        order = sortperm(sum(C, dims=2)[:, 1])
        C = C[order, :]

        A = ones(n, d)
        for i in 1 : n
            for j in 1 : d
                for k in 1 : ndim
                    A[i, j] = A[i, j] * polyStraight[k, P[i, k], C[j, k]]
                end
            end
        end

        return A
        
    end

    function _leverageScore(A)
        U, _, _ = svd(A)
        tau = sum(U .^ 2, dims=2)
        return tau
    end

    function _generateSpringDistance(nPerDim)
        
        c = 0.5
        k = 2.0
        f = 0.5
        w = 0.8
        p = [c, k, f, w]
        yinit = [0.5, 0.0]
        tmax = 20.0
        kvals = collect(LinRange(1, 3, nPerDim))
        wvals = collect(LinRange(0, 2, nPerDim))

        function odemodel(t, y)
            deriv = [y[2], p[3] * cos(p[4] * t) - p[2] * y[1] - p[1] * y[2]]
            return deriv
        end

        b_0 = zeros(nPerDim, nPerDim)
        for i in 1 : nPerDim
            for j in 1 : nPerDim
                p[4] = wvals[i]
                p[2] = kvals[j]
                _, y = ode45(odemodel, yinit, [0.0, tmax])
                y = reduce(vcat, y')
                b_0[i, j] = maximum(y[:, 1])
            end
        end

        return b_0

    end

    function _generateHeatEquation(nPerDim)

        diffusion_coef = 1.0 / pi ^ 2
        fvals = LinRange(0, 5, nPerDim)
        x = range(0, 1, length=100)
        time = LinRange(0, 1.0, nPerDim)
        
        function f!(du, u, p, t)
            Q, D, diffusion_coef = p
            du .= diffusion_coef * D * Q * u
        end

        b_0 = zeros(nPerDim, nPerDim)
        for i in eachindex(fvals)
            freq = fvals[i]
            u0 = sin.(freq * pi * x) .+ 1.0
            Q = Dirichlet0BC(eltype(u0))
            D = CenteredDifference{1}(2, 3, Float64(x.step), x.len)
            p = [Q, D, diffusion_coef]
            tspan = (0.0, 1.0)
            prob = ODEProblem(f!, u0, tspan, p)
            sol = solve(prob)
            u = reduce(vcat, sol(time).u')
            b_0[i, :] = maximum(u, dims=2)
        end
        b_0 = b_0 .- minimum(b_0)

        return b_0

    end

end