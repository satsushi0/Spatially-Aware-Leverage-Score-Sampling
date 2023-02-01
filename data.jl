# data.jl
# Generate dataset for multi-dimensional polynomial regression.

module data

    using Combinatorics
    using LinearAlgebra
    using ODE
    using DiffEqOperators
    using DifferentialEquations
    using MATLAB
    using Base.Threads
    
    function generate(ndim, nPerDim, dpoly, AType="grid", polyType="Legendre", target="spring")
        # ===== Parameters =====
        # ndim      Dimensionality of the target function.
        # nPerDim   Number of points to generate for each coordinate.
        # dpoly     Polynomial degree for the regression.
        # AType     How to generate initial data points for the matrix A. 
        #               grid:               [-1, 1]^ndim grid.
        #               Gaussian_nomanip    Drawn from the Gaussian distribution without any manipulation.
        #               Gaussian_range:     Drawn from the Gaussian distribution of mean 0, std 0.75. Values are in range [-1, 1]
        #               Gaussian:           Gaussian of mean 0 and std 0.75 but not limited to range [-1, 1].
        #                                   Truncated only to make sure the ODE & PDE work.
        #               uniform:            Drawn from the uniform distribution.
        #               ChebyshevNodes:     Drawn uniformly at random from the Chebyshev Nodes of 10000 values.
        # polyType  Type of polynomials. [Chebyshev, Legendre, QR, None]
        # target    Target function. [spring, heat]
        #               spring:             Spring distance for 2D or 3D space.
        #               heat:               Heat equation for 2D only.
        #               heat_matlab:        Heat equation solved by PDE solver in Matlab. 2D only.
        #
        # ===== Outputs =====
        # A     Data points of n by d matrix. 
        #       The columns are sorted in the order of polynomial degree. The lower degree comes to the left, the higher is settled to the right.
        #       The first column is the bias term (polynomial degree is 0).
        #       The second column corresponds to the term with polynomial degree 1.
        # tau   Leverage score of matrix A (not normalized).
        # b_0   Target function values. n by 1 vector.

        A = _generateA(ndim, nPerDim, dpoly, AType, polyType)
        tau = _leverageScore(A)
        if ndim == 2
            if target == "spring"
                b_0 = _generateSpringDistance(A[:, 2 : 3])
            elseif target == "heat"
                b_0 = _generateHeatEquation(A[:, 2 : 3]) 
            elseif target == "heat_matlab"
                b_0 = _generateHeatEquationMatlab(A[:, 2 : 3])
            end
        elseif ndim == 3
            if target == "spring"
                b_0 = _generateSpringDistance3D(A[:, 2 : 4])
            end
        end
        return A, tau, b_0

    end

    function _generateA(ndim, nPerDim, dpoly, AType, polyType)

        n = nPerDim ^ ndim
        d = binomial(dpoly + ndim, ndim)    # Number of features.
        base = zeros(ndim, n)               # The matrix of only polynomial degree 1 terms.

        # Fill up the matrix base.
        if AType == "grid"
            P = zeros(Int64, n, ndim)       # A matrix holding all the permutation tuple (i_1, ..., i_ndim) where i in [n].
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
        elseif AType == "Gaussian_nomanip"
            std = 0.75
            for i in 1 : n
                r = randn(ndim) * std
                base[:, i] = r
            end
        elseif AType == "Gaussian_range"
            std = 0.75
            for i in 1 : n
                r = randn(ndim) * std
                while sum(abs.(r) .> 1.0) > 0
                    r[abs.(r) .> 1.0] .= randn(sum(abs.(r) .> 1.0))
                end
                base[:, i] = r
            end
        elseif AType == "Gaussian"
            # All values should be bigger than -1 to make the ODE & PDE valid.
            std = 0.75
            for i in 1 : n
                r = randn(ndim) * std
                while sum(r .< -1.0) > 0
                    r[r .< -1.0] .= randn(sum(r .< -1.0))
                end
                base[:, i] = r
            end
        elseif AType == "uniform"
            for i in 1 : n
                base[:, i] = [rand() * 2.0 - 1.0, rand() * 2.0 - 1.0]
            end
        elseif AType == "ChebyshevNodes"
            num = 10000
            ChebyshevNodes = cos.((2.0 .* collect(1 : num) .- 1.0) ./ (2.0 * num) .* pi)
            for i in 1 : ndim
                base[i, :] = ChebyshevNodes[rand(1 : num, n)]
            end
        end

        # Compute the polynomial values.
        polyStraight = ones(Float64, ndim, n, dpoly + 1)    # The third dimension of length dpoly+1 corresponds to the polynomial degree.
        # As the polyStraight is initialized as all ones, we don't need to do additional for the bias terms polyStraight[:, :, 1].
        polyStraight[:, :, 2] = base    # Insert the base as the polynomial degree 1 terms.
        # Compute the values of polynomial degree 2 or higher.
        for k in 3 : dpoly + 1
            if polyType == "Legendre"
                polyStraight[:, :, k] = (2 * k - 1) / k * polyStraight[:, :, k - 1] .* base - (k - 1) / k * polyStraight[:, :, k - 2]
            elseif polyType == "Chebyshev"
                polyStraight[:, :, k] = 2 * polyStraight[:, :, k - 1] .* base - polyStraight[:, :, k - 2]
            else
                polyStraight[:, :, k] = polyStraight[:, :, k - 1] .* base
            end
        end

        # Construct a d by ndim matrix for final output A. 
        # Each row indicates how to mix up the polynomials in polyStraight.
        # E.g. (3, 4, 5) means x_1^2 * x_2^3 * x_3^4, pointing the location in polyStraight.
        C = collect(combinations(1 : dpoly + ndim, ndim))
        C = reduce(vcat, C')
        for i in ndim : -1 : 2
            C[:, i] = C[:, i] .- C[:, i - 1]
        end
        order = sortperm(sum(C, dims=2)[:, 1])
        C = C[order, :]

        A = ones(Float64, n, d)
        if polyType == "QR"
            A = ones(Float64, n)
            C = C .- 1      # Now, C is not pointing a location in polyStraight but polynomial degree.

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
        else
            for j in 1 : d
                for k in 1 : ndim
                    A[:, j] = A[:, j] .* polyStraight[k, :, C[j, k]]
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


    # Spring Distance Target.
    # Damped harmonic oscillator with periodic driving force.
    # Example taken from https://www5.in.tum.de/lehre/vorlesungen/algo_uq/ss18/06_polynomial_chaos.pdf

    function _generateSpringDistance(A_1)
        # A_1   A matrix of size n by ndim, which corresponds to the entries of polynomial degree 1 in A.
        
        n = size(A_1, 1)

        c = 0.5             # Damping coefficient
        k = 2.0             # Spring constant
        f = 0.5             # Forcing amplitude
        w = 0.8             # Frequency
        p = [c, k, f, w]
        yinit = [0.5, 0.0]  # Initial position
        tmax = 20.0

        function odemodel(t, y)
            deriv = [y[2], p[3] * cos(p[4] * t) - p[2] * y[1] - p[1] * y[2]]
            return deriv
        end

        b_0 = zeros(Float64, n)
        for i in 1 : n
            p[2] = A_1[i, 1] + 2.0
            p[4] = A_1[i, 2] + 1.0
            _, y = ode45(odemodel, yinit, [0.0, tmax])
            y = reduce(vcat, y')
            b_0[i] = maximum(y[:, 1])
        end

        return b_0

    end

    function _generateSpringDistance3D(A_1)

        n = size(A_1, 1)

        c = 0.5
        k = 2.0
        f = 0.5
        w = 0.8
        p = [c, k, f, w]
        yinit = [0.5, 0.0]
        tmax = 20.0

        function odemodel(t, y)
            deriv = [y[2], p[3] * cos(p[4] * t) - p[2] * y[1] - p[1] * y[2]]
            return deriv
        end

        b_0 = zeros(Float64, n)
        for i in 1 : n
            p[2] = A_1[i, 1] + 2.0
            p[3] = A_1[i, 2] + 1.0
            p[4] = A_1[i, 3] + 1.0
            _, y = ode45(odemodel, yinit, [0.0, tmax])
            y = reduce(vcat, y')
            b_0[i] = maximum(y[:, 1])
        end

        return b_0

    end

    # Heat Equation Target.
    # Example taken from http://courses.washington.edu/amath581/PDEtool.pdf
    # As there is no PDE solver in Julia, I used the approximation method in https://discourse.julialang.org/t/solving-heat-diffusion-pde-using-diffeqtools-jl-and-differentialequations-jl/41630

    function _generateHeatEquation(A_1)

        n = size(A_1, 1)

        diffusion_coef = 1.0 / pi ^ 2
        x = range(0, 1, length=50)
        
        function f!(du, u, p, t)
            Q, D, diffusion_coef = p
            du .= diffusion_coef * D * Q * u
        end

        b_0 = zeros(Float64, n)
        for i in 1 : n
            freq = (A_1[i, 2] + 1.0) * 2.5
            time = (A_1[i, 1] + 1.0) * 1.5
            u0 = sin.(freq * pi * x) .+ 1.0
            Q = Dirichlet0BC(eltype(u0))
            D = CenteredDifference{1}(2, 3, Float64(x.step), x.len)
            p = [Q, D, diffusion_coef]
            tspan = (0.0, 3.0)
            prob = ODEProblem(f!, u0, tspan, p)
            sol = solve(prob)
            u = sol(time)
            b_0[i] = maximum(u)
        end
        b_0 = b_0 .- minimum(b_0)

        return b_0

    end

    # function _generateHeatEquation(nPerDim)
        # This implementation is only for grid.

    #     diffusion_coef = 1.0 / pi ^ 2
    #     fvals = LinRange(0, 5, nPerDim)
    #     x = range(0, 1, length=100)
    #     time = LinRange(0, 3.0, nPerDim)
        
    #     function f!(du, u, p, t)
    #         Q, D, diffusion_coef = p
    #         du .= diffusion_coef * D * Q * u
    #     end

    #     b_0 = zeros(nPerDim, nPerDim)
    #     for i in eachindex(fvals)
    #         freq = fvals[i]
    #         u0 = sin.(freq * pi * x) .+ 1.0
    #         Q = Dirichlet0BC(eltype(u0))
    #         D = CenteredDifference{1}(2, 3, Float64(x.step), x.len)
    #         p = [Q, D, diffusion_coef]
    #         tspan = (0.0, 3.0)
    #         prob = ODEProblem(f!, u0, tspan, p)
    #         sol = solve(prob)
    #         u = reduce(vcat, sol(time).u')
    #         b_0[i, :] = maximum(u, dims=2)
    #     end
    #     b_0 = b_0 .- minimum(b_0)

    #     return b_0

    # end

    function _generateHeatEquationMatlab(A_1)

        dir = @__DIR__
        mat"addpath($dir)"

        b_0 = mxcall(:generateHeatEquationMatlab, 1, A_1)

        return b_0
    end

end