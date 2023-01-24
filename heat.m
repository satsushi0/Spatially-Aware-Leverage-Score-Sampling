% Heat Equation target calculation.
% Analogous to _generateHeatEquation in data.jl except that this uses a PDE solver.

function b_0 = generateHeatEquationMatlab(A_1)
    
    n = size(A_1, 1);
    x = linspace(0, 1, 100);

    b_0 = zeros(n, 1);
    for i = 1 : n
        freq = A_1(i, 2);
        time = [A_1(i, 1) * 0.9, A_1(i, 1), A_1(i, 1) * 1.1];
        u0 = @(x) sin(freq * pi * x) + 1.0;
        u = pdepe(0.0, 'pdex1pde', u0, 'pdex1bc', x, time);
        b_0(i) = max(u(2, :));
    end
    
end