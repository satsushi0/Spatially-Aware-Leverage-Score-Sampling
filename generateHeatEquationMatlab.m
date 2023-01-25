% Heat Equation target calculation.
% Analogous to _generateHeatEquation in data.jl except that this uses a PDE solver.

function b_0 = generateHeatEquationMatlab(A_1)
    
    n = size(A_1, 1);

    b_0 = zeros(n, 1);
    
    times = (A_1(:, 1) + 1.0) * 1.5;
    freqs = (A_1(:, 2) + 1.0) * 2.5;
    
    parfor i = 1 : n
        freq = freqs(i);
        time = times(i);
        b_0(i) = pdesolver(freq, time);
    end
    
end

function b = pdesolver(freq, time)
    x = linspace(0, 1, 100);
    if time <= 0.0
        t = [0.0, 0.1, 0.2];
    else
        t = [0.0, time, time + 0.1];
    end
    u0 = @(x) sin(freq * pi * x) + 1.0;
    u = pdepe(0.0, 'pdex1pde', u0, 'pdex1bc', x, t);
    if time <= 0.0
        b = max(u(1, :));
    else
        b = max(u(2, :));
    end
end