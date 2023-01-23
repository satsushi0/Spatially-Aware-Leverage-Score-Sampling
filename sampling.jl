# sampling.jl

module sampling

    using MultivariateStats
    using LinearAlgebra
    
    function bernoulliSampling(_prob, k)
        prob = _prob ./ sum(_prob) .* k
        sample = Int64[]
        random = rand(length(prob))
        for i in eachindex(prob)
            if random[i] < prob[i]
                push!(sample, i)
            end
        end
        return sample, prob[sample]
    end

    function createBinaryTree(A_1, _prob, k, method="coordwise", pcaFirstRotation=45.0)
        
        n, ndim = size(A_1)
        A_1 = [A_1 collect(1 : 1 : n)]
        prob = _prob ./ sum(_prob) .* k
        exponent = ceil(Int64, log2(n))
        binaryTree = zeros(2 ^ (exponent + 1) - 1, 2)

        function fillUp(mat, loc, coord=0)
            _n = size(mat, 1)
            _exponent = ceil(Int64, log2(_n))
            if _n == 1
                index = trunc(Int64, mat[ndim + 1])
                binaryTree[loc, :] = [index, prob[index]]
                return
            end
            if method == "coordwise"
                mat = mat[sortperm(mat[:, coord]), :]
                coord = coord % ndim + 1
            elseif method == "PCA"
                pca = fit(PCA, mat[:, 1 : ndim]'; maxoutdim=2)
                _mat = predict(pca, mat[:, 1 : ndim]')'
                # mat = mat[sortperm(_mat[:, size(pca)[2]]), :]
                mat = mat[sortperm(_mat[:, 1]), :]
            end
            fillUp(mat[1 : div(_n, 2), :], loc, coord)
            fillUp(mat[div(_n, 2) + 1 : _n, :], loc + 2 ^ (_exponent - 1), coord)
        end

        if method == "coordwise"
            fillUp(A_1, 2^exponent, 1)
        elseif method == "PCA"
            if pcaFirstRotation == 0.0
                fillUp(A_1, 2^exponent, 1)
            else
                deg = pi * pcaFirstRotation / 180.0
                rotate = Matrix{Float64}(I, ndim, ndim)
                rotate[ndim - 1 : ndim, ndim - 1 : ndim] = [cos(deg) -sin(deg); sin(deg) cos(deg)]
                mat = [A_1[:, 1 : ndim] * rotate A_1[:, ndim + 1]]
                mat = mat[sortperm(mat[:, 2]), :]
                fillUp(mat[1 : div(n, 2), :], 2^exponent, 0)
                fillUp(mat[div(n, 2) + 1 : n, :], 2^exponent + 2 ^ (exponent - 1), 0)
            end
        end

        return binaryTree

    end

    function btPivotalSampling(binaryTree, _prob, k)

        exponent = ceil(Int64, log2(size(binaryTree, 1))) - 1
        prob = _prob ./ sum(_prob) .* k
        sample = Int64[]
        for e in exponent : -1 : 1
            for i in 2 ^ (e - 1) : 2 ^ e - 1
                l, r = 2 * i, 2 * i + 1
                pl, pr = binaryTree[l, 2], binaryTree[r, 2]
                if pl + pr <= 1.0
                    if rand() < pl / (pl + pr)
                        binaryTree[i, :] = [binaryTree[l, 1], pl + pr]
                    else
                        binaryTree[i, :] = [binaryTree[r, 1], pl + pr]
                    end
                else
                    if rand() < (1 - pl) / (2 - pl - pr)
                        binaryTree[i, :] = [binaryTree[l, 1], pl + pr - 1.0]
                        push!(sample, trunc(Int64, binaryTree[r, 1]))
                    else
                        binaryTree[i, :] = [binaryTree[r, 1], pl + pr - 1.0]
                        push!(sample, trunc(Int64, binaryTree[l, 1]))
                    end    
                end
            end
        end
        if size(sample, 1) < k
            push!(sample, trunc(Int64, binaryTree[1, 1]))
        end
        return sample, prob[sample]

    end

    function distance(A_1)
        n, ndim = size(A_1)
        dist = zeros(Float64, n, n)
        dist_order = zeros(Int64, n, n)
        pos_exp, neg_exp = exp.(A_1), exp.(-A_1)
        for i in 1 : ndim
            dist = dist + log.(pos_exp[:, i] * neg_exp[:, i]') .^ 2
        end
        for i in 1 : n
            dist_order[i, :] = sortperm(dist[i, :])
        end
        return dist_order
    end

    function distPivotalSampling(dist_order, _prob, k)

        n = size(dist_order, 1)
        prob = _prob ./ sum(_prob) .* k
        sample = Int64[]
        indexProb = [1 : n prob]
        randomOrder = sortperm(rand(n))
        loc = collect(1 : n)
        for i in randomOrder[1 : n - 1]
            ix_i, p_i = Int(indexProb[i, 1]), indexProb[i, 2]
            j, ix_j, p_j = 0, 0, 0.0
            for k = 2 : n
                ix_j = dist_order[ix_i, k]
                if loc[ix_j] != 0
                    j = loc[ix_j]
                    p_j = indexProb[j, 2]
                    break
                end
            end

            if p_i + p_j < 1.0
                if rand() * (p_i + p_j) < p_j
                    indexProb[j, :] = [ix_j, p_i + p_j]
                    loc[ix_i] = 0
                    loc[ix_j] = j
                else
                    indexProb[j, :] = [ix_i, p_i + p_j]
                    loc[ix_i] = j
                    loc[ix_j] = 0
                end
            else
                if rand() * (2 - p_i - p_j) < 1 - p_j
                    indexProb[j, :] = [ix_j, p_i + p_j - 1.0]
                    loc[ix_i] = 0
                    push!(sample, ix_i)
                    loc[ix_j] = j
                else
                    indexProb[j, :] = [ix_i, p_i + p_j - 1.0]
                    loc[ix_i] = j
                    loc[ix_j] = 0
                    push!(sample, ix_j)
                end
            end
        end
        if size(sample, 1) < k
            push!(sample, Int(indexProb[randomOrder[n], 1]))
        end
        return sample, prob[sample]

    end

end