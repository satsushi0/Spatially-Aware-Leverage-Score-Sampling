# sampling.jl

module sampling
    
    function bernoulliSampling(_prob, k)
        prob = _prob ./ sum(_prob) .* k
        sample = Int64[]
        random = rand(length(prob))
        for i in 1 : length(prob)
            if random[i] < prob[i]
                push!(sample, i)
            end
        end
        return sample, prob[sample]
    end

    function createBinaryTree(A_1, _prob, k)
        
        n, ndim = size(A_1)
        A_1 = [A_1 collect(1 : 1 : n)]
        prob = _prob ./ sum(_prob) .* k
        expo = ceil(Int64, log2(n))
        binaryTree = zeros(2 ^ (expo + 1) - 1, 2)

        function fillUp(mat, loc, coord)
            _n = size(mat, 1)
            _expo = ceil(Int64, log2(_n))
            if _n == 1
                index = trunc(Int64, mat[ndim + 1])
                binaryTree[loc, :] = [index, prob[index]]
                return
            end
            mat = mat[sortperm(mat[:, coord]), :]
            coord = coord % ndim + 1
            fillUp(mat[1 : div(_n, 2), :], loc, coord)
            fillUp(mat[div(_n, 2) + 1 : _n, :], loc + 2 ^ (_expo - 1), coord)
        end

        fillUp(A_1, 2^expo, 1)
        return binaryTree

    end

    function btPivotalSampling(binaryTree, _prob, k)

        expo = ceil(Int64, log2(size(binaryTree, 1))) - 1
        prob = _prob ./ sum(_prob) .* k
        sample = Int64[]
        for e in expo : -1 : 1
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

end