# sampling.jl
# Functions for sampling.
# Bernoulli sampling                    For each data point, decide to select or not from its inclusion probability.
#                                       The samples are independent. The number of sample is not always k but k in expectation.
# Pivotal sampling with binary tree     Parcolate up the tree with pivotal match. The samples are negatively associated.
# Pivotal sampling by distance          Do the pivotal match on a pair of close points. We don't know about the resulting distribution.
#                                       https://www.jstor.org/stable/23270453#metadata_info_tab_contents
#
# ===== Common Parameters =====
# _prob     The inclusion probability. This will be scaled to be summed up to k.
# k         The number of samples.
# A_1       A matrix of size n by ndim, which corresponds to the entries of polynomial degree 1 in A.
#
# ===== Common Outputs =====
# sample        Selected indices in A.
# prob[sample]  The inclusion probability of each sample. This will be used to rescale the sampled rows in A.

module sampling

    using MultivariateStats
    using LinearAlgebra
    using Random
    
    function bernoulliSampling(_prob, k)
        # Scale the inclusion probability.
        prob = _prob ./ sum(_prob) .* k
        # If we have probabilities bigger than 1.0, we pick those deterministically.
        # Then, adjust the other probabilities so that we get exactly k samples (in expectation in Bernoulli sampling).
        while abs(sum(clamp.(prob, 0.0, 1.0)) - k) > 1e-5
            rescaleFactor = (k - sum(prob .>= 1.0)) / sum(prob[prob .< 1.0])
            prob = prob * rescaleFactor
        end
        sample = Int64[]
        random = rand(length(prob))
        for i in eachindex(prob)
            if random[i] < prob[i]
                push!(sample, i)
            end
        end
        return sample, clamp.(prob, 0.0, 1.0)[sample]
    end

    function createBinaryTree(A_1, _prob, k, method="coordwise", pcaFirstRotation=45.0)
        # Construct a binary tree for the pivotal sampling with binary tree.
        
        n, ndim = size(A_1)             # n: number of samples, ndim: the dimensionality.
        A_1 = [A_1 collect(1 : 1 : n)]  # As A_1 will be shuffled, add collect(1 : 1 : n) to keep track of the initial position.
        prob = _prob ./ sum(_prob) .* k
        # If we have probabilities bigger than 1.0, we pick those deterministically.
        # Then, adjust the other probabilities so that we get exactly k samples (in expectation in Bernoulli sampling).
        while abs(sum(clamp.(prob, 0.0, 1.0)) - k) > 1e-5
            rescaleFactor = (k - sum(prob .>= 1.0)) / sum(prob[prob .< 1.0])
            prob = prob * rescaleFactor
        end
        exponent = ceil(Int64, log2(n)) # The height of the binary tree.

        binaryTree = zeros(Float64, 2 ^ (exponent + 1) - 1, 2)
        # Each row has two values: index and probability.
        # The first row corresponds to the root. Its left child is in row 2 and right child is in row 3.
        # All the points in A_1 are stored only in the leaves.

        function fillUp(mat, loc, coord=0)
            # mat       A fraction of rows in A.
            # loc       The leftmost leaf for mat.
            # coord     In coordwise partitioning, the value in coord will be used. 
            _n = size(mat, 1)                   # The number of data points in mat.
            _exponent = ceil(Int64, log2(_n))   # The height of the subtree.
            if _n == 1  # Base case. Place mat at loc.
                index = trunc(Int64, mat[ndim + 1])
                binaryTree[loc, :] = [index, prob[index]]
                return
            end
            if method == "coordwise"
                mat = mat[sortperm(mat[:, coord]), :]   # Sort the mat by values in coord.
                coord = coord % ndim + 1                # Increment coord by 1.
            elseif method == "PCA"
                # Sort the mat in the direction of maximum variance.
                pca = fit(PCA, mat[:, 1 : ndim]'; maxoutdim=2)
                _mat = predict(pca, mat[:, 1 : ndim]')'
                # mat = mat[sortperm(_mat[:, size(pca)[2]]), :]
                mat = mat[sortperm(_mat[:, 1]), :]
            end
            # Divide the mat into two subgroups and call the function itself.
            fillUp(mat[1 : div(_n, 2), :], loc, coord)
            fillUp(mat[div(_n, 2) + 1 : _n, :], loc + 2 ^ (_exponent - 1), coord)
        end

        if method == "coordwise"
            fillUp(A_1, 2^exponent, 1)
        elseif method == "PCA"
            if pcaFirstRotation == 0.0
                fillUp(A_1, 2^exponent, 1)
            else
                # When the initial data points have the form of grid, the PCA might produce a partition similar to the one by coordwise especially in the early steps.
                # To address this issue, the first cut can be done deterministically by setting the pcaFirstRotation.
                # It rotates the input matrix and divide it by values in the second coordinate.
                deg = pi * pcaFirstRotation / 180.0
                rotate = Matrix{Float64}(I, ndim, ndim)     # The matrix for the rotation.
                rotate[ndim - 1 : ndim, ndim - 1 : ndim] = [cos(deg) -sin(deg); sin(deg) cos(deg)]
                mat = [A_1[:, 1 : ndim] * rotate A_1[:, ndim + 1]]
                mat = mat[sortperm(mat[:, 2]), :]           # Sort by the values in the second coordinate.
                fillUp(mat[1 : div(n, 2), :], 2^exponent, 0)
                fillUp(mat[div(n, 2) + 1 : n, :], 2^exponent + 2 ^ (exponent - 1), 0)
            end
        end

        return binaryTree

    end

    function btPivotalSampling(binaryTree, _prob, k)

        exponent = ceil(Int64, log2(size(binaryTree, 1))) - 1
        prob = _prob ./ sum(_prob) .* k
        while abs(sum(clamp.(prob, 0.0, 1.0)) - k) > 1e-5
            rescaleFactor = (k - sum(prob .>= 1.0)) / sum(prob[prob .< 1.0])
            prob = prob * rescaleFactor
        end
        sample = Int64[]

        # Loop over all the leaves. 
        # If the probability is bigger than 1.0, pick it as a sample and set the probability to -2.0.
        # Note that binaryTree is passed to this function by reference. 
        # To reuse the same tree multiple times, need to set the probabilities of deterministically chosen points smaller than -1.0.
        for i in 2 ^ exponent : 2 ^ (exponent + 1) - 1
            if abs(binaryTree[i, 2]) >= 1.0
                push!(sample, trunc(Int64, binaryTree[i, 1]))
                binaryTree[i, 2] = -2.0
            end
        end

        for e in exponent : -1 : 1                              # From the second bottom layer to top layer.
            for i in 2 ^ (e - 1) : 2 ^ e - 1                    # From the left node to the right node.
                l, r = 2 * i, 2 * i + 1                         # The indices of the children.
                pl, pr = binaryTree[l, 2], binaryTree[r, 2]     # The inclusion probability.
                if pl <= 0.0
                    binaryTree[i, :] = [binaryTree[r, 1], pr]
                elseif pr <= 0.0
                    binaryTree[i, :] = [binaryTree[l, 1], pl]
                elseif pl + pr < 1.0
                    if rand() < pl / (pl + pr)                  # The case l promotes.
                        binaryTree[i, :] = [binaryTree[l, 1], pl + pr]
                    else                                        # The case r promotes
                        binaryTree[i, :] = [binaryTree[r, 1], pl + pr]
                    end
                else
                    if rand() < (1 - pl) / (2 - pl - pr)        # The case l promotes.
                        binaryTree[i, :] = [binaryTree[l, 1], pl + pr - 1.0]
                        push!(sample, trunc(Int64, binaryTree[r, 1]))
                    else                                        # The case r promotes.
                        binaryTree[i, :] = [binaryTree[r, 1], pl + pr - 1.0]
                        push!(sample, trunc(Int64, binaryTree[l, 1]))
                    end    
                end
            end
        end
        # Deal with the floating point issue.
        # In the last step where we have p_i + p_j = 1.0, it may be categorized as p_i + p_j < 1.0.
        # In this case, the sample that should have been chosen is in the root node.
        if size(sample, 1) < k
            push!(sample, trunc(Int64, binaryTree[1, 1]))
        end
        return sample, clamp.(prob, 0.0, 1.0)[sample]

    end

    function distance(A_1)
        # Compute the l2 distance for all the pairs of points in A_1.
        # Returns dist_order, a matrix of size n by n.
        # The first entry in i-th row is the index of the closest point to i (which is i itself). 
        # The rightmost entry is the index of the farthest point to i.
        n, ndim = size(A_1)
        dist = zeros(Float64, n, n)     # A matrix for l2 distance.
        dist_order = zeros(Int64, n, n)
        # When ndim is small, using the outer product seems to be faster.
        pos_exp, neg_exp = exp.(A_1), exp.(-A_1)
        for i in 1 : ndim
            dist = dist + log.(pos_exp[:, i] * neg_exp[:, i]') .^ 2
        end
        # When ndim is large, the naive way might be faster.
        # for i in 1 : n - 1
        #     for j in i + 1 : n
        #         dist[i, j] = sum((A_1[i, :] .- A_1[j, :]) .^ 2)
        #         dist[j, i] = dist[i, j]
        #     end            
        # end
        for i in 1 : n
            dist_order[i, :] = sortperm(dist[i, :])
        end
        return dist_order
    end

    function distPivotalSampling(dist_order, _prob, k)

        n = size(dist_order, 1)
        prob = _prob ./ sum(_prob) .* k
        # If we have probabilities bigger than 1.0, we pick those deterministically.
        # Then, adjust the other probabilities so that we get exactly k samples (in expectation in Bernoulli sampling).
        while abs(sum(clamp.(prob, 0.0, 1.0)) - k) > 1e-5
            rescaleFactor = (k - sum(prob .>= 1.0)) / sum(prob[prob .< 1.0])
            prob = prob * rescaleFactor
        end
        sample = collect(1 : n)[:, :][prob .>= 1.0]
        indexProb = [collect(1 : n)[:, :][prob .< 1.0] prob[prob .< 1.0]]   # The first column is the index, the second column is the probability.
        randomOrder = sortperm(rand(size(indexProb, 1)))                    # Look at the indexProb in order.
        loc = zeros(Int64, n)[:, :]                                         # Keep the location in indexProb where each index is stored.
        loc[prob .< 1.0] = collect(1 : size(indexProb, 1))
                                                                            # i, j correspond to the rows in indexProb.
                                                                            # ix_i, ix_j correspond to the rows in A.
        for i in randomOrder[1 : size(randomOrder, 1) - 1]                  # Loop over the rows in indexProb.
            ix_i, p_i = Int(indexProb[i, 1]), indexProb[i, 2]               # Look at the index and probability at i.
            j, ix_j, p_j = 0, 0, 0.0
            for k = 2 : n                                                   # Using the dist_order to find the nearest unfinished neighbor.
                ix_j = dist_order[ix_i, k]
                if loc[ix_j] != 0                                           # If ix_j is finished, we have loc[ix_j] == 0.
                    j = loc[ix_j]                                           # Look up the location of ix_j in indexProb.
                    p_j = indexProb[j, 2]
                    break
                end
            end
            # Do the pivotal match on ix_i and ix_j.
            # The winner index is placed in j.
            if p_i + p_j < 1.0
                if rand() * (p_i + p_j) < p_j
                    indexProb[j, :] = [ix_j, p_i + p_j]
                    loc[ix_i] = 0                               # Mark ix_i as finished.
                    loc[ix_j] = j                               # Update the location of the winner.
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
        # Deal with the floating point issue.
        # In the last step where we have p_i + p_j = 1.0, it may be categorized as p_i + p_j < 1.0.
        if size(sample, 1) < k
            push!(sample, Int(indexProb[randomOrder[size(randomOrder, 1)], 1]))
        end
        return sample, clamp.(prob, 0.0, 1.0)[sample]

    end

end