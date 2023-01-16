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

end