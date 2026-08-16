function margin_loss(lengths::AbstractMatrix, T::AbstractMatrix;
                     m_plus=0.9f0, m_minus=0.1f0, lambda=0.5f0)
    L = T .* relu.(m_plus .- lengths) .^ 2 .+
        lambda .* (1f0 .- T) .* relu.(lengths .- m_minus) .^ 2
    return mean(sum(L; dims=1))
end
