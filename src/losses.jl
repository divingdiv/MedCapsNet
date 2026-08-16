function margin_loss(lengths::AbstractMatrix, T::AbstractMatrix;
                     m_plus=0.9f0, m_minus=0.1f0, lambda=0.5f0)
    L = T .* relu.(m_plus .- lengths) .^ 2 .+
        lambda .* (1f0 .- T) .* relu.(lengths .- m_minus) .^ 2
    return mean(sum(L; dims=1))
end

function reconstruction_loss(x̂, x)
    xf = reshape(x, size(x̂, 1), :)
    return mean(sum(abs2, x̂ .- xf; dims=1))
end

"""Total CapsNet objective: margin loss + alpha · reconstruction SSE (alpha=0.0005)."""
function capsnet_loss(m::CapsNet, x, T; alpha=5f-4)
    v = capsules(m, x)
    return margin_loss(class_lengths(v), T) +
           alpha * reconstruction_loss(reconstruct(m, v, T), x)
end
