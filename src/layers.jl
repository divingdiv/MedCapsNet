safe_norm(s; dims=1, eps=1f-7) = sqrt.(sum(abs2, s; dims) .+ eps)

function squash(s; dims=1, eps=1f-7)
    sn = sum(abs2, s; dims)
    return (sn ./ (1f0 .+ sn)) .* (s ./ sqrt.(sn .+ eps))
end
