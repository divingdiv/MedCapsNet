safe_norm(s; dims=1, eps=1f-7) = sqrt.(sum(abs2, s; dims) .+ eps)

function squash(s; dims=1, eps=1f-7)
    sn = sum(abs2, s; dims)
    return (sn ./ (1f0 .+ sn)) .* (s ./ sqrt.(sn .+ eps))
end

"""Conv stack → (8, 1152, B) primary capsules: 6×6×256 output regrouped as
32 capsule maps × 8 consecutive channels per spatial position, then squashed."""
function primary_capsules(conv1, conv2, x)
    h = conv2(conv1(x))                                   # (6, 6, 256, B)
    B = size(h, 4)
    u = reshape(permutedims(h, (3, 1, 2, 4)), 8, :, B)    # channel-major grouping
    return squash(u; dims=1)
end
