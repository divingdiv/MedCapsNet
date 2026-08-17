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

"""û[k, i, j, b] = W[:, :, i, j] * u[:, i, b] via batched matmul over the 1152
primary capsules — no (16,8,1152,n,B) materialization; identical values to the
broadcast formulation (parity-tested)."""
function prediction_vectors(W, u)
    dk, dm, ni, nc = size(W, 1), size(W, 2), size(W, 3), size(W, 4)
    B = size(u, 3)
    W2 = reshape(permutedims(dropdims(W; dims=5), (1, 4, 2, 3)), dk * nc, dm, ni)  # (16n, 8, 1152)
    u2 = permutedims(u, (1, 3, 2))                                                 # (8, B, 1152)
    r = batched_mul(W2, u2)                                                        # (16n, B, 1152)
    return permutedims(reshape(r, dk, nc, B, ni), (1, 4, 2, 3))                    # (16, 1152, n, B)
end

"""Dynamic routing (Sabour et al. 2017), `iters` iterations, softmax over classes."""
function routing(û::AbstractArray{T,4}, iters::Int) where {T<:Real}
    b = zero(T) .* sum(û; dims=1)                       # (1, 1152, n, B) zeros, AD-safe
    v = squash(sum(softmax(b; dims=3) .* û; dims=2); dims=1)
    for _ in 2:iters
        b = b .+ sum(û .* v; dims=1)                    # agreement update
        v = squash(sum(softmax(b; dims=3) .* û; dims=2); dims=1)
    end
    return dropdims(v; dims=2)                          # (16, n, B)
end
