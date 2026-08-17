struct CapsNet{C1,C2,A,D}
    conv1::C1
    conv2::C2
    W::A                     # (16, 8, 1152, n_class, 1)
    decoder::D
    n_class::Int
    routing_iters::Int
end

Flux.@layer CapsNet

"""CapsNet(n_class; channels=1, routing_iters=3, rng=Random.default_rng())

`rng` seeds only the primary-to-class capsule weight tensor `W`; the `conv1`/
`conv2` layers and the reconstruction `decoder` are initialized from Flux's
default layer constructors, which draw from the global RNG rather than `rng`."""
function CapsNet(n_class::Int; channels::Int=1, routing_iters::Int=3,
                 rng::Random.AbstractRNG=Random.default_rng())
    conv1 = Conv((9, 9), channels => 256, relu)
    conv2 = Conv((9, 9), 256 => 256, relu; stride=2)
    W = 0.01f0 .* randn(rng, Float32, 16, 8, 1152, n_class, 1)
    decoder = Chain(Dense(16 * n_class => 512, relu),
                    Dense(512 => 1024, relu),
                    Dense(1024 => 784, sigmoid))
    return CapsNet(conv1, conv2, W, decoder, n_class, routing_iters)
end

function capsules(m::CapsNet, x)
    u = primary_capsules(m.conv1, m.conv2, x)
    return routing(prediction_vectors(m.W, u), m.routing_iters)
end

class_lengths(v) = dropdims(safe_norm(v; dims=1); dims=1)

"""Decode from class capsules `v` masked by `T` (one-hot: true labels at train
time, predicted labels at test time)."""
function reconstruct(m::CapsNet, v, T)
    masked = v .* reshape(T, 1, m.n_class, :)
    return m.decoder(reshape(masked, 16 * m.n_class, :))
end
