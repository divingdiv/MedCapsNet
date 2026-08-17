scores(m::CapsNet, x) = class_lengths(capsules(m, x))
scores(m::Chain, x) = m(x)

"""Argmax class per sample, evaluated in batches (reference uses batches of 512)."""
function predict_classes(m, x::AbstractArray{<:Real,4}; batchsize::Int=512, device=identity)
    m = device(m)
    Flux.testmode!(m)
    preds = Int[]
    for r in Iterators.partition(1:size(x, 4), batchsize)
        s = cpu(scores(m, device(x[:, :, :, r])))
        append!(preds, vec(getindex.(argmax(s; dims=1), 1)))
    end
    return preds
end

function confusion_matrix(ytrue::AbstractVector{Int}, ypred::AbstractVector{Int}, n_class::Int)
    C = zeros(Int, n_class, n_class)
    for (t, p) in zip(ytrue, ypred)
        C[t, p] += 1
    end
    return C
end

function classification_report(C::AbstractMatrix{<:Integer})
    n = size(C, 1)
    prec = [C[i, i] / max(sum(C[:, i]), 1) for i in 1:n]
    rec  = [C[i, i] / max(sum(C[i, :]), 1) for i in 1:n]
    f1   = [p + r == 0 ? 0.0 : 2p * r / (p + r) for (p, r) in zip(prec, rec)]
    return (; precision=prec, recall=rec, f1, accuracy=sum(C[i, i] for i in 1:n) / sum(C))
end

function print_report(io::IO, rep)
    @printf(io, "%8s %10s %10s %10s\n", "class", "precision", "recall", "f1")
    for i in eachindex(rep.precision)
        @printf(io, "%8d %10.3f %10.3f %10.3f\n", i, rep.precision[i], rep.recall[i], rep.f1[i])
    end
    @printf(io, "accuracy: %.4f\n", rep.accuracy)
end
