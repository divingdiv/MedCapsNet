using MedCapsNet, Flux, Random, JLD2

getopt(args, name, default) = (i = findfirst(==(name), args); i === nothing ? default : args[i + 1])
hasflag(args, name) = name in args

build_model(arch::Symbol, n_class::Int; rng) =
    arch === :capsnet  ? CapsNet(n_class; rng) :
    arch === :lenet    ? lenet(n_class) :
    arch === :baseline ? baseline_cnn(n_class) :
    error("unknown arch $arch")

build_lossfn(arch::Symbol) = arch === :capsnet ? capsnet_loss : cnn_loss

function resolve_device(args)
    if getopt(args, "--device", "cpu") == "gpu"
        @eval Main using Metal
        return Flux.gpu
    end
    return identity
end

function load_splits(dataset::Symbol, args, rng)
    if dataset === :medical
        dir = getopt(args, "--datadir", "data/diaretdb1")
        tr, val, te = load_medical(dir)
        if val === nothing
            tr, val = split_validation(tr, max(1, MedCapsNet.nobs(tr) ÷ 10); rng)
        end
        return tr, val, te
    end
    full = load_vision(dataset; split=:train)
    tr, val = split_validation(full, 5000; rng)
    return tr, val, load_vision(dataset; split=:test)
end
