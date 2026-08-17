include("common.jl")
using Statistics

dataset  = Symbol(get(ARGS, 1, "mnist"))
epochs   = parse(Int, getopt(ARGS, "--epochs", "25"))
seeds    = parse.(Int, split(getopt(ARGS, "--seeds", "1,2,3"), ","))
percents = parse.(Float64, split(getopt(ARGS, "--percents", "5,10,50,100"), ","))
archs    = Symbol.(split(getopt(ARGS, "--archs", "capsnet,lenet,baseline"), ","))
out      = getopt(ARGS, "--out", "results.csv")
device   = resolve_device(ARGS)
reclaim  = resolve_reclaim(ARGS)

function run_config(; dataset, arch, epochs, seed, pct=100.0, unb=nothing, augment=false,
                    device=identity, reclaim=() -> nothing)
    rng = Xoshiro(seed)
    full = load_vision(dataset; split=:train)
    tr, val = split_validation(full, 5000; rng)
    pct < 100 && (tr = limit_data(tr, pct; rng))
    unb !== nothing && (tr = unbalance_data(tr, unb.labels, unb.pct; rng))
    augment && (tr = augment_data(tr; flips=dataset === :fashion, rng))
    n = 10
    model = build_model(arch, n; rng)
    dir = mktempdir()
    train!(build_lossfn(arch), model, tr, val, n; epochs, dir, rng, device, reclaim,
           batchsize=arch === :capsnet ? 64 : 128,
           weight_decay=arch === :baseline ? 5f-4 : 0f0)
    Flux.loadmodel!(model, JLD2.load(joinpath(dir, "best.jld2"))["state"])
    te = load_vision(dataset; split=:test)
    return classification_report(confusion_matrix(te.y, predict_classes(model, te.x; device), n))
end

isfile(out) || open(io -> println(io, "dataset,arch,experiment,param,seed,accuracy,mean_f1"), out, "w")

for arch in archs, seed in seeds
    rec(exp, param, rep) = open(out, "a") do io
        println(io, join([dataset, arch, exp, param, seed,
                          round(rep.accuracy; digits=4), round(mean(rep.f1); digits=4)], ","))
    end
    for pct in percents                                        # study 1: limited data
        rec("limited", pct, run_config(; dataset, arch, epochs, seed, pct, device, reclaim))
    end
    rec("imbalance", "1;9@20",                                  # study 2: imbalance
        run_config(; dataset, arch, epochs, seed, unb=(labels=[1, 9], pct=20.0), device, reclaim))
    rec("augment", "on",                                        # study 3: augmentation
        run_config(; dataset, arch, epochs, seed, augment=true, device, reclaim))
    @info "finished" arch seed
end
println("results appended to $out")
