include("common.jl")

dataset = Symbol(ARGS[1]); arch = Symbol(ARGS[2])
epochs  = parse(Int, getopt(ARGS, "--epochs", "50"))
pct     = parse(Float64, getopt(ARGS, "--percentage", "100"))
seed    = parse(Int, getopt(ARGS, "--seed", "1"))
unb     = getopt(ARGS, "--unbalance", "")          # e.g. "1,9@20" (1-based labels)
rng     = Xoshiro(seed)
device  = resolve_device(ARGS)
reclaim = resolve_reclaim(ARGS)
if arch === :capsnet && device !== identity
    @warn "CapsNet forced to CPU: Metal.jl MPS matmul leak makes GPU training unsafe (see README)"
    device, reclaim = identity, () -> nothing
end

train_d, val_d, _ = load_splits(dataset, ARGS, rng)
pct < 100 && (train_d = limit_data(train_d, pct; rng))
if !isempty(unb)
    labels_str, p_str = split(unb, "@")
    train_d = unbalance_data(train_d, parse.(Int, split(labels_str, ",")), parse(Float64, p_str); rng)
end
hasflag(ARGS, "--augment") && (train_d = augment_data(train_d; flips=dataset === :fashion, rng))

n_class = maximum(train_d.y)
model   = build_model(arch, n_class; rng)
dir     = getopt(ARGS, "--modeldir", joinpath("models", String(dataset), String(arch)))
@info "training" dataset arch n_train = MedCapsNet.nobs(train_d) n_val = MedCapsNet.nobs(val_d) epochs dir

train!(build_lossfn(arch), model, train_d, val_d, n_class;
       epochs, dir, rng, device, reclaim,
       reclaim_every = arch === :capsnet ? 1 : 50,
       batchsize = arch === :capsnet ? 64 : 128,
       weight_decay = arch === :baseline ? 5f-4 : 0f0)
jldsave(joinpath(dir, "meta.jld2"); n_class, arch=String(arch))
