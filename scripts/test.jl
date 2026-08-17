include("common.jl")

dataset = Symbol(ARGS[1]); arch = Symbol(ARGS[2])
rng = Xoshiro(parse(Int, getopt(ARGS, "--seed", "1")))
dir = getopt(ARGS, "--modeldir", joinpath("models", String(dataset), String(arch)))

meta = JLD2.load(joinpath(dir, "meta.jld2"))
model = build_model(Symbol(meta["arch"]), meta["n_class"]; rng)
Flux.loadmodel!(model, JLD2.load(joinpath(dir, "best.jld2"))["state"])

_, _, test_d = load_splits(dataset, ARGS, rng)
ŷ = predict_classes(model, test_d.x)
C = confusion_matrix(test_d.y, ŷ, meta["n_class"])
println("confusion matrix (rows=true, cols=pred):"); display(C); println()
MedCapsNet.print_report(stdout, classification_report(C))
