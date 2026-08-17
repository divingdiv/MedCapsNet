include("common.jl")

dataset = Symbol(get(ARGS, 1, "mnist"))
rng = Xoshiro(parse(Int, getopt(ARGS, "--seed", "1")))
dir = getopt(ARGS, "--modeldir", joinpath("models", String(dataset), "capsnet"))
meta = JLD2.load(joinpath(dir, "meta.jld2"))
model = build_model(:capsnet, meta["n_class"]; rng)
Flux.loadmodel!(model, JLD2.load(joinpath(dir, "best.jld2"))["state"])

_, _, test_d = load_splits(dataset, ARGS, rng)
x = test_d.x[:, :, :, 1:5]
mkpath("images")
preds = reconstruction_grid(model, x, test_d.y[1:5]; path="images/reconstructions.png")
println("true: ", test_d.y[1:5], "  predicted: ", preds)
for dim in 1:16
    perturbation_grid(model, x, dim; path="images/tweak_dim$(dim).png")
end
println("wrote images/reconstructions.png and images/tweak_dim{1..16}.png")
