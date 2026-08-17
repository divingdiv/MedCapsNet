include("common.jl")
using FileIO, Colors, Statistics, DelimitedFiles
using MedCapsNet: extract_patch, macenko_hematoxylin
using ImageTransformations: imrotate

const RAW = "raw_data/tupac16"
const OUT = "data/tupac16"
const N_POS_PER_IMAGE = 30             # spec: 30 jittered positives per image
const N_NEG_PER_IMAGE = 3              # spec: 3 negative grid cells per image
const HALF = 50                        # spec: 100×100 patches

isdir(joinpath(RAW, "mitoses_image_data")) ||
    error("TUPAC16 mitoses aux dataset not found — download into $RAW/ first (see README)")

rng = Xoshiro(parse(Int, getopt(ARGS, "--seed", "1")))

# split lists: one case id per line; fall back to 70/10/20 by sorted case order
function split_cases(cases)
    if all(isfile.(joinpath.(RAW, ["train.csv", "val.csv", "test.csv"])))
        rd(f) = vec(string.(readdlm(joinpath(RAW, f))))
        return rd("train.csv"), rd("val.csv"), rd("test.csv")
    end
    n = length(cases)
    return cases[1:round(Int, 0.7n)],
           cases[round(Int, 0.7n)+1:round(Int, 0.8n)],
           cases[round(Int, 0.8n)+1:end]
end

function image_patches(img_path, csv_path; augment::Bool)
    h = macenko_hematoxylin(load(img_path))
    H, W = size(h)
    centers = isfile(csv_path) ?
        [(Int(r[1]), Int(r[2])) for r in eachrow(readdlm(csv_path, ','))] : Tuple{Int,Int}[]
    xs, ys = Matrix{Float32}[], Int[]
    # spec: independent 50% probability horizontal flip and 50% probability
    # vertical flip (a patch may get neither, either, or both)
    function maybe_aug!(p)
        augment || return p
        rand(rng) < 0.5 && (p = reverse(p; dims=2))
        rand(rng) < 0.5 && (p = reverse(p; dims=1))
        return p
    end
    if !isempty(centers)                                   # positives: label 1
        for k in 1:N_POS_PER_IMAGE
            cy, cx = centers[mod1(k, length(centers))]
            push!(xs, maybe_aug!(extract_patch(h, cy, cx; half=HALF, jitter=30, rng)))
            push!(ys, 1)
        end
    end
    # negatives: 3×3 grid, cells containing no mitosis center — label 2
    ch, cw = H ÷ 3, W ÷ 3
    free = [(r, c) for r in 1:3, c in 1:3 if
            !any(((cy, cx),) -> (cy - 1) ÷ ch + 1 == r && (cx - 1) ÷ cw + 1 == c, centers)]
    for (r, c) in shuffle(rng, vec(free))[1:min(N_NEG_PER_IMAGE, length(free))]
        cy, cx = (r - 1) * ch + ch ÷ 2, (c - 1) * cw + cw ÷ 2
        push!(xs, maybe_aug!(extract_patch(h, cy, cx; half=HALF, jitter=30, rng)))
        push!(ys, 2)
    end
    return xs, ys
end

function collect_split(cases; augment)
    xs, ys = Matrix{Float32}[], Int[]
    for case in cases
        cdir = joinpath(RAW, "mitoses_image_data", case)
        isdir(cdir) || continue
        for f in filter(endswith(".tif"), readdir(cdir))
            csv = joinpath(RAW, "mitoses_ground_truth", case, replace(f, ".tif" => ".csv"))
            px, py = image_patches(joinpath(cdir, f), csv; augment)
            append!(xs, px); append!(ys, py)
        end
    end
    x4 = Array{Float32,4}(undef, 28, 28, 1, length(ys))
    for (i, p) in enumerate(xs); x4[:, :, 1, i] = p; end
    p = randperm(rng, length(ys))
    return x4[:, :, :, p], ys[p]
end

cases = sort(readdir(joinpath(RAW, "mitoses_image_data")))
tr_c, va_c, te_c = split_cases(cases)
train_x, train_y = collect_split(tr_c; augment=true)
val_x, val_y     = collect_split(va_c; augment=false)
test_x, test_y   = collect_split(te_c; augment=false)
mkpath(OUT)
jldsave(joinpath(OUT, "data.jld2"); train_x, train_y, val_x, val_y, test_x, test_y,
        mean_value=Float32(mean(train_x)))
println("wrote $(OUT)/data.jld2: train=$(length(train_y)) val=$(length(val_y)) test=$(length(test_y))")
