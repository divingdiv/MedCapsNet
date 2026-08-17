include("common.jl")
using FileIO, Colors, ImageContrastAdjustment, Statistics
using MedCapsNet: crop_bbox, extract_patch
using ImageTransformations: imrotate

const RAW = "raw_data/diaretdb1"
const OUT = "data/diaretdb1"
const CLASSES = [["softexudates", "hardexudates"], ["hemorrhages", "redsmalldots"]]
const GRID = 300                       # px grid cells (spec)
const TRAIN_PER_CLASS = 480            # augmented counts (spec)
const TEST_COUNTS = [60, 70]

isdir(joinpath(RAW, "ddb1_fundusimages")) ||
    error("DIARETDB1 not found — download it into $RAW/ first (see README)")

rng = Xoshiro(parse(Int, getopt(ARGS, "--seed", "1")))
all_names = sort(readdir(joinpath(RAW, "ddb1_fundusimages")))
# DIARETDB1 protocol: images 1–28 train, 29–89 test (use the repo's txt lists if present)
train_names = isfile(joinpath(RAW, "train_images.txt")) ?
    readlines(joinpath(RAW, "train_images.txt")) : all_names[1:28]
test_names = isfile(joinpath(RAW, "test_images.txt")) ?
    readlines(joinpath(RAW, "test_images.txt")) : all_names[29:end]

"CLAHE on luminance then green channel in [0,1] (reference: LAB CLAHE clip 3.0)."
function enhance_green(img)
    eq = adjust_histogram(img, AdaptiveEqualization(nbins=256, clip=0.03))
    return Float32.(green.(eq))
end

function lesion_mask(name, dirs, box)
    m = falses(length(box[1]), length(box[2]))
    for d in dirs
        p = joinpath(RAW, "ddb1_groundtruth", d, name)
        isfile(p) || continue
        m .|= (Float32.(Gray.(load(p)))[box...] .> 0.5f0)
    end
    return m
end

"Centers of mass of lesion pixels per occupied GRID×GRID cell."
function lesion_centers(mask)
    centers = Tuple{Int,Int}[]
    H, W = size(mask)
    for r0 in 1:GRID:H, c0 in 1:GRID:W
        cell = @view mask[r0:min(r0 + GRID - 1, H), c0:min(c0 + GRID - 1, W)]
        idx = findall(cell)
        isempty(idx) && continue
        push!(centers, (r0 - 1 + round(Int, mean(getindex.(idx, 1))),
                        c0 - 1 + round(Int, mean(getindex.(idx, 2)))))
    end
    return centers
end

function collect_patches(names, per_class_target; augment::Bool)
    xs, ys = Matrix{Float32}[], Int[]
    for name in names
        rgb = load(joinpath(RAW, "ddb1_fundusimages", name))
        box = crop_bbox(Float32.(green.(rgb)))
        g = enhance_green(rgb[box...])
        for (cls, dirs) in enumerate(CLASSES)
            for (cy, cx) in lesion_centers(lesion_mask(name, dirs, box))
                p = extract_patch(g, cy, cx; half=100, jitter=30, rng)
                push!(xs, p); push!(ys, cls)
                if augment                              # 50% rot ±10°, hflip, vflip
                    rand(rng) < 0.5 && (push!(xs, Float32.(imrotate(p, deg2rad((2rand(rng) - 1) * 10), axes(p); fillvalue=0f0))); push!(ys, cls))
                    rand(rng) < 0.5 && (push!(xs, reverse(p; dims=2)); push!(ys, cls))
                    rand(rng) < 0.5 && (push!(xs, reverse(p; dims=1)); push!(ys, cls))
                end
            end
        end
    end
    # balance to target counts per class by resampling with fresh jitter is complex;
    # simple truncation/shuffle per class matches reference counts:
    out_x, out_y = Matrix{Float32}[], Int[]
    for cls in 1:2
        idx = shuffle(rng, findall(==(cls), ys))
        tgt = per_class_target isa Vector ? per_class_target[cls] : per_class_target
        length(idx) < tgt &&
            @warn "class patch count fell short of target" class=cls got=length(idx) target=tgt
        idx = idx[1:min(tgt, length(idx))]
        append!(out_x, xs[idx]); append!(out_y, fill(cls, length(idx)))
    end
    x4 = Array{Float32,4}(undef, 28, 28, 1, length(out_y))
    for (i, p) in enumerate(out_x); x4[:, :, 1, i] = p; end
    return x4, out_y
end

train_x, train_y = collect_patches(train_names, TRAIN_PER_CLASS; augment=true)
test_x, test_y   = collect_patches(test_names, TEST_COUNTS; augment=false)
mkpath(OUT)
jldsave(joinpath(OUT, "data.jld2"); train_x, train_y, test_x, test_y,
        mean_value=Float32(mean(train_x)))
println("wrote $(OUT)/data.jld2: train=$(length(train_y)) test=$(length(test_y))")
