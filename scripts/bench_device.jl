include("common.jl")
using Statistics: median

const N_CLASS = 10
const N_INFER = 1024   # images for the per-image inference benchmark

"""
    bench(f, sync!; warm=1, runs=7) -> (; min, median)   [milliseconds]

Runs `f()` `warm` times (untimed) then `runs` times (timed), returning the
minimum and median wall-clock duration in ms. 7 runs / median chosen over the
brief's 5 given the ~11% run-to-run spread observed in Task 1's spike.

`sync!()` is called *inside* the timed region, immediately after `f()`. This
matters on GPU: Metal dispatches kernels asynchronously, so a bare `@elapsed`
around `Flux.withgradient` only measures how long it takes to *submit* work,
not how long the GPU takes to finish it. The forward-pass loss is a plain
scalar (`margin_loss`/`cnn_loss` reduce all the way down with a dims-less
`sum`/`mean`), and pulling a single scalar off the GPU already blocks until
that reduction is done — so `fwd` timings are honest even with `sync! = ()
-> nothing`. The backward pass is different: `Flux.withgradient` returns
gradients as *unread* device arrays, so nothing forces the CPU to wait for
the backward kernels to actually finish running — only for them to be
enqueued. Without an explicit sync, `step` timings on GPU silently measure
kernel-submission time instead of kernel-execution time (this is exactly the
implausible "GPU 100x faster" failure mode called out in the task notes).
`sync!() = Metal.synchronize()` drains the whole command queue, fixing this.
`predict_classes` calls `cpu(...)` on every batch internally, which forces a
blocking readback per batch, so its `sync!()` call is also belt-and-suspenders
rather than load-bearing — kept for uniformity/defensiveness, not because it
changes the measurement.

Sync method used: `Metal.synchronize()` (a plain function, not the
`Metal.@sync` macro). `Metal.@sync ex` macro-expands at the point its
surrounding top-level form is lowered — which for code written above the
`for devname in ["cpu", "gpu"]` loop happens *before* `resolve_device("gpu")`
has ever run `@eval Main using Metal`, so `Metal` would not yet be a
resolvable name and the macro would fail to expand. `Metal.synchronize()` is
an ordinary function call, resolved dynamically at the time it *executes*
(not when the enclosing code is defined), so it is safe to reference from a
closure built once for both devices, as long as `Metal` is loaded by the time
that closure actually runs on the GPU branch — which it always is, since GPU
rows run strictly after `resolve_device(["--device", "gpu"])`.
"""
function bench(f, sync!; warm=1, runs=7)
    for _ in 1:warm
        f(); sync!()
    end
    ts = [(@elapsed (f(); sync!())) for _ in 1:runs]
    return (min=round(minimum(ts) * 1000; digits=1), median=round(median(ts) * 1000; digits=1))
end

fmt(t) = "$(t.min) / $(t.median)"

results = NamedTuple[]
println("device | arch | step ms (min/median) | fwd ms (min/median) | infer ms/1k (min/median)")
println("--- | --- | --- | --- | ---")
for devname in ["cpu", "gpu"]
    device = resolve_device(["--device", devname])
    sync! = devname == "gpu" ? (() -> Metal.synchronize()) : (() -> nothing)
    devname == "gpu" && Metal.allowscalar(false)
    for (arch, batch) in [(:capsnet, 64), (:lenet, 128)]
        m = device(build_model(arch, N_CLASS; rng=Xoshiro(1)))
        lossfn = build_lossfn(arch)
        x = device(rand(Float32, 28, 28, 1, batch))
        T = device(Float32.(Flux.onehotbatch(rand(1:N_CLASS, batch), 1:N_CLASS)))

        step = bench(() -> Flux.withgradient(mm -> lossfn(mm, x, T), m), sync!)
        fwd  = bench(() -> lossfn(m, x, T), sync!)

        ximg  = device(rand(Float32, 28, 28, 1, N_INFER))
        infer = bench(() -> predict_classes(m, ximg; device=identity), sync!)
        infer_per1k = (min=round(infer.min / N_INFER * 1000; digits=1),
                       median=round(infer.median / N_INFER * 1000; digits=1))

        push!(results, (; devname, arch, step, fwd, infer=infer_per1k))
        println("$devname | $arch | $(fmt(step)) | $(fmt(fwd)) | $(fmt(infer_per1k))")
    end
end

cpu_step = only(r.step for r in results if r.devname == "cpu" && r.arch === :capsnet)
gpu_step = only(r.step for r in results if r.devname == "gpu" && r.arch === :capsnet)

println()
println("capsnet grad step (median ms): cpu=$(cpu_step.median)  gpu=$(gpu_step.median)")
if gpu_step.median <= cpu_step.median
    println("DECISION: rerun device = GPU (gpu capsnet step $(gpu_step.median)ms <= cpu $(cpu_step.median)ms)")
else
    println("DECISION: STOP — GPU capsnet step ($(gpu_step.median)ms) is slower than CPU's ($(cpu_step.median)ms). Do not proceed to Task 4 reruns without surfacing this to the user.")
end
