using KernelAbstractions
using Polyester
using CUDA
using CUDA.CUDAKernels

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using StencilAbstractions

# Make sure all operations are being properly compiled for GPU
CUDA.allowscalar(false)

@stencil function conway(A::Grid, B::Grid) 
    cnt = (A[-1, -1] + A[-1, 0] + A[-1, 1] + A[0, -1] + A[0, 1] + A[1, -1] + A[1, 0] + A[1, 1])
    if (A[0, 0] == 1)
        if (cnt < 2 || cnt > 3) B = 0
        else B = 1
        end
    else
        if (cnt == 3) B = 1
        else B = 0
        end
    end
end

@compile_stencil(CONWAY, conway, ArrayBacking(), Halo(), CUDABackend(), BasicSettings())

function loop(in, out, t_max)
    for t in 1:t_max
        CONWAY(in, out)
        in, out = out, in    
    end    
end

function graph_loop(in, out, t_max)
    @iter(div(t_max, 2), CUDABackend(), 6) do
        CONWAY(in, out)
        CONWAY(out, in)
    end
    for t in 1:(t_max % 2)
        CONWAY(in, out)
        in, out = out, in
    end
end

# Profiling Options
samples = 100
t = 100
test_size = (200, 200)

# Main loops
for i in 1:samples
    init = Int32.(rand(Bool, test_size))
    in1, in_view = alloc(CONWAY.settings, init) 
    out1, out_view = zeros(CONWAY.settings, eltype(init), size(in_view))
    in2 = copy(in1)
    out2 = copy(out1)
    loop(in1, out1, t)
    graph_loop(in2, out2, t)
    @assert in2 == in1
    @assert out2 == out1
end

println("Passed Test")