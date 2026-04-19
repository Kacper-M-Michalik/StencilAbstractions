module Overlapped

# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))
# using StencilAbstractions
using KernelAbstractions
using Polyester
# using CUDA
# using CUDA.CUDAKernels
using StaticArrays

# # Make sure all operations are being properly compiled for GPU
# CUDA.allowscalar(false)
# CUDA.set_runtime_version!(v"12.6")

# Temp
include("./../src/StencilAbstractions.jl")
using .StencilAbstractions


@stencil function conway(A::Grid) 
    cnt = (A[-1, -1] + A[-1, 0] + A[-1, 1] + A[0, -1] + A[0, 1] + A[1, -1] + A[1, 0] + A[1, 1])
    if (A[0, 0] == 1)
        if (cnt < 2 || cnt > 3) A = 0
        else A = 1
        end
    else
        if (cnt == 3) A = 1
        else A = 0
        end
    end
end

@stencil function laplace(A::Grid) 
    val = (A[0, 0] + A[-1, 0] + A[0, -1] + A[0, 1] + A[1, 0])
    A = val / 5.0
end


@compile_stencil(CONWAY, conway, CartesianGrid(), DefaultValue{Int32}(0), CPUBackend(), BasicSettings())
@compile_stencil(LAP, laplace, CartesianGrid(), DefaultValue{Int32}(0), CPUBackend(), BasicSettings())

println(Greedy(LAP.settings, (5, 5)))

# init = Int32.(rand(Bool, test_size))
# in, _ = alloc(CONWAY.settings, init) 
# out, _ = zeros(CONWAY.settings, eltype(init), size(in_view))

# CONWAY((in, out))

# @stencil function box_blur_copy(A::Grid, B::Grid)
#     B = sum(@Box(A, 2, 1)) / 9.0
# end

# @compile_stencil(CONWAY, box_blur_copy, CartesianGrid(), DefaultValue{Int32}(0), CPUBackend(), SpatialSettings())

# init = Float32.(rand(Bool, (10, 10)))
# in, _ = alloc(CONWAY.settings, init) 
# out, _ = zeros(CONWAY.settings, eltype(init), size(init))
# CONWAY(in, out)
# println(in)
# println(out)

end