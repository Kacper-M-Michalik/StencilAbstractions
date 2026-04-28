module StencilTests

# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))
# using StencilAbstractions
using KernelAbstractions
using Polyester
using CUDA
using CUDA.CUDAKernels
using BenchmarkTools

# Make sure all operations are being properly compiled for GPU
# CUDA.allowscalar(false)
# CUDA.set_runtime_version!(v"12.6")

# Temp
include("./../src/StencilAbstractions.jl")
using .StencilAbstractions

# one = 1
# @stencil function adder(A::Grid) 
#     A = A[0,0] + one
# end

@stencil function conway(A::Grid{Int32}, B::Grid) 
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

#println(@macroexpand @compile_stencil(conway_cpu_single, conway, ArrayBacking(), Halo(), CPUBackend(), SpatialSettings(threading=SingleThread())))
#println(@macroexpand @compile_stencil(conway_gpu_spatial, conway, ArrayBacking(), Halo(), CUDABackend(), SpatialSettings()))
# println(@macroexpand @compile_stencil(conway_gpu, conway, ArrayBacking(), Halo(), CUDABackend(), BasicSettings()))
# println(@macroexpand @compile_stencil(conway_gpu_spatial, conway, ArrayBacking(), Halo(), CUDABackend(), SpatialSettings()))
@compile_stencil(conway_cpu_single, conway, ArrayBacking(), Halo(), CPUBackend(), SpatialSettings(processing=RedBlack(), threading=SingleThread()))
@compile_stencil(conway_cpu_julia, conway, ArrayBacking(), Halo(), CPUBackend(), SpatialSettings(processing=RedBlack(), threading=JuliaThreads()))
@compile_stencil(conway_cpu, conway, ArrayBacking(), Halo(), CPUBackend(), SpatialSettings(threading=PolyesterThreads()))
@compile_stencil(conway_gpu, conway, ArrayBacking(), Halo(), CUDABackend(), BasicSettings())
@compile_stencil(conway_gpu_spatial, conway, ArrayBacking(), Halo(), CUDABackend(), SpatialSettings())


print_arrs = false
test_size = (1400, 1800)
arr = Int32[0 1 0 1
            0 0 1 1
            1 1 1 1
            0 0 0 0]
arr = Int32.(rand(Bool, test_size))

println("ALLOC")

# in = zeros(Int32, test_size .+ 2)
# out = zeros(Int32, test_size .+ 2)
# inS = zeros(Int32, test_size .+ 2)
# outS = zeros(Int32, test_size .+ 2)
# c_range = (2:1401, 2:1401)
# c_rangeS = (2:1401, 2:1401)
# In, Out are both buffers allocated on the correct device with appropriate padding, with views poiinting to the inner data only if halos are enabled
in, c_range = alloc(conway_cpu.settings, arr) 
out, c_range = zeros(conway_cpu.settings, eltype(arr), size(arr))
inJ, c_range = alloc(conway_cpu_julia.settings, arr) 
outJ, c_range = zeros(conway_cpu_julia.settings, eltype(arr), size(arr))
inS, c_rangeS = alloc(conway_cpu_single.settings, arr) 
outS, c_rangeS = zeros(conway_cpu_single.settings, eltype(arr), size(arr))
inG, c_rangeG = alloc(conway_gpu.settings, arr)
outG, c_rangeG = zeros(conway_gpu.settings, eltype(arr), size(arr))
inGS, c_rangeGS = alloc(conway_gpu_spatial.settings, arr)
outGS, c_rangeGS = zeros(conway_gpu_spatial.settings, eltype(arr), size(arr))

println("COMPUTE")
# # println(in)
# # println(out)
# # println(inG)
# # println(outG)
# # print("\nVIEWS:\n")
# # print(in[bulk])
# # print("\n")
# # print(out[bulk])
# # print("\n")

conway_cpu_single(inS, outS)
conway_cpu_julia(in, out)
conway_cpu(in, out)
@time conway_cpu_single(inS, outS)
@time conway_cpu_julia(inJ, outJ)
@time conway_cpu(in, out)

conway_gpu(inG, outG)
conway_gpu_spatial(inGS, outGS)
@time (conway_gpu(inG, outG); CUDA.synchronize())
@time (conway_gpu_spatial(inGS, outGS); CUDA.synchronize())
# # conway_gpu_spatial(inGS, outGS; range=c_rangeGS)

println("RESULTS")
# if (print_arrs)
#     println(in)
#     println(out)
#     println(inG)
#     println(outG)
#     # print("\nVIEWS:\n")
#     # print(in_view)
#     # print("\n")
#     # print(out_view)
#     # print("\n")
# end

@assert outS == out
@assert Array(outG) == out
@assert Array(outGS) == out

println("PASSED")

end