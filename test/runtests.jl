module StencilTests

# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))
# using StencilAbstractions
using KernelAbstractions
using Polyester
using CUDA
using CUDA.CUDAKernels

# Make sure all operations are being properly compiled for GPU
CUDA.allowscalar(false)
# CUDA.set_runtime_version!(v"12.6")

test = quote
    @stencil function conway(A::Grid{Float64}, B::Grid{Float64}) 
    end
end
dump(test)

# Temp
include("./../src/StencilAbstractions.jl")
using .StencilAbstractions


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

one = 1
@stencil function adder(A::Grid) 
    A = A[0,0] + one
end

# @compile_stencil(CONWAY, conway, CartesianGrid(), (DefaultValue{Int32}(0)), CPUBackend(), BasicSettings())
# @compile_stencil(ADDER, adder, CartesianGrid(), (DefaultValue{Int32}(0)), CPUBackend(), BasicSettings())
#conway_spec = StencilSpecification(conway, CartesianGrid(), DefaultValue{Int32}(0), CPUBackend(), SpatialSettings())
#alloc(conway_spec, arr)

@compile_stencil(conway_cpu_single, conway, ArrayBacking(), Halo(), CPUBackend(), SpatialSettings(threading=SingleThread()))
@compile_stencil(conway_cpu, conway, ArrayBacking(), Halo(), CPUBackend(), SpatialSettings(threading=PolyesterThreads()))
@compile_stencil(conway_gpu, conway, ArrayBacking(), Halo(), CUDABackend(), BasicSettings())
@compile_stencil(conway_gpu_spatial, conway, ArrayBacking(), Halo(), CUDABackend(), SpatialSettings())

print_arrs = false
test_size = (8000, 8000)
arr = Int32[0 1 0 1
            0 0 1 1
            1 1 1 1
            0 0 0 0]
arr = Int32.(rand(Bool, test_size))

println("ALLOC")
# In, Out are both buffers allocated on the correct device with appropriate padding, with views poiinting to the inner data only if halos are enabled
in, c_range = alloc(conway_cpu.settings, arr) 
out, c_range = zeros(conway_cpu.settings, eltype(arr), size(arr))
inS, c_rangeS = alloc(conway_cpu_single.settings, arr) 
outS, c_rangeS = zeros(conway_cpu_single.settings, eltype(arr), size(arr))
inG, c_rangeG = alloc(conway_gpu.settings, arr)
outG, c_rangeG = zeros(conway_gpu.settings, eltype(arr), size(arr))
inGS, c_rangeGS = alloc(conway_gpu_spatial.settings, arr)
outGS, c_rangeGS = zeros(conway_gpu_spatial.settings, eltype(arr), size(arr))
println("COMPUTE")
# println(in)
# println(out)
# println(inG)
# println(outG)
# print("\nVIEWS:\n")
# print(in[bulk])
# print("\n")
# print(out[bulk])
# print("\n")

# conway_cpu_single(inS, outS; range=c_range)
# conway_cpu(in, out; range=c_range)
# @time conway_cpu_single(inS, outS; range=c_range)
# @time conway_cpu(in, out; range=c_range)

# conway_gpu(inG, outG; range=c_rangeG)
conway_gpu_spatial(inGS, outGS; range=c_rangeGS)
# CUDA.synchronize()
# sleep(0.2)
# @time conway_gpu(inG, outG; range=c_rangeG)
# @time conway_gpu_spatial(inGS, outGS; range=c_rangeGS)
CUDA.@profile conway_gpu_spatial(inGS, outGS; range=c_rangeGS)

println("RESULTS")
if (print_arrs)
    println(in)
    println(out)
    println(inG)
    println(outG)
    # print("\nVIEWS:\n")
    # print(in_view)
    # print("\n")
    # print(out_view)
    # print("\n")
end

# @assert outS == out
# @assert Array(outG) == out
# @assert Array(outGS) == out

# println("PASSED")

sleep(2)

end