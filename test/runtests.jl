module StencilTests

using KernelAbstractions
using CUDA
using CUDA.CUDAKernels
#using StencilAbstractions

# Make sure all operations are being properly compiled for GPU
CUDA.allowscalar(false)
CUDA.set_runtime_version!(v"12.6")

# Temp
include("./../src/StencilAbstractions.jl")
using .StencilAbstractions


@stencil function conway(@Input(A)) 
    local_val::Int32 = (A[-1, -1] + A[-1, 0] + A[-1, 1] + A[0, -1] + A[0, 1] + A[1, -1] + A[1, 0] + A[1, 1])
    if (A[0, 0] == 1)
        if (local_val < 2 || local_val > 3) return 0 
        else return 1
        end
    else
        if (local_val == 3) return 1
        else return 0
        end
    end
end

conway_spec = StencilSpecification(conway, CartesianGrid(false), DefaultValue{Int32}(0), CPUBackend(), OptimisationSettings(compiler=BasicGPU()))

arr = Int32[0 1 0 1
            0 0 1 1
            1 1 1 1
            0 0 0 0]

# In, Out are both buffers allocated on the correct device with appropriate padding
in = alloc(conway_spec, arr)
out = zeros(conway_spec, eltype(arr), size(arr))

kernel = compile_stencil(conway_spec)

print(in)
print("\n")
print(out)
print("\n")

# Call kernel, can be run over a specific area by specifying lower and upper
kernel(in, out)

print(in)
print("\n")
print(out)
print("\n")

end