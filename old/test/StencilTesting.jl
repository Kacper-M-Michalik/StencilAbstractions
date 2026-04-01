module StencilTesting

using KernelAbstractions
using CUDA
using CUDA.CUDAKernels
# Make sure all operations are being properly compiled for GPU
CUDA.allowscalar(false)

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

@stencil function conway2(@Input(A), @Output(B)) 
    local_val::Int32 = (A[-1, -1] + A[-1, 0] + A[-1, 1] + A[0, -1] + A[0, 1] + A[1, -1] + A[1, 0] + A[1, 1])
    if (A[0,0] == 1) 
        if (local_val < 2 || local_val > 3) @return 0, 1 
        else @returns 1, 0
        end
    else
        if (local_val == 3) @return 1, 0
        else @returns 0, 1
        end
    end
end

# @stencil function conway3(A, @Output(B)) 
#     local_val::Int32 = (A[-1, -1] + A[-1, 0] + A[-1, 1] + A[0, -1] + A[0, 1] + A[1, -1] + A[1, 0] + A[1, 1])
#     if (A[0, 0] == 1)
#         if (local_val < 2 || local_val > 3) 
#             B = 0 + 1
#             return 
#         else
#             B = 1
#             return
#         end
#     else
#         if (local_val == 3) 
#             B = 1
#             return
#         else 
#             B = 0
#             return
#         end
#     end
# end

@stencil function data_copy(@Input(A)) 
    return A[0, 0] 
end

@stencil function laplace(@Input(A)) 
    return 0.25 * (A[1, 0] + A[-1, 0] + A[0, 1] + A[0, -1])
end

# Perform expr inlining
#new_stencil = compose(data_copy, laplace)
#lowered_m = compile_stencil(multi_arg, StencilSettings(CartesianGrid(), DefaultValue{Float32}(5)))
#lowered_laplace_const = compile_stencil(laplace_const, StencilSettings(CartesianGrid(), DefaultValue{Float32}(0)))
#get_backend(zeros(1, 1))

setting1 = CartesianGrid(DefaultValue{Int32}(0), Haloless(), CUDABackend())
setting2 = CartesianGrid(DefaultValue{Int32}(0), Haloless(), CPUBackend())
setting3 = CartesianGrid(DefaultValue{Int32}(0), Haloless(), CUDABackend(), BasicGPU())

arr = Int32[0 1 0 1
            0 0 1 1
            1 1 1 1
            0 0 0 0]

arr2 = zeros(Int32, size(arr))

# In, Out are both buffers allocated on the correct device with appropriate padding
in = alloc(conway, setting2, arr)
out = alloc(conway, setting2, arr2)

# Prints that uneccesary copy was performed given settings
unnecessary = alloc(conway, setting2, arr) 

kernel = compile_stencil(conway, setting2, OptimisationSettings(enable_fastmath=true, enable_bounds_checks=false))

# Allow views to be run in range, and a default range which covers grid
kernel(in, out; range = size(in))

print(in)
print(out)

end