module StencilTests

# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))
# using StencilAbstractions
using KernelAbstractions
using Polyester
# using CUDA
# using CUDA.CUDAKernels

# # Make sure all operations are being properly compiled for GPU
# CUDA.allowscalar(false)
# CUDA.set_runtime_version!(v"12.6")

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

const one = 1
@stencil function adder(A::Grid) 
    A = A[0,0] + one
end


@stencil function compute_V_stencil(Vx::Grid, Vy::Grid, P::Grid, dt::Float64, p::Float64, dx::Float64, dy::Float64)
    Vx = Vx[0, 0] - dt/p * (P[1, 0] - P[0, 0])/dx
    Vy = Vy[0, 0] - dt/p * (P[0, 1] - P[0, 0])/dy
end

@compile_stencil(CONWAY, conway, CartesianGrid(), DefaultValue{Int32}(0), CPUBackend(), BasicSettings())
@compile_stencil(ADDER, adder, CartesianGrid(), DefaultValue{Int32}(0), CPUBackend(), BasicSettings())
#conway_spec = StencilSpecification(conway, CartesianGrid(), DefaultValue{Int32}(0), CPUBackend(), SpatialSettings())

arr = Int32[0 1 0 1
            0 0 1 1
            1 1 1 1
            0 0 0 0]

# In, Out are both buffers allocated on the correct device with appropriate padding, with views poiinting to the inner data only if halos are enabled
in, in_view = alloc(CONWAY.settings, arr) #alloc(conway_spec, arr)
out, out_view = zeros(CONWAY.settings, eltype(arr), size(in_view))

print("ALLOC:\n")
print(in)
print("\n")
print(out)
print("\nVIEWS:\n")
print(in_view)
print("\n")
print(out_view)
print("\n")

# Call kernel, can be run over a specific area by specifying lower and upper

# kernel = compile_stencil(conway_spec)
#ADDER(in)
CONWAY(in, out)

print("RESULTS:\n")
print(in)
print("\n")
print(out)
print("\nVIEWS:\n")
print(in_view)
print("\n")
print(out_view)
print("\n")

end