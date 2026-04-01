module StencilAbstractions

using DataStructures
using StaticArrays
using KernelAbstractions
using LoopVectorization
using CUDA
# using AMDGPU
# using oneAPI
# using Metal

export
    CPUBackend,
    CartesianGrid, TreeGrid, 
    GridBoundary, DefaultValue, WrapAround, Mirror, MirrorWithBounce,
    Halo, Haloless,
    OptimisationSettings, BasicCPU, RegisterPackingCPU, RPDiamondTilingCPU, BasicGPU, GatherComputeGPU, OverlappedTilingGPU,
    @stencil, @Output, @Input, @returns,
    StencilEquation, compile_stencil, alloc

include("Defs.jl")
include("Parser.jl")
include("CPUHalolessCompiler.jl")
include("GPUHalolessCompiler.jl")

macro stencil(equation)      
    if !(equation isa Expr && equation.head === :function)
        error("Invalid stencil declaration!")
    end
    
    stencil_name = equation.args[1].args[1]
    stencil_object = process_stencil_equation(equation)

    return esc(quote    
        $stencil_name = $stencil_object
    end)
end

macro stencil(vars...)
    error("Invalid stencil declaration!")
end

macro Input(expr)
    return esc(expr)
end

macro Output(expr)
    return esc(expr)
end

macro returns(expr)
    return esc(expr)
end

function alloc(stencil::StencilEquation, grid::CartesianGrid{B, Halo, CPUBackend, S}, array::AbstractArray)
    # Calculate halo size, allcoate then return
    # TODO implement
    return Array(array)
end
function alloc(stencil::StencilEquation, grid::CartesianGrid{B, Haloless, CPUBackend, S}, array::AbstractArray)
    if (array isa Array)
        print("Unnecessary array cast through GridSettings, CPU array being copied to CPU array!\n")
    end
    return Array(array)
end

function alloc(stencil::StencilEquation, grid::CartesianGrid{B, Halo, CUDA.CUDABackend}, array::AbstractArray) where {B <: GridBoundary}
    # Calculate halo size, allcoate then return
    # TODO implement
    return CuArray(array)
end
function alloc(stencil::StencilEquation, grid::CartesianGrid{B, Haloless, CUDA.CUDABackend}, array::AbstractArray) where {B <: GridBoundary}
    return CuArray(array)
end


# function (grid::CartesianGrid{B, Halo, CPUBackend})(stencil::StencilEquation, array::AbstractArray) where {B <: GridBoundary}
#     # Calculate halo size, allcoate then return
#     # TODO implement
#     return Array(array)
# end
# function (grid::CartesianGrid{B, Haloless, CPUBackend})(stencil::StencilEquation, array::AbstractArray) where {B <: GridBoundary}
#     if (array isa Array)
#         print("Unnecessary array cast through GridSettings, CPU array being copied to CPU array!\n")
#     end
#     return Array(array)
# end

# function (grid::CartesianGrid{B, Halo, CUDA.CUDABackend})(stencil::StencilEquation, array::AbstractArray) where {B <: GridBoundary}
#     # Calculate halo size, allcoate then return
#     # TODO implement
#     return CuArray(array)
# end
# function (grid::CartesianGrid{B, Haloless, CUDA.CUDABackend})(stencil::StencilEquation, array::AbstractArray) where {B <: GridBoundary}
#     return CuArray(array)
# end

# function (grid::CartesianGrid{B, H, AMDGPU.ROCBackend})(stencil::StencilEquation, array::AbstractArray) where {B <: GridBoundary, H <: HaloType}
#     print("HELLO AMD!")
# end

# function (grid::CartesianGrid{B, H, Metal.MetalBackend})(stencil::StencilEquation, array::AbstractArray) where {B <: GridBoundary, H <: HaloType}
#     print("HELLO METAL!")
# end

# function (grid::CartesianGrid{B, H, oneAPI.oneAPIBackend})(stencil::StencilEquation, array::AbstractArray) where {B <: GridBoundary, H <: HaloType}
#     print("HELLO ONEAPI!")
# end

#paramterise on gridsettings -> halo requires calculatign view before running
function (stencil::CompiledStencil)(args...; range)
    # TODO: Should we check arg types?
    # TODO: For kernels with 4D+ need to deal with group size dimension limits
    stencil.kernel_pointer(args...; ndrange = range)
    # print("Ev:\n")
    # print(ev)
    # print("\n")
    # dump(ev)

    # TODO: DO EVENT BASED SYNCHRONISATION INSTEAD?
    KernelAbstractions.synchronize(stencil.settings.backend)
end

end