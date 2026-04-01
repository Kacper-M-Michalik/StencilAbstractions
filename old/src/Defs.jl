const CPUBackend = KernelAbstractions.CPU
# const CUDABackend = CUDA.CUDABackend
# const ROCBackend = AMDGPU.ROCBackend
# const MetalBackend = Metal.MetalBackend
# const oneAPIBackend = oneAPI.oneAPIBackend

abstract type GridBoundary end
struct WrapAround <: GridBoundary end
struct Mirror <: GridBoundary end
struct MirrorWithBounce <: GridBoundary end
struct DefaultValue{T} <: GridBoundary 
    value::T
end
# Skip boundary?

abstract type HaloType end
struct Halo <: HaloType end
struct Haloless <: HaloType end

# Change into backend, then strategy types?
# Make register blocking into a toggle not strategy? 
abstract type ComputationStrategy end
abstract type CPUStrategy <: ComputationStrategy end
abstract type GPUStrategy <: ComputationStrategy end
struct BasicCPU <: CPUStrategy end
struct RegisterBlockingCPU <: CPUStrategy end
struct RegisterBlockingDiamondTilingCPU <: CPUStrategy end
struct BasicGPU <: GPUStrategy end
struct GatherComputeGPU <: GPUStrategy end
struct OverlappedTilingGPU <: GPUStrategy 
    timesteps::Int32
end

Base.@kwdef struct OptimisationSettings
    # Shared 
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false
    # GPU Only
    block_size = (16, 16)
    warp_size = 32
    tile_size = 0
    # CPU Only
    thread_count = 4    
end

# May have to change from KA, need to discrimiante bwteen CUDA and ROC etc.
abstract type GridSettings end
struct CartesianGrid{B <: GridBoundary, H <: HaloType, D <: KernelAbstractions.Backend, S <: ComputationStrategy} <: GridSettings 
    boundary::B
    halo::H
    backend::D
    strategy::S

    function CartesianGrid(Boundary::B, Halo::H, Backend::D) where {B <: GridBoundary, H <: HaloType, D <: KernelAbstractions.GPU}
        new{B, H, D, BasicGPU}(Boundary, Halo, Backend, BasicGPU())
    end

    function CartesianGrid(Boundary::B, Halo::H, Backend::D, Strategy::S) where {B <: GridBoundary, H <: HaloType, D <: KernelAbstractions.GPU, S <: GPUStrategy}
        new{B, H, D, S}(Boundary, Halo, Backend, Strategy)
    end

    function CartesianGrid(Boundary::B, Halo::H, Backend::D) where {B <: GridBoundary, H <: HaloType, D <: KernelAbstractions.CPU}
        new{B, H, D, BasicCPU}(Boundary, Halo, Backend, BasicCPU())
    end
    
    function CartesianGrid(Boundary::B, Halo::H, Backend::D, Strategy::S) where {B <: GridBoundary, H <: HaloType, D <: KernelAbstractions.CPU, S <: CPUStrategy}
        new{B, H, D, S}(Boundary, Halo, Backend, Strategy)
    end
end
struct TreeGrid <: GridSettings end

# TODO: REMOVE
struct StencilSettings{G <: GridSettings, B <: GridBoundary, D <: KernelAbstractions.Backend}
    gridtype::G
    boundary::B
    backend::D
end
#@enum ArrayAccessPattern RowMajor=1 ColumnMajor=2 MortonCurve=3 


mutable struct StencilDimension
    min::Int64
    max::Int64
end

struct StencilSource
    symbol::Symbol    
    radii::Vector{StencilDimension}
    offsets::Vector{Vector{Int64}}
end

struct StencilEquation
    name::Symbol
    args::Array{Tuple{Symbol, Union{Symbol, Nothing}}}
    input_grids::Array{Symbol}
    output_grids::Array{Symbol}
    body::Expr
    dimensions::Int64
    sources::Dict{Symbol, StencilSource}
end

struct ComposedStencil
    stencils::Array{StencilEquation}
end

struct CompiledStencil
    kernel_pointer
    settings::GridSettings
end

struct TranslationLine
    source::Expr
    dest::Expr
end



@inline Base.@propagate_inbounds function _get_const(grid, index, value)
    if checkbounds(Bool, grid, index)
        return grid[index]
    end
    return value
end

# mod(Index - 1, dimension_size) + 1 -> +-1 required as arrays are 1 indexed
# mod1 implements this behaviour, we also use broadcasting to apply mod element wise
@inline Base.@propagate_inbounds function _get_wrap_around(grid, index)
    if checkbounds(Bool, grid, index)
        return grid[index]
    end    
    return grid[CartesianIndex(mod1.(index.I, size(grid)))]
end

@inline Base.@propagate_inbounds function _get_mirror(grid, index::CartesianIndex{N}) where {N}
    if checkbounds(Bool, grid, index)
        return grid[index]
    end
    
    bounds = size(grid)
    # This block is emitted/unrolled for each dimension d
    new_indices = ntuple(Val(N)) do d
        val = index[d]
        bound = bounds[d]
        
        if val < 1
            return 1 - val
        elseif val > bound
            return 2 * bound + 1 - val
        else
            return val
        end
    end
    new_index = CartesianIndex(new_indices)    

    # Checks if in bounds, may disable for optimisation?
    return grid[new_index]
end

# # If index still not in range after mirroring, repeat mirroring till in range
# @inline @propagate_inbounds function _get_mirror_bounce(grid, index)
#     if checkbounds(Bool, grid, index)
#         return grid[index]
#     end    
#     return 
# end

function _perform_gather(symbol, offset, boundary::DefaultValue{T}) where {T}
    return :($(_get_const)($(symbol), $offset, $(boundary.value)))
end
function _perform_gather(symbol, offset, boundary::WrapAround)
    return :($(_get_wrap_around)($(symbol), $offset))  
end
function _perform_gather(symbol, offset, boundary::Mirror)
    return :($(_get_mirror)($(symbol), $offset))  
end
function _perform_gather(symbol, offset, boundary::B) where {B}
    error("Unknown boundary type!")
end

const _KERNEL_END = Expr(:macrocall, Symbol("@label"), nothing, Symbol("kernel_end"))
const _KERNEL_GOTO = Expr(:macrocall, Symbol("@goto"), nothing, Symbol("kernel_end"))
const _SIZE_SYMBOL = Symbol("ndrange")