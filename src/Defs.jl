const CPUBackend = KernelAbstractions.CPU
# const CUDABackend = CUDA.CUDABackend
# const ROCBackend = AMDGPU.ROCBackend
# const MetalBackend = Metal.MetalBackend
# const oneAPIBackend = oneAPI.oneAPIBackend

abstract type BoundaryType end
struct DefaultValue{T} <: BoundaryType 
    value::T
end
struct WrapAround <: BoundaryType end
struct Mirror <: BoundaryType end
struct MirrorWithBounce <: BoundaryType end

# TODO: change to jsut basic, etc., then do check to block overlapped on cpu for example
abstract type ComputationStrategy end
abstract type CPUStrategy <: ComputationStrategy end
abstract type GPUStrategy <: ComputationStrategy end
struct BasicCPU <: CPUStrategy end
struct SpatialCPU <: CPUStrategy end
struct DiamondCPU <: CPUStrategy end
struct CandyCPU <: CPUStrategy end

struct BasicGPU <: GPUStrategy end
struct GatherGPU <: GPUStrategy end
struct OverlappedGPU <: GPUStrategy 
    timesteps::Int32
end

# May have to change from KA, need to discrimiante bwteen CUDA and ROC etc.
abstract type Structure end
Base.@kwdef struct CartesianGrid <: Structure 
    halo::Bool = true
end
struct TreeGrid <: Structure end

Base.@kwdef struct OptimisationSettings    
    # Shared 
    compiler::ComputationStrategy = BasicCPU()
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false
    tile_size = (32, 32)
    # GPU Only
    block_size = (32, 16)
    warp_size = 32    
    # CPU Only
    thread_count = 4    
end

Base.@kwdef mutable struct StencilDimension
    min::Int64 = 0
    max::Int64 = 0
end

struct StencilSource
    symbol::Symbol    
    radii::Vector{StencilDimension}
    offsets::Vector{Vector{Int64}}
end

struct StencilEquation
    name::Symbol
    args::Array{Tuple{Symbol, Union{Symbol, Nothing}}}
    body::Expr
    input_symbols::Array{Symbol} 
    output_symbols::Array{Symbol}   
    required_outputs::Int64
    first_grid_index::Int64
    sources::Dict{Symbol, StencilSource}   
    dimensions::Int64
end

struct StencilSpecification{G <: Structure, B <: BoundaryType, D <: KernelAbstractions.Backend}
    stencil::StencilEquation
    structure::G
    boundary::B
    device::D
    settings::OptimisationSettings
end
StencilSpecification(s, g, b, d) = StencilSpecification(s, g, b, d, OptimisationSettings())

struct CompiledStencil
    kernel_pointer
    settings::Structure
end

struct TranslationLine
    source::Expr
    dest::Expr
end

const _KERNEL_END = Expr(:macrocall, Symbol("@label"), nothing, Symbol("kernel_end"))
const _KERNEL_GOTO = Expr(:macrocall, Symbol("@goto"), nothing, Symbol("kernel_end"))
const _SIZE_SYMBOL = Symbol("_RANGE")
const _UPPER_BOUND = Symbol("_UPPER")
const _LOWER_BOUND = Symbol("_LOWER")