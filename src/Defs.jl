const CPUBackend = KernelAbstractions.CPU

abstract type BoundaryType end
struct DefaultValue{T} <: BoundaryType 
    value::T
end
struct WrapAround <: BoundaryType end
struct Mirror <: BoundaryType end
struct MirrorWithBounce <: BoundaryType end
struct None <: BoundaryType end

abstract type Structure end
Base.@kwdef struct CartesianGrid <: Structure 
    halo::Bool = true
end
struct TreeGrid <: Structure end

abstract type ComputationSettings end
Base.@kwdef struct BasicSettings <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    # GPU Only
    block_size = (32, 16)
    # CPU Only
    thread_count = 4    
end

Base.@kwdef struct SpatialSettings <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    tile_size = (32, 32)
    thread_count = 4    
end

Base.@kwdef struct DiamondSettings <: ComputationSettings
end

Base.@kwdef struct CandySettings <: ComputationSettings
end

Base.@kwdef struct GatherSettings <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    tile_size = (32, 32)
    block_size = (256)
end

Base.@kwdef struct OverlappedSettings <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    tile_size = (32, 32)
    block_size = (256)
    timesteps::Int64 = 2 
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

struct StencilSpecification{G <: Structure, B <: BoundaryType, D <: KernelAbstractions.Backend, S <: ComputationSettings}
    stencil::StencilEquation
    structure::G
    boundary::B
    device::D
    settings::S
end
StencilSpecification(s, g, b, d) = StencilSpecification(s, g, b, d, BasicSettings())

struct CompiledStencil
    kernel
    settings::StencilSpecification
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